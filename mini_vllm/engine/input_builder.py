from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..cache.manager import BlockManager
from ..config import Config
from ..model_runner.base import (
    AttentionGroup,
    AttentionMetadata,
    ModelInput,
    SequenceExecutionInfo,
)
from ..sequence.sequence import Sequence


class ModelInputBuilder:
    """Build ``ModelInput`` from scheduled sequences and ``BlockManager``.

    This class is the SINGLE place where per-step metadata tensors are
    constructed.  ``BlockManager`` provides read-only block table queries;
    ``ModelInputBuilder`` does the serialisation into GPU-ready tensors.

    Responsibilities:
    - Concatenate prefill and decode token IDs into a single tensor
    - Compute absolute positions for every token
    - Compute physical slot mappings via ``BlockManager.get_block_table()``
    - Build padded block-table tensors for GPU kernel input
    - Build ``AttentionGroup`` entries with precise length semantics
    - Compute ``sample_token_indices`` for selective LM head
    """

    def __init__(
        self,
        block_manager: BlockManager,
        config: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self._block_manager = block_manager
        self._config = config
        self._device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> ModelInput:
        """Construct ``ModelInput`` for one engine step.

        This method does NOT modify ``BlockManager`` state — it only
        reads block tables.
        """
        # 1. Collect per-sequence metadata
        prefill_meta = self._collect_prefill_metadata(prefill_seqs)
        decode_meta = self._collect_decode_metadata(decode_seqs)

        # 2. Build token tensors
        all_input_ids: List[int] = []
        all_positions: List[int] = []
        all_slot_mapping: List[int] = []

        sample_indices: List[int] = []
        token_offset = 0

        # Prefill tokens
        for meta in prefill_meta:
            chunk_len = meta.query_len
            for i in range(chunk_len):
                pos = meta.cached_len_before + i
                all_positions.append(pos)
                slot = self._compute_slot(meta.seq, pos)
                all_slot_mapping.append(slot)

            if meta.prefill_completes:
                # Only sample the last token of the completed prefill
                sample_indices.append(token_offset + chunk_len - 1)

            # input_ids for prefill: prompt tokens in this chunk
            start = meta.cached_len_before
            end = start + chunk_len
            all_input_ids.extend(meta.seq.prompt_token_ids[start:end])
            token_offset += chunk_len

        # Decode tokens
        for meta in decode_meta:
            pos = meta.cached_len_before  # absolute position of new token
            all_positions.append(pos)
            slot = self._compute_slot(meta.seq, pos)
            all_slot_mapping.append(slot)

            # Every decode token needs sampling
            sample_indices.append(token_offset)

            # input_id for decode: the last generated token
            all_input_ids.append(meta.seq.output_token_ids[-1])
            token_offset += 1

        # 3. Build block table tensors
        max_blocks = self._compute_max_blocks(prefill_seqs, decode_seqs)
        prefill_block_tables = self._build_block_table_tensor(
            prefill_seqs, max_blocks
        )
        decode_block_tables = self._build_block_table_tensor(
            decode_seqs, max_blocks
        )

        # 4. Build attention groups
        groups = self._build_groups(
            prefill_meta, decode_meta,
            len(prefill_seqs), len(decode_seqs),
        )

        # 5. Build sequence execution info (Phase 1.5)
        seq_info: List[SequenceExecutionInfo] = []
        sample_idx_counter = 0

        for meta in prefill_meta:
            s_idx = None
            if meta.prefill_completes:
                s_idx = sample_idx_counter
                sample_idx_counter += 1
            seq_info.append(SequenceExecutionInfo(
                sequence_id=meta.seq.seq_id,
                phase="prefill",
                query_start=meta.cached_len_before,
                query_len=meta.query_len,
                cached_len_before=meta.cached_len_before,
                kv_len_after=meta.cached_len_before + meta.query_len,
                sample_output_index=s_idx,
            ))

        for meta in decode_meta:
            seq_info.append(SequenceExecutionInfo(
                sequence_id=meta.seq.seq_id,
                phase="decode",
                query_start=meta.cached_len_before,
                query_len=1,
                cached_len_before=meta.cached_len_before,
                kv_len_after=meta.cached_len_before + 1,
                sample_output_index=sample_idx_counter,
            ))
            sample_idx_counter += 1

        # Length invariant assertions
        for si in seq_info:
            assert si.kv_len_after == si.cached_len_before + si.query_len, (
                f"kv_len_after ({si.kv_len_after}) != "
                f"cached_len_before ({si.cached_len_before}) + "
                f"query_len ({si.query_len}) for seq={si.sequence_id}"
            )
            if si.phase == "decode":
                assert si.query_len == 1, (
                    f"decode query_len ({si.query_len}) != 1 for seq={si.sequence_id}"
                )

        # 6. Wrap into tensors
        device = self._device
        num_prefill = sum(m.query_len for m in prefill_meta)
        num_decode = len(decode_meta)

        attn_metadata = AttentionMetadata(
            groups=groups,
            prefill_slot_mapping=torch.tensor(
                all_slot_mapping[:num_prefill], dtype=torch.long, device=device
            ) if num_prefill > 0 else torch.tensor([], dtype=torch.long),
            prefill_block_tables=prefill_block_tables.to(device),
            prefill_positions=torch.tensor(
                all_positions[:num_prefill], dtype=torch.long, device=device
            ) if num_prefill > 0 else torch.tensor([], dtype=torch.long),
            decode_block_tables=decode_block_tables.to(device),
            decode_slot_mapping=torch.tensor(
                all_slot_mapping[num_prefill:], dtype=torch.long, device=device
            ) if num_decode > 0 else torch.tensor([], dtype=torch.long),
            decode_positions=torch.tensor(
                all_positions[num_prefill:], dtype=torch.long, device=device
            ) if num_decode > 0 else torch.tensor([], dtype=torch.long),
            block_size=self._config.block_size,
            num_kv_heads=0,  # filled by ModelRunner
            head_dim=0,      # filled by ModelRunner
        )

        return ModelInput(
            input_ids=torch.tensor(all_input_ids, dtype=torch.long, device=device),
            positions=torch.tensor(all_positions, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(all_slot_mapping, dtype=torch.long, device=device),
            attn_metadata=attn_metadata,
            sample_token_indices=torch.tensor(
                sample_indices, dtype=torch.long, device=device
            ),
            sequence_info=tuple(seq_info),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_slot(self, seq: Sequence, token_position: int) -> int:
        """Compute the physical slot for a token position.

        Uses ``BlockManager.get_block_table()`` as the single truth source.
        Formula: ``physical_slot = block_id * block_size + offset``
        """
        block_size = self._config.block_size
        block_ids = self._block_manager.get_block_table(seq.seq_id)

        logical_idx = token_position // block_size
        block_offset = token_position % block_size

        if logical_idx >= len(block_ids):
            raise RuntimeError(
                f"Block not allocated for seq={seq.seq_id} "
                f"position={token_position} (logical_idx={logical_idx}, "
                f"num_blocks={len(block_ids)})"
            )

        block_id = block_ids[logical_idx]
        return block_id * block_size + block_offset

    def _compute_max_blocks(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> int:
        """Find the maximum number of blocks across all sequences."""
        max_blocks = 0
        for seq in prefill_seqs:
            n = len(self._block_manager.get_block_table(seq.seq_id))
            max_blocks = max(max_blocks, n)
        for seq in decode_seqs:
            n = len(self._block_manager.get_block_table(seq.seq_id))
            max_blocks = max(max_blocks, n)
        return max(1, max_blocks)  # at least 1

    def _build_block_table_tensor(
        self,
        seqs: List[Sequence],
        max_blocks: int,
    ) -> torch.Tensor:
        """Build a padded block table tensor for GPU kernel input.

        Shape: ``[num_seqs, max_blocks]`` with ``-1`` padding for
        unused trailing entries.
        """
        if not seqs:
            return torch.zeros((0, 0), dtype=torch.long)

        rows = []
        for seq in seqs:
            block_ids = self._block_manager.get_block_table(seq.seq_id)
            padded = block_ids + [-1] * (max_blocks - len(block_ids))
            rows.append(padded[:max_blocks])

        return torch.tensor(rows, dtype=torch.long)

    # ------------------------------------------------------------------
    # Metadata collection
    # ------------------------------------------------------------------

    def _collect_prefill_metadata(
        self,
        seqs: List[Sequence],
    ) -> List["_PrefillMeta"]:
        result = []
        chunk_size_cfg = self._config.max_prefill_chunk_size
        for seq in seqs:
            remaining = len(seq.prompt_token_ids) - seq.prefill_cursor
            chunk_size = min(chunk_size_cfg, remaining)
            cached = seq.prefill_cursor
            result.append(_PrefillMeta(
                seq=seq,
                cached_len_before=cached,
                query_len=chunk_size,
                prefill_completes=(cached + chunk_size >= len(seq.prompt_token_ids)),
            ))
        return result

    @staticmethod
    def _collect_decode_metadata(
        seqs: List[Sequence],
    ) -> List["_DecodeMeta"]:
        result = []
        for seq in seqs:
            # Decode invariant: must have at least one generated token as input
            assert seq.num_generated_tokens >= 1, (
                f"Decode sequence {seq.seq_id} has num_generated_tokens=0"
            )
            assert len(seq.output_token_ids) >= 1, (
                f"Decode sequence {seq.seq_id} has empty output_token_ids"
            )
            assert seq.num_generated_tokens == len(seq.output_token_ids), (
                f"Decode sequence {seq.seq_id} num_generated={seq.num_generated_tokens} "
                f"!= len(output_token_ids)={len(seq.output_token_ids)}"
            )
            # cached_len_before = prompt tokens + output tokens already in KV
            # The last output token (output_token_ids[-1]) is the *input* to
            # this decode step — its KV hasn't been written yet.  So subtract 1.
            cached = len(seq.prompt_token_ids) + seq.num_generated_tokens - 1
            result.append(_DecodeMeta(
                seq=seq,
                cached_len_before=cached,
            ))
        return result

    def _build_groups(
        self,
        prefill_meta: List[_PrefillMeta],
        decode_meta: List[_DecodeMeta],
        num_prefill: int,
        num_decode: int,
    ) -> List[AttentionGroup]:
        groups: List[AttentionGroup] = []
        device = self._device

        # Prefill group(s) — currently all go to the GPU prefill path.
        # The "prefill_ref" type is available for test comparison.
        if num_prefill > 0:
            pref_cached = torch.tensor(
                [m.cached_len_before for m in prefill_meta],
                dtype=torch.long, device=device,
            )
            pref_query = torch.tensor(
                [m.query_len for m in prefill_meta],
                dtype=torch.long, device=device,
            )
            pref_after = pref_cached + pref_query
            groups.append(AttentionGroup(
                seq_indices=list(range(num_prefill)),
                attention_type="prefill_gpu",
                cached_len_before=pref_cached,
                query_len=pref_query,
                kv_len_after=pref_after,
            ))

        # Decode group
        if num_decode > 0:
            dec_cached = torch.tensor(
                [m.cached_len_before for m in decode_meta],
                dtype=torch.long, device=device,
            )
            dec_query = torch.ones(num_decode, dtype=torch.long, device=device)
            groups.append(AttentionGroup(
                seq_indices=list(range(num_prefill, num_prefill + num_decode)),
                attention_type="decode_gpu",
                cached_len_before=dec_cached,
                query_len=dec_query,
                kv_len_after=dec_cached + dec_query,
            ))

        return groups


# ---------------------------------------------------------------------------
# Internal metadata helpers (not part of public API)
# ---------------------------------------------------------------------------

class _PrefillMeta:
    """Per-sequence prefill metadata for one step."""
    __slots__ = ("seq", "cached_len_before", "query_len", "prefill_completes")

    def __init__(
        self,
        seq: Sequence,
        cached_len_before: int,
        query_len: int,
        prefill_completes: bool,
    ) -> None:
        self.seq = seq
        self.cached_len_before = cached_len_before
        self.query_len = query_len
        self.prefill_completes = prefill_completes


class _DecodeMeta:
    """Per-sequence decode metadata for one step."""
    __slots__ = ("seq", "cached_len_before")

    def __init__(self, seq: Sequence, cached_len_before: int) -> None:
        self.seq = seq
        self.cached_len_before = cached_len_before
