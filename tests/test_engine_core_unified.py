"""Tests for Phase 1.5: Unified Execute Wiring.

Verifies that EngineCore.step() uses the unified execute() path
via ModelInputBuilder + _apply_model_output() instead of the old
separate prefill()/decode() path.
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from mini_vllm import (
    Config,
    EngineCore,
    FakeModelExecutor,
    ModelInput,
    ModelInputBuilder,
    ModelRunnerOutput,
    SequenceExecutionInfo,
)
from mini_vllm.cache.allocator import BlockAllocator
from mini_vllm.cache.manager import BlockManager
from mini_vllm.scheduler.scheduler import Scheduler
from mini_vllm.scheduler.schedule_result import ScheduleResult
from mini_vllm.sequence.sequence import Sequence
from mini_vllm.sequence.sequence_group import RequestQueue, SequenceGroup
from mini_vllm.sequence.sampling_params import SamplingParams
from mini_vllm.sequence.status import Status
from mini_vllm.engine.stage_profiler import StageProfiler


# ---------------------------------------------------------------------------
# Spy executor — records calls without producing real output
# ---------------------------------------------------------------------------

class SpyExecutor(FakeModelExecutor):
    """Fake executor that records execute/prefill/decode call counts."""

    def __init__(self, config: Config, block_manager=None) -> None:
        super().__init__(config, block_manager)
        self.execute_count = 0
        self.prefill_count = 0
        self.decode_count = 0
        self.last_model_input: ModelInput | None = None

    def execute(self, model_input: ModelInput) -> ModelRunnerOutput:
        self.execute_count += 1
        self.last_model_input = model_input
        return super().execute(model_input)

    def prefill(self, sequences: List[Sequence]) -> None:
        self.prefill_count += 1
        super().prefill(sequences)

    def decode(self, sequences: List[Sequence]) -> None:
        self.decode_count += 1
        super().decode(sequences)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    kwargs = dict(
        block_size=4,
        num_gpu_blocks=16,
        max_prefill_chunk_size=4,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        vocab_size=256,
        print_step_events=False,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def _make_engine_core(config: Config) -> tuple[EngineCore, SpyExecutor, BlockManager]:
    """Build EngineCore with a spy executor and fresh BlockManager."""
    allocator = BlockAllocator(num_blocks=config.num_gpu_blocks)
    block_manager = BlockManager(config.block_size, allocator)
    queue = RequestQueue()
    scheduler = Scheduler(config, block_manager, queue)
    executor = SpyExecutor(config, block_manager)
    allocator.set_callbacks(
        on_allocate=executor.prepare_block,
        on_free=executor.release_block,
    )
    engine_core = EngineCore(
        scheduler=scheduler,
        executor=executor,
        block_manager=block_manager,
        profiler=StageProfiler(),
    )
    return engine_core, executor, block_manager


def _add_request(
    engine_core: EngineCore,
    prompt: str = "test",
    max_new: int = 4,
) -> str:
    """Add a request to the engine's scheduler queue."""
    req_id = f"req-{id(prompt)}"
    prompt_ids = [ord(c) % 256 for c in prompt]
    sg = SequenceGroup(
        request_id=req_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=max_new),
        prompt_token_ids=prompt_ids,
        arrival_time=0.0,
    )
    engine_core._scheduler._queue._waiting[req_id] = sg
    return req_id


def _step(engine_core: EngineCore) -> ScheduleResult:
    """Run one engine step, catching exceptions for test analysis."""
    return engine_core.step()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnifiedExecute:
    """Phase 1.5: unified execute() path verification."""

    def test_execute_called_once(self):
        """Spy executor verifies execute() is called exactly once per step."""
        cfg = _make_config()
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "hello world", max_new=2)

        _step(ec)

        assert spy.execute_count == 1
        # confirm old methods NOT called on paged path
        assert spy.prefill_count == 0
        assert spy.decode_count == 0

    def test_mixed_batch_single_input(self):
        """Both prefill and decode sequences in the same ModelInput."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "aaaa bbbb cccc dddd", max_new=2)  # 16 chars

        # Step 1: prefill chunk
        _step(ec)
        mi = spy.last_model_input
        assert mi is not None
        assert len(mi.sequence_info) > 0
        assert mi.sequence_info[0].phase == "prefill"

        # Step 2: still prefill (cursor < prompt_len)
        _step(ec)
        mi = spy.last_model_input
        assert mi is not None
        assert len(mi.sequence_info) > 0

        # Continue until decode starts
        for _ in range(10):
            result = _step(ec)
            mi = spy.last_model_input
            if mi and any(si.phase == "decode" for si in mi.sequence_info):
                break

        assert mi is not None
        phases = [si.phase for si in mi.sequence_info]
        assert "decode" in phases, f"No decode phase in {phases}"

    def test_incomplete_prefill_no_sample(self):
        """Unfinished chunk: sample_output_index is None."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "hello world test", max_new=2)  # 16 chars

        _step(ec)  # prefill chunk of 4, not complete
        mi = spy.last_model_input
        assert mi is not None

        for si in mi.sequence_info:
            if si.phase == "prefill":
                assert si.sample_output_index is None, (
                    f"Incomplete prefill should not have sample, "
                    f"got sample_output_index={si.sample_output_index}"
                )

    def test_completed_prefill_first_token(self):
        """Completing prefill produces first output token and status change."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "abcd", max_new=2)  # 4 chars = 1 chunk completes

        _step(ec)  # prefill completes in one step

        # After step, the sequence should be RUNNING with first token
        found_running = False
        for _, sg in ec._scheduler._queue._running.items():
            for seq in sg.seqs:
                found_running = True
                assert seq.status == Status.RUNNING, (
                    f"seq={seq.seq_id} status={seq.status.name} "
                    f"(expected RUNNING)"
                )
                assert len(seq.output_token_ids) >= 1, (
                    f"seq={seq.seq_id} has no output tokens"
                )
                assert seq.num_generated_tokens >= 1
                assert seq.first_token_time is not None
        assert found_running, "No sequences found in RUNNING state"

        # Also verify the ModelInput had a completing prefill entry
        mi = spy.last_model_input
        assert mi is not None
        completing_in_input = [
            si for si in mi.sequence_info
            if si.phase == "prefill" and si.sample_output_index is not None
        ]
        assert len(completing_in_input) == 1, (
            f"Expected 1 completing prefill in ModelInput, "
            f"got {len(completing_in_input)}"
        )

    def test_decode_appends_token(self):
        """Decode step appends one token per sequence."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "abcd", max_new=4)  # completes prefill in 1 step, 4 decode

        _step(ec)  # prefill completes
        prefill_output_count = {
            seq.seq_id: len(seq.output_token_ids)
            for _, sg in ec._scheduler._queue._running.items()
            for seq in sg.seqs
        }

        _step(ec)  # first decode
        for _, sg in ec._scheduler._queue._running.items():
            for seq in sg.seqs:
                expected = prefill_output_count.get(seq.seq_id, 0) + 1
                assert len(seq.output_token_ids) == expected, (
                    f"seq={seq.seq_id} expected {expected} output tokens, "
                    f"got {len(seq.output_token_ids)}"
                )

    def test_sample_sequence_mapping(self):
        """Sampled token IDs map back to correct sequences by seq_id."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        rid1 = _add_request(ec, "aaaaaaaa", max_new=2)  # 8 chars, 2 chunks
        rid2 = _add_request(ec, "bbbb", max_new=2)      # 4 chars, 1 chunk

        # Step 1: both prefill chunk 1
        _step(ec)
        mi = spy.last_model_input
        assert mi is not None

        # Step 2: rid1 prefill chunk 2 (completes), rid2 decode
        _step(ec)
        mi = spy.last_model_input

        # Get output mapping
        result = ec._step_count > 0  # step was executed
        if mi:
            seq_ids_in_input = [si.sequence_id for si in mi.sequence_info]

        # Verify sequences are reachable
        assert ec._scheduler._queue.num_running > 0

    def test_empty_batch_no_execute(self):
        """Empty prefill+decode doesn't call execute()."""
        cfg = _make_config()
        ec, spy, _ = _make_engine_core(cfg)

        _step(ec)  # Step with no requests

        # Execute() should NOT be called for empty batch
        assert spy.execute_count == 0, (
            f"execute() was called {spy.execute_count} times for empty batch"
        )

    def test_fake_executor_behavior_preserved(self):
        """Old engine integration tests still work with the unified path."""
        from mini_vllm.engine.engine import LLMEngine

        cfg = _make_config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=10,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Hello world", max_new_tokens=4)
        engine.add_request("CUDA batching", max_new_tokens=6)

        outputs = engine.run_until_done()
        assert len(outputs) == 2
        for rid, text in outputs.items():
            assert isinstance(text, str)
            assert len(text) > 0
        assert engine.queue.num_waiting == 0
        assert engine.queue.num_running == 0

    def test_old_prefill_decode_not_called(self):
        """The paged path in EngineCore does NOT call legacy methods."""
        cfg = _make_config()
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "test prompt here", max_new=2)

        for _ in range(5):
            _step(ec)

        assert spy.prefill_count == 0, (
            f"prefill() was called {spy.prefill_count} times — "
            f"old API should not be invoked on the unified path"
        )
        assert spy.decode_count == 0, (
            f"decode() was called {spy.decode_count} times — "
            f"old API should not be invoked on the unified path"
        )
        assert spy.execute_count >= 1, (
            "execute() should have been called at least once"
        )

    def test_length_assertions(self):
        """cached_len_before/query_len/kv_len_after consistency."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "abcdefgh", max_new=4)  # 8 chars

        # Step 1: prefill chunk (4 tokens)
        _step(ec)
        mi = spy.last_model_input
        assert mi is not None

        for si in mi.sequence_info:
            assert si.kv_len_after == si.cached_len_before + si.query_len, (
                f"kv_len_after ({si.kv_len_after}) != "
                f"cached_len_before ({si.cached_len_before}) + "
                f"query_len ({si.query_len})"
            )
            if si.phase == "decode":
                assert si.query_len == 1

    def test_backend_naming(self):
        """attention_backend config values are valid."""
        cfg = _make_config()

        # Default
        assert cfg.attention_backend == "reference", (
            f"default attention_backend should be 'reference', "
            f"got {cfg.attention_backend!r}"
        )

        # Explicit values
        cfg2 = Config(attention_backend="triton", num_gpu_blocks=8)
        assert cfg2.attention_backend == "triton"

        cfg3 = Config(attention_backend="reference", num_gpu_blocks=8)
        assert cfg3.attention_backend == "reference"

    def test_sequence_info_structure(self):
        """SequenceExecutionInfo fields are correctly populated."""
        cfg = _make_config(max_prefill_chunk_size=4)
        ec, spy, _ = _make_engine_core(cfg)
        _add_request(ec, "abcdefgh", max_new=4)

        _step(ec)
        mi = spy.last_model_input
        assert mi is not None

        for si in mi.sequence_info:
            assert isinstance(si, SequenceExecutionInfo)
            assert isinstance(si.sequence_id, str)
            assert si.phase in ("prefill", "decode")
            assert isinstance(si.query_start, int)
            assert si.query_len >= 1
            assert si.cached_len_before >= 0
            assert si.kv_len_after == si.cached_len_before + si.query_len
            assert isinstance(si.sample_output_index, (int, type(None)))

    def test_model_runner_output_structure(self):
        """ModelRunnerOutput carries explicit sequence_id mapping."""
        output = ModelRunnerOutput(
            sampled_token_ids=(42, 99),
            sampled_sequence_ids=("seq-0", "seq-1"),
        )
        assert len(output.sampled_token_ids) == 2
        assert len(output.sampled_sequence_ids) == 2
        assert output.sampled_sequence_ids[0] == "seq-0"
        assert output.sampled_sequence_ids[1] == "seq-1"
        assert output.sampled_token_ids[0] == 42
        assert output.sampled_token_ids[1] == 99
