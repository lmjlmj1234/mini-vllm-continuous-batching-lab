"""Tests for the Continuous Batching Scheduler."""

from mini_vllm import (
    BlockAllocator,
    BlockManager,
    Config,
    RequestQueue,
    SamplingParams,
    Scheduler,
    Sequence,
    SequenceGroup,
    Status,
)


def _make_config(**kwargs) -> Config:
    defaults = dict(max_num_seqs=4, max_num_batched_tokens=16, block_size=4, num_gpu_blocks=8)
    defaults.update(kwargs)
    return Config(**defaults)


def _make_group(rid: str, prompt_len: int = 4, max_new: int = 4) -> SequenceGroup:
    return SequenceGroup(
        request_id=rid,
        prompt="x" * prompt_len,
        sampling_params=SamplingParams(max_tokens=max_new),
        prompt_token_ids=list(range(prompt_len)),
    )


def _simulate_prefill(seq: Sequence) -> None:
    seq.num_generated_tokens = 1
    seq.status = Status.RUNNING


def _make_scheduler(config: Config) -> tuple:
    queue = RequestQueue()
    alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
    mgr = BlockManager(config.block_size, alloc)
    sched = Scheduler(config, mgr, queue)
    return sched, queue, mgr


class TestScheduler:
    def test_admit_waiting_request(self) -> None:
        cfg = _make_config()
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0"))

        result = sched.schedule()
        assert len(result.scheduled_prefill_groups) == 1
        assert result.scheduled_prefill_groups[0].request_id == "r0"
        assert len(result.scheduled_decode_groups) == 0

    def test_prefill_then_decode(self) -> None:
        cfg = _make_config()
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0"))

        r1 = sched.schedule()
        assert len(r1.scheduled_prefill_groups) == 1
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)

        r2 = sched.schedule()
        assert len(r2.scheduled_decode_groups) == 1
        assert r2.scheduled_decode_groups[0].request_id == "r0"

    def test_finished_request_removed(self) -> None:
        cfg = _make_config()
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=2, max_new=2))

        r1 = sched.schedule()
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)

        r2 = sched.schedule()
        seq = r2.scheduled_decode_groups[0].seqs[0]
        seq.num_generated_tokens = 2

        r3 = sched.schedule()
        assert len(r3.finished_groups) == 1
        assert r3.finished_groups[0].request_id == "r0"

    def test_max_num_seqs_limit(self) -> None:
        cfg = _make_config(max_num_seqs=2)
        sched, queue, _ = _make_scheduler(cfg)
        for i in range(4):
            queue.add(_make_group(f"r{i}"))

        result = sched.schedule()
        assert len(result.scheduled_prefill_groups) <= 2

    def test_ondemand_admits_without_allocating_blocks(self) -> None:
        """On-demand: scheduler admits without allocating any blocks."""
        cfg = _make_config(num_gpu_blocks=2, block_size=4)
        sched, queue, mgr = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=4, max_new=8))

        result = sched.schedule()
        # Admitted even though only 2 blocks exist
        assert len(result.scheduled_prefill_groups) == 1
        assert len(result.rejected_groups) == 0
        seq = result.scheduled_prefill_groups[0].seqs[0]
        assert len(mgr.get_block_table(seq.seq_id)) == 0  # no blocks allocated at admission

    def test_sequence_id_format(self) -> None:
        cfg = _make_config()
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("req-0000"))

        result = sched.schedule()
        seq = result.scheduled_prefill_groups[0].seqs[0]
        assert seq.seq_id == "req-0000-seq-0"

    def test_schedule_result_token_counts(self) -> None:
        cfg = _make_config(max_num_batched_tokens=32, num_gpu_blocks=64)
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=4, max_new=2))

        r1 = sched.schedule()
        assert r1.num_prefill_tokens == 4
        assert r1.num_decode_tokens == 0
        assert r1.num_batched_tokens == 4
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)

        r2 = sched.schedule()
        assert r2.num_prefill_tokens == 0
        assert r2.num_decode_tokens == 1
        assert r2.num_batched_tokens == 1

    def test_ignored_when_budget_full(self) -> None:
        cfg = _make_config(max_num_seqs=2, max_num_batched_tokens=8, num_gpu_blocks=64)
        sched, queue, _ = _make_scheduler(cfg)

        queue.add(_make_group("r0", prompt_len=4, max_new=2))
        r1 = sched.schedule()
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)

        queue.add(_make_group("r1", prompt_len=4, max_new=2))
        queue.add(_make_group("r2", prompt_len=4, max_new=2))
        result = sched.schedule()
        assert len(result.scheduled_prefill_groups) == 1
        assert result.scheduled_prefill_groups[0].request_id == "r1"
        assert len(result.ignored_groups) == 1
        assert result.ignored_groups[0].request_id == "r2"

    # ------------------------------------------------------------------
    # Phase 2 new tests
    # ------------------------------------------------------------------

    def test_decode_first(self) -> None:
        """Running decode sequences are scheduled even when prefill
        cannot fit (sequence budget exhausted by running decode)."""
        cfg = _make_config(
            max_num_seqs=1, max_num_batched_tokens=16,
            num_gpu_blocks=64,
        )
        sched, queue, _ = _make_scheduler(cfg)

        queue.add(_make_group("r0", prompt_len=2, max_new=4))
        r1 = sched.schedule()
        assert len(r1.scheduled_prefill_groups) == 1
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)  # r0 now RUNNING

        queue.add(_make_group("r1", prompt_len=4, max_new=2))
        r2 = sched.schedule()

        # r0 should still be in decode (occupy the only seq slot)
        assert len(r2.scheduled_decode_groups) == 1
        assert r2.scheduled_decode_groups[0].request_id == "r0"

        # r1 should be ignored (max_num_seqs=1, r0 already running)
        assert len(r2.scheduled_prefill_groups) == 0
        assert len(r2.ignored_groups) == 1
        assert r2.ignored_groups[0].request_id == "r1"

    def test_chunked_prefill(self) -> None:
        """Long prompt is split across multiple prefill steps."""
        cfg = _make_config(
            max_num_seqs=2, max_num_batched_tokens=16,
            max_num_prefill_tokens=16, max_prefill_chunk_size=4,
            chunked_prefill_enabled=True, num_gpu_blocks=64,
        )
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=12, max_new=2))

        # Step 1: first chunk (4 tokens)
        r1 = sched.schedule()
        assert len(r1.scheduled_prefill_groups) == 1
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        assert r1.num_prefill_tokens == 4
        assert r1.num_decode_tokens == 0
        assert seq.status == Status.PREFILL
        assert seq.prefill_cursor == 0  # scheduler doesn't advance cursor

        # Simulate executor advancing cursor
        seq.prefill_cursor = 4

        # Step 2: second chunk (4 tokens)
        r2 = sched.schedule()
        assert len(r2.scheduled_prefill_groups) == 1
        assert r2.num_prefill_tokens == 4
        assert r2.num_decode_tokens == 0
        seq.prefill_cursor = 8

        # Step 3: third chunk (4 tokens) — prefill completes
        r3 = sched.schedule()
        assert len(r3.scheduled_prefill_groups) == 1
        assert r3.num_prefill_tokens == 4
        seq.prefill_cursor = 12
        _simulate_prefill(seq)  # executor's job: set RUNNING

        # Step 4: now in decode
        r4 = sched.schedule()
        assert len(r4.scheduled_decode_groups) == 1
        assert r4.scheduled_decode_groups[0].request_id == "r0"
        assert r4.num_prefill_tokens == 0
        assert r4.num_decode_tokens == 1

    def test_prefill_not_finished_not_decode(self) -> None:
        """Partially-prefilled sequences should not appear in decode."""
        cfg = _make_config(
            max_num_seqs=2, max_num_batched_tokens=16,
            max_num_prefill_tokens=16, max_prefill_chunk_size=4,
            chunked_prefill_enabled=True, num_gpu_blocks=64,
        )
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=8, max_new=2))

        r1 = sched.schedule()
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        assert seq.status == Status.PREFILL

        # Partial prefill — cursor advanced but not complete
        seq.prefill_cursor = 4
        assert not seq.is_prefill_finished

        r2 = sched.schedule()
        # Still in prefill, not decode
        assert len(r2.scheduled_decode_groups) == 0
        assert len(r2.scheduled_prefill_groups) == 1

    def test_ignored_reasons(self) -> None:
        """Ignored groups should have a reason in ignored_reasons."""
        cfg = _make_config(
            max_num_seqs=1, max_num_batched_tokens=4,
            max_num_prefill_tokens=4, num_gpu_blocks=64,
        )
        sched, queue, _ = _make_scheduler(cfg)
        queue.add(_make_group("r0", prompt_len=2, max_new=4))
        r1 = sched.schedule()
        seq = r1.scheduled_prefill_groups[0].seqs[0]
        _simulate_prefill(seq)

        queue.add(_make_group("r1", prompt_len=4, max_new=2))
        result = sched.schedule()
        assert len(result.ignored_groups) == 1
        assert result.ignored_groups[0].request_id == "r1"
        assert result.ignored_reasons["r1"] != ""
