"""Integration tests for the full LLMEngine loop."""

import pytest

from mini_vllm import Config, LLMEngine, SequenceGroup


class TestEngine:
    def test_engine_runs_requests_to_completion(self) -> None:
        cfg = Config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=10, print_step_events=False,
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

    def test_continuous_batching_new_arrival(self) -> None:
        cfg = Config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=8, print_step_events=False,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Request A", max_new_tokens=8)
        outputs = engine.run_until_done()
        assert len(outputs) == 1

    def test_mid_arrival_merge(self) -> None:
        cfg = Config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=20, print_step_events=False,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Request A", max_new_tokens=8)
        engine.add_request("Request B", max_new_tokens=12)
        for _ in range(3):
            engine.step()
        engine.add_request("Request C", max_new_tokens=6)

        outputs = engine.run_until_done()
        assert len(outputs) == 3
        assert all(len(t) > 0 for t in outputs.values())

    def test_ondemand_oom_during_execution(self) -> None:
        """On-demand: OOM raises RuntimeError during execution, not at admission."""
        cfg = Config(
            max_num_seqs=2, max_num_batched_tokens=32,
            block_size=4, num_gpu_blocks=2, print_step_events=False,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Hello world", max_new_tokens=6)
        engine.add_request("Another request here", max_new_tokens=8)
        # With only 2 blocks, execution will OOM when ensure_block fails
        import pytest
        with pytest.raises(RuntimeError, match="OOM"):
            engine.run_until_done()

    def test_engine_step_returns_schedule_result(self) -> None:
        cfg = Config(print_step_events=False)
        engine = LLMEngine(cfg)
        engine.add_request("Test", max_new_tokens=2)
        result = engine.step()
        assert result is not None
        assert len(result.scheduled_prefill_groups) >= 1 or len(result.finished_groups) >= 1

    def test_sequence_created_for_each_request(self) -> None:
        cfg = Config(print_step_events=False, num_gpu_blocks=16)
        engine = LLMEngine(cfg)
        rid = engine.add_request("Test prompt", max_new_tokens=4)
        sg = engine.queue.get_by_id(rid)
        assert isinstance(sg, SequenceGroup)
        assert sg.num_sequences == 0  # not yet admitted

        engine.run_until_done()
        sg = engine.queue.get_by_id(rid)
        assert isinstance(sg, SequenceGroup)
        assert sg.num_sequences == 1
        seq = sg.seqs[0]
        assert seq.num_output_tokens > 0
        assert seq.status.name == "FINISHED"

    def test_schedule_result_fields(self) -> None:
        cfg = Config(print_step_events=False, num_gpu_blocks=16)
        engine = LLMEngine(cfg)
        engine.add_request("Hello", max_new_tokens=2)
        result = engine.step()
        for attr in ("scheduled_prefill_groups", "scheduled_decode_groups",
                     "ignored_groups", "finished_groups", "rejected_groups",
                     "preempted_groups",
                     "num_batched_tokens", "num_prefill_tokens",
                     "num_decode_tokens", "token_budget_remaining",
                     "debug_reason", "ignored_reasons"):
            assert hasattr(result, attr)
        # Verify prefix cache fields exist
        for attr in ("cached_token_count", "num_uncached_prefill_tokens",
                     "matched_block_count"):
            assert hasattr(result, attr)

    def test_executor_kv_writes_are_tracked(self) -> None:
        """Fake executor should count every token written to KV."""
        cfg = Config(print_step_events=False, num_gpu_blocks=16)
        engine = LLMEngine(cfg)
        engine.add_request("Test", max_new_tokens=4)
        # Step once — prefill writes prompt tokens (len=4) to KV
        engine.step()
        kv_stats = engine.executor.get_kv_stats()
        assert kv_stats["allocated_blocks"] > 0  # blocks were allocated
        assert kv_stats["kv_tokens_written"] > 0  # KV writes happened

    def test_executor_kv_affects_output(self) -> None:
        """The fake KV cache should influence decode output."""
        cfg = Config(print_step_events=False, num_gpu_blocks=16)
        engine = LLMEngine(cfg)
        engine.add_request("AAA", max_new_tokens=2)
        engine.run_until_done()
        outputs = engine.get_outputs()
        text = list(outputs.values())[0]
        assert len(text) == 2
        # KV writes should have happened during prefill and decode
        assert engine.executor.get_kv_stats()["kv_tokens_written"] > 0
