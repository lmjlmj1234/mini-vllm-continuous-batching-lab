"""Tests for Status / SamplingParams / Sequence / SequenceGroup."""

from mini_vllm import SamplingParams, Sequence, SequenceGroup, Status


class TestStatus:
    def test_finished_statuses(self) -> None:
        assert Status.FINISHED is not Status.REJECTED

    def test_status_order(self) -> None:
        assert Status.WAITING.value < Status.PREFILL.value


class TestSamplingParams:
    def test_defaults(self) -> None:
        sp = SamplingParams()
        assert sp.max_tokens == 16
        assert sp.temperature == 1.0
        assert sp.top_p == 1.0
        assert sp.top_k == -1

    def test_custom_max_tokens(self) -> None:
        sp = SamplingParams(max_tokens=64)
        assert sp.max_tokens == 64

    def test_stop_lists(self) -> None:
        sp = SamplingParams(stop_token_ids=[1, 2], stop_strings=["."])
        assert sp.stop_token_ids == [1, 2]
        assert sp.stop_strings == ["."]


class TestSequence:
    def test_initial_state(self) -> None:
        seq = Sequence(
            seq_id="test-seq", group_id="test-group",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(), arrival_time=100.0,
        )
        assert seq.status == Status.WAITING
        assert not seq.finished
        assert seq.prompt_length == 3
        assert seq.num_output_tokens == 0

    def test_status_lifecycle(self) -> None:
        seq = Sequence(
            seq_id="test-seq", group_id="test-group",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(), arrival_time=100.0,
        )
        seq.status = Status.PREFILL
        seq.status = Status.RUNNING
        seq.num_generated_tokens = 1
        seq.status = Status.FINISHED
        assert seq.finished

    def test_rejected_is_finished(self) -> None:
        seq = Sequence(
            seq_id="test-seq", group_id="test-group",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(), arrival_time=100.0,
        )
        seq.status = Status.REJECTED
        assert seq.finished

    def test_to_dict(self) -> None:
        seq = Sequence(
            seq_id="test-seq", group_id="test-group",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(), arrival_time=100.0,
        )
        d = seq.to_dict()
        assert d["seq_id"] == "test-seq"
        assert d["num_prompt_tokens"] == 3
        assert d["status"] == "WAITING"


class TestSequenceGroup:
    def test_create_sequence(self) -> None:
        sg = SequenceGroup(
            request_id="g0", prompt="hello",
            sampling_params=SamplingParams(),
            prompt_token_ids=[1, 2, 3, 4, 5],
        )
        seq = sg.create_sequence("g0-seq-0")
        assert seq.seq_id == "g0-seq-0"
        assert seq.group_id == "g0"
        assert seq.prompt_token_ids == [1, 2, 3, 4, 5]
        assert seq.sampling_params is sg.sampling_params
        assert sg.num_sequences == 1
        assert not sg.is_finished

    def test_is_finished_all_done(self) -> None:
        sg = SequenceGroup(
            request_id="g0", prompt="hi",
            sampling_params=SamplingParams(),
        )
        s1 = sg.create_sequence("s1")
        s2 = sg.create_sequence("s2")
        s1.status = Status.FINISHED
        s2.status = Status.FINISHED
        assert sg.is_finished
        assert sg.num_finished == 2

    def test_is_finished_partial(self) -> None:
        sg = SequenceGroup(
            request_id="g0", prompt="hi",
            sampling_params=SamplingParams(),
        )
        s1 = sg.create_sequence("s1")
        sg.create_sequence("s2")
        s1.status = Status.FINISHED
        assert not sg.is_finished

    def test_empty_group_not_finished(self) -> None:
        sg = SequenceGroup(
            request_id="g0", prompt="hi",
            sampling_params=SamplingParams(),
        )
        assert not sg.is_finished

    def test_get_unfinished_seqs(self) -> None:
        sg = SequenceGroup(
            request_id="g0", prompt="hi",
            sampling_params=SamplingParams(),
        )
        sg.create_sequence("s1")
        sg.create_sequence("s2")
        sg.seqs[0].status = Status.FINISHED
        unfinished = sg.get_unfinished_seqs()
        assert len(unfinished) == 1
        assert unfinished[0].seq_id == "s2"
