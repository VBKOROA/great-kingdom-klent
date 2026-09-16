from pathlib import Path
from types import SimpleNamespace

import pytest
from great_kingdom_ai.klent import cli, trainer


def test_loop_parser_is_explicit_and_exclusive() -> None:
    parser = cli.build_parser()
    assert not parser.parse_args(["--config", "train.yaml"]).loop
    assert parser.parse_args(["--config", "train.yaml", "--loop"]).loop
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "train.yaml", "--loop", "--iterations", "2"])


def test_stream_keeps_one_resumed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    starts = []
    calls = []

    def initial(config, *, resume):
        starts.append(resume)
        return SimpleNamespace(iteration=7)

    def iteration(state, config):
        calls.append(state.iteration)
        return SimpleNamespace(iteration=state.iteration + 1), state.iteration

    monkeypatch.setattr(trainer, "_initial_state", initial)
    monkeypatch.setattr(trainer, "run_klent_iteration", iteration)
    stream = trainer.iter_klent_training(trainer.KlentTrainConfig(work_dir=tmp_path))
    assert [next(stream) for _ in range(3)] == [7, 8, 9]
    stream.close()
    assert starts == [True]
    assert calls == [7, 8, 9]


def test_loop_cli_streams_and_handles_interrupt(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["klent", "--config", "train.yaml", "--loop"])
    monkeypatch.setattr(cli, "load_klent_train_config", lambda _: object())

    def stream(config, *, resume):
        assert resume
        yield SimpleNamespace(to_dict=lambda: {"iteration": 4})
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "iter_klent_training", stream)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 130
    output = capsys.readouterr().out
    assert '"event": "klent_iteration"' in output
    assert '"iteration": 4' in output
    assert '"event": "klent_stopped"' in output
