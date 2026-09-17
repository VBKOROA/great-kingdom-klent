"""Container CPU accounting checks; no GPU or privileged cgroup writes needed."""

import os
from pathlib import Path
import subprocess

import pytest


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "monitor_cpu.sh"


def run_shell(command, **env):
    return subprocess.check_output(
        ["bash", "-c", 'source "$HELPER"\n' + command],
        env={**os.environ, "HELPER": str(HELPER), **env},
        text=True,
    ).strip()


def test_usage_uses_elapsed_time_and_container_capacity():
    # Three CPU-seconds over three wall-seconds means one busy vCPU.
    output = run_shell("format_cpu_usage 100 5000000 103 8000000 32")
    assert "3.1% of capacity, 1.00/32.00 vCPU busy" in output


@pytest.mark.parametrize("args", ["100 5 100 6 32", "100 5 103 4 32"])
def test_invalid_counter_deltas_wait(args):
    assert "측정 대기" in run_shell(f"format_cpu_usage {args}")


@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_capacity_respects_parent_quota_and_affinity(tmp_path, kind):
    child = tmp_path / "pod"
    child.mkdir()
    if kind == "v2":
        (child / "cpu.max").write_text("max 100000\n")
        (tmp_path / "cpu.max").write_text("150000 100000\n")
    else:
        for directory, quota in [(child, -1), (tmp_path, 150000)]:
            (directory / "cpu.cfs_quota_us").write_text(str(quota))
            (directory / "cpu.cfs_period_us").write_text("100000")
    command = 'CPU_LIMIT_DIR="$CHILD"; CPU_LIMIT_MOUNT="$ROOT"; cpu_capacity'
    env = dict(CPU_KIND=kind, CHILD=str(child), ROOT=str(tmp_path))
    assert float(run_shell("nproc() { echo 8; }\n" + command, **env)) == 1.5
    assert float(run_shell("nproc() { echo 1; }\n" + command, **env)) == 1


@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_snapshot_normalizes_cpu_time_to_microseconds(tmp_path, kind):
    (tmp_path / "cpu.stat").write_text("usage_usec 1234567\nuser_usec 1000000\n")
    (tmp_path / "cpuacct.usage").write_text("1234567000\n")
    output = run_shell("cpu_snapshot", CPU_KIND=kind, CPU_USAGE_DIR=str(tmp_path))
    assert output.split()[1] == "1234567"


def test_missing_accounting_does_not_fall_back_to_host():
    assert run_shell("cpu_snapshot || echo unavailable", CPU_KIND="unavailable") == "unavailable"
