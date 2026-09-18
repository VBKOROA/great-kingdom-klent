# Runpod KLENT 학습 실행 매뉴얼

이 문서는 [KLENT 구현 계획](klent-implementation-plan.md)의 동기식 수집/학습 경로를 실행하는 방법이다.
설정 원본은 [klent-train.yaml](../configs/runpod/klent-train.yaml)이다.
학습 후 `latest.pt`나 snapshot을 제출용 모델로 변환하려면
[KLENT 제출용 ONNX export](klent-submission-export.md)를 참고한다.
KLENT는 **하나의 프로세스에서 수집 → 학습 → 체크포인트·ONNX 공개를 반복**한다.

## 1. 환경 준비

- RTX 3090 24GB, AMD EPYC 7C13. 실제 할당 CPU/RAM도 확인한다.
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (+Jupyter).
- Python은 `.venv`를 사용한다. 아래 명령은 Bash와 저장소 루트를 기준으로 한다.
- 저장소와 학습 결과는 Pod에서 확인한 영속 볼륨 안에 둔다. 아래에서는 저장소 위치를
  `/workspace/great-kingdom-ai`로 가정하며 실제 경로가 다르면 변경한다.

최신 KLENT 코드가 포함된 저장소를 준비한 뒤:

```bash
cd /workspace/great-kingdom-ai
git log -1 --oneline
bash scripts/setup_runpod.sh
source .venv/bin/activate
nvidia-smi
```

[설치 스크립트](../scripts/setup_runpod.sh)는 다음을 수행한다.

- `--system-site-packages` venv로 이미지의 PyTorch/CUDA를 유지한다.
- 개발·ONNX 의존성을 설치하고 CUDA 라이브러리 경로를 venv activation에 추가한다.
- Rust 확장을 `extension-module,onnx-cuda` 기능과 release 모드로 빌드한다.
- Rust/Python 테스트를 실행한다. 설치·빌드·테스트 중 오류가 나면 해결 후 진행한다.

Runpod에서는 로컬용 `pip install -e '.[dev,ai]'` 대신 위 스크립트를 사용한다.
새 터미널에서도 `source .venv/bin/activate`를 실행해야 Rust ONNX의 CUDA
라이브러리 경로가 적용된다.

```bash
python - <<'PY'
import torch
import great_kingdom_core as core
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'
assert hasattr(core, 'KlentZeroSearchBatch'), 'Rebuild the Rust extension'
print('GPU:', torch.cuda.get_device_name(0))
PY
```

이 확인은 PyTorch CUDA와 확장 존재 확인이다. Rust CUDA 추론의 실제 동작은 다음 단계에서 검증한다.

## 2. 작은 규모로 FP32와 FP16 각각 2회 실행

원본 YAML을 복사해 사전 실행용 설정 두 개를 만든다. 두 실험은 별도 디렉터리를 사용한다.
모델은 실제 `strong_attn_klent`를 유지하고 수집량·배치만 줄인다.

```bash
mkdir -p data/runpod/klent-configs
python - <<'PY'
from pathlib import Path
import yaml

base = yaml.safe_load(Path('configs/runpod/klent-train.yaml').read_text())
for precision in ('fp32', 'fp16'):
    config = dict(base)
    config.update(
        work_dir=f'data/runpod/klent-smoke-{precision}',
        min_transitions=128,
        max_games_per_iteration=128,
        rust_self_play_batch_size=16,
        batch_size=32,
        fit_epochs=1,
        amp=precision == 'fp16',
        onnx_precision=precision,
        check_onnx_parity=precision == 'fp32',
    )
    path = Path(f'data/runpod/klent-configs/smoke-{precision}.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)
PY

set -o pipefail
python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/smoke-fp32.yaml --iterations 2 \
  2>&1 | tee data/runpod/klent-configs/smoke-fp32.log

python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/smoke-fp16.yaml --iterations 2 \
  2>&1 | tee data/runpod/klent-configs/smoke-fp16.log
```

FP32는 Rust CUDA 추론과 학습을 실행하고, 공개 시 PyTorch CPU/ONNX Runtime CPU parity도 검사한다.
FP16은 Rust CUDA FP16 추론과 learner AMP를 실행한다. `check_onnx_parity: false`이므로
**실행 성공만으로 FP16 수치 오차 검증까지 끝난 것은 아니다.** 현재 공통 parity 함수는 CPU
FP32 비교용이며 CUDA FP16 오차 측정은 별도 검증이 필요하다.

확인할 결과:

- 프로세스가 성공 종료하고 마지막 JSON에 iteration 0, 1의 결과가 있다.
- 각 반복의 `transitions`가 최소 수집량 이상이고 `epoch_losses`가 유한하다.
- 새 `epoch_metrics`와 `self_play_metrics`가 유한하고 종료 원인별 대국 수 합이 `games`와 같다.
- `checkpoints/latest.pt`, `onnx/current.json`, version 2의 actor/eval/manifest가 존재한다.
- FP32 manifest의 두 `parity_passed`는 모두 true다. FP16에서는 검사를 생략했으므로 false다.

`--iterations` 모드는 전체 요청 반복이 끝난 뒤 결과 JSON을 출력한다. `--loop` 모드는
각 반복 완료 직후 결과 JSON을 출력한다. 수집·학습 중 상세 진행률이나
단계별 처리량 로그가 없어 한동안 출력이 없어도 곧바로 정지 상태라고 판단하지 않는다.

다른 터미널에서 GPU와 디스크를 관찰한다.

```bash
nvidia-smi -l 5
```

CPU와 GPU를 함께 보려면 `bash scripts/monitor.sh 3`으로 3초마다 확인한다.
`monitor.sh`는 같은 디렉터리의 `monitor_cpu.sh`를 사용한다.
CPU는 현재 cgroup(v1/v2)의 누적 사용 시간 차이를 실제 경과 시간으로 나눈
`vCPU busy`와, 보이는 CPU quota·상위 cgroup quota·`nproc` 중 가장 작은 용량 대비
사용률을 표시한다. 호스트 `/proc/stat`을 Pod 사용량으로 환산하지 않으며,
cgroup 사용 시간을 읽지 못하면 `측정 불가`를 표시한다.
컨테이너 밖의 상위 제한은 보이지 않을 수 있고, telemetry의 집계 구간·분모에 따라
값이 다를 수 있다. RAM은 여전히 `free` 기준이므로 호스트 메모리를 포함할 수 있다.

```bash
du -sh data/runpod/klent-smoke-*
df -h .
```

사전 실행을 다시 처음부터 비교하려면 YAML의 `work_dir`를 새로운 경로로 바꾼다.
같은 경로에서 같은 `--iterations 2`를 다시 실행하면 완료 상태를 재개하므로 추가 학습하지 않는다.

## 3. 본 학습

사전 실행 통과 후 운영 설정을 복사한다. 원본의 `batch_size: 512`,
`rust_self_play_batch_size: 256`은 RTX 3090에서 실측 전 시작값이다.
먼저 이 배치에서도 2회를 검증한다.

```bash
cp configs/runpod/klent-train.yaml data/runpod/klent-configs/train.yaml
set -o pipefail
python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/train.yaml --iterations 2 \
  2>&1 | tee data/runpod/klent-configs/train-2.log
```

장시간 실행은 `tmux` 등 연결 종료 후에도 유지되는 터미널에서 진행한다.
그 터미널에서도 저장소 루트로 이동하고 `.venv`를 활성화한다.
다음 명령은 **중단할 때까지 계속 학습**한다. 완료된 상태가 있으면 거기서 재개한다.

```bash
set -o pipefail
python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/train.yaml --loop \
  2>&1 | tee -a data/runpod/klent-configs/train-loop.log
```

`--iterations`를 생략하는 것만으로는 무한 실행되지 않는다. 아무 옵션도 지정하지 않으면
기존 기본값인 총 1회까지 실행한다. `--loop`와 `--iterations`는 동시에 지정할 수 없다.
총 100회까지만 실행하려면 `--loop` 대신 `--iterations 100`을 사용한다.
loop는 완료 이력을 메모리에 계속 누적하지 않고 반복별로 출력한다.
Ctrl+C로 중단하면 종료 코드 130을 반환한다. 오류가 발생하면 자동 재시도하지 않고 종료하므로
원인을 해결한 뒤 같은 명령으로 재개한다. artifact worker와 snapshot 스크립트는 별도로 중지한다.

운영 기본값은 반복마다 최소 32,768 transition을 완결된 대국으로 수집하고, 해당 버퍼를
한 epoch 학습한다. 이전 반복의 shard가 디스크에 남아 있어도 다음 반복 학습에는 재사용하지 않는다.
같은 `work_dir`에서 학습 프로세스를 여러 개 실행하지 않는다.

수집 상한 `max_games_per_iteration`은 8,192판이다. 이전 기본값 2,048판에서는 평균
대국 길이가 16 transition보다 짧아지면 32,768개를 채우지 못해 중단될 수 있다.
이미 복사한 `data/runpod/klent-configs/train.yaml`은 원본 설정을 수정해도 자동으로 바뀌지
않으므로 해당 파일의 상한도 변경한다. `min_transitions`는 유지하고 같은 `--loop` 명령으로
재개하면 마지막 완료 checkpoint를 복원한 뒤 실패한 반복의 수집부터 다시 시작한다.
`--no-resume`나 checkpoint 삭제는 필요 없다. 상한을 늘려도 목표를 못 채우면 다시 중단한다.
대국 길이가 계속 줄어들면 종료 원인과 pass 빈도, arena 기력을 함께 확인한다.

### 3.1. 반복 JSON 진단 지표

각 반복 결과 JSON에는 기존 필드(`games`, `transitions`, `epoch_losses` 등)와 함께
`metrics_schema_version`, `epoch_metrics`, `self_play_metrics`가 포함된다. 별도 플래그는
필요 없고 `--iterations`와 `--loop` 모두 같은 스키마를 쓴다. 예시는 다음과 같다.

```json
{
  "iteration": 0,
  "epoch_losses": [1.5],
  "metrics_schema_version": 1,
  "epoch_metrics": [
    {
      "epoch": 0,
      "policy_loss": 1.2,
      "q_loss": 0.3,
      "total_loss": 1.5,
      "target_policy_entropy": 1.1,
      "policy_kl": 0.1
    }
  ],
  "self_play_metrics": {
    "mean_game_length": 48.0,
    "pass_count": 1536,
    "pass_rate": 0.041666666666666664,
    "end_reason_counts": {
      "opponent_castle_destroyed": 192,
      "own_castle_destroyed": 0,
      "consecutive_passes": 576
    },
    "end_reason_rates": {
      "opponent_castle_destroyed": 0.25,
      "own_castle_destroyed": 0.0,
      "consecutive_passes": 0.75
    }
  }
}
```

`epoch_metrics`는 각 optimizer 갱신 직전, 즉 그 배치를 학습하기 전에 측정한 값의 epoch
집계다. 고정된 최종 checkpoint의 평가 손실이 아니다. 자연로그(nats)를 쓰고 각 항목은
`sum(sample_weight * 값) / sum(sample_weight)`로 가중 평균한다. `policy_loss`는 저장된
정책 `pi'`의 cross-entropy, `target_policy_entropy`는 `pi'`의 엔트로피,
`policy_kl = policy_loss - target_policy_entropy`다. `q_loss`는 선택한 수의 Q MSE이고
`total_loss = policy_loss + q_loss`다. 정책 손실이 커졌을 때 목표 엔트로피 증가와 KL 증가를
구분해서 볼 수 있다.

기존 `epoch_losses`는 호환을 위해 **배치별 total loss 평균의 비가중 평균**을 그대로 유지한다.
`epoch_metrics[].total_loss`는 같은 per-sample 손실을 sample weight로 가중 집계하므로 마지막
배치 크기가 다르거나 sample weight가 불균일하면 두 값이 다를 수 있다. 이는 버그가 아니라
정의 차이다.

`self_play_metrics`는 수집 직후, 증강 전 완결 대국으로 계산한다. `mean_game_length`는 전체
transition을 대국 수로 나눈 값이고 `pass_rate`는 **전체 착수 중 pass의 비율**이다(대국별
pass 비율의 평균이 아니다). `end_reason_counts`와 `end_reason_rates`는 종료 원인별 대국 수와
비율이며 알려진 코드는 0이어도 항상 출력한다. 알 수 없는 코드는 `unknown_<code>`로 남겨
대국 수와 비율이 합에서 어긋나지 않게 한다. 대국 길이 상한 초과는 정상 종료가 아니라 오류다.

측정값이 유한하지 않으면 JSON으로 출력하지 않고 측정 위치를 포함한 예외를 발생시킨다. 과거
로그나 새 필드가 없는 로그는 해당 값을 미측정으로 간주한다. checkpoint와 shard 형식은
그대로라 기존 checkpoint에서 재개할 수 있고, 지난 반복의 지표를 소급해 채우지 않는다.

## 4. 중단과 재개

- 기본값이 resume다. 같은 YAML과 목표 `--iterations`로 다시 실행한다.
- 완료한 반복의 모델·optimizer·AMP scaler·step을 복원한다.
- 학습 완료 체크포인트는 있지만 ONNX 공개가 실패한 경우 재개 과정에서 공개를 복구한다.
- 수집 또는 fitting 중 중단된 반복은 배치 중간부터 이어지지 않는다. 마지막 완료
  체크포인트부터 그 반복을 다시 수행한다.
- 새 실험은 새로운 `work_dir`를 사용한다. 기존 디렉터리의 `--no-resume` 재사용은
  오래된 산출물과 복구 동작을 혼동하기 쉽다.
- KLENT 계수 변경은 새 실험으로 분리한다. resume는 저장된 KLENT 설정과의 불일치를 거부한다.
- optimizer를 유지하는 resume에서는 LR/weight decay도 체크포인트에서 복원되므로 YAML만
  수정했다고 적용된다고 가정하지 않는다.

모델 파일만 별도 보관하려면 `latest.pt`를 **실행 중 파일과 다른 위치로 복사**한다.
전체 재현·운영 복원을 위해서는 사용한 YAML, 코드 커밋, `checkpoints/run.json`,
체크포인트와 필요한 로그도 함께 보관한다.

## 5. Artifact pruning과 디스크 보관

trainer 자체의 보관 동작은 아래와 같다. 최근 N개 shard·체크포인트·ONNX만 유지하려면 별도
worker를 실행한다. 전체 디스크 용량 기준의 pruning은 지원하지 않는다.

| 설정/산출물 | 현재 동작 |
|---|---|
| `keep_shards: true` (Runpod 기본값) | 모든 반복 shard와 메타데이터 보관 |
| `keep_shards: false` | 현재 반복의 정상 학습·공개·latest 갱신 뒤 해당 shard와 메타데이터 삭제 |
| 이전에 남은 shard / 실패 후 복구한 반복의 shard | 위 옵션으로 소급 정리하지 않음 |
| 반복 체크포인트, `checkpoints/actor-source-*.pt` | worker로 완료분 최근 N개 보관 |
| `onnx/version-*`, `onnx/actor-source-*.onnx` | worker로 완료분 최근 N개 보관 |

따라서 `keep_shards: false`도 전체 디스크 사용량 상한을 보장하지 않는다.
학습 데이터 재사용 정책(매 반복 새 버퍼)과 디스크 보관 정책은 별개다.
진단을 위해 초기 실험은 true로 두고, 저장 공간을 줄이려면 운영 YAML에서 false로 변경한다.

```yaml
keep_shards: false
```

삭제는 학습 타깃으로 사용한 `iterations/iteration-NNNN.npz`와 그 메타데이터에만 적용된다.
체크포인트의 `last_shard`와 결과 JSON에 삭제된 경로가 남는 것은 이 설정에서 정상이다.
과거 산출물을 수동 정리하려면 학습을 멈추고 백업 및 최신 모델·포인터 참조를 확인한다.

### 완료 artifact pruning worker

[prune_klent_artifacts.py](../scripts/prune_klent_artifacts.py)는 현재 실행의 `latest.pt`를
기준으로 학습이 끝난 shard·체크포인트·ONNX만 정리한다. `latest.pt`, `run.json`, ONNX 게시
포인터(`onnx/current.json`)가 가리키는 버전, 진행 중인 반복의 파일은 건드리지 않는다.

- shard: latest의 iteration이 4라면 shard 0~3이 완료 대상이고, 수집·학습 중인 shard 4 이상은
  보존한다.
- `checkpoints/iteration-NNNN.pt`: 완료분 중 최근 `--keep-iterations`개만 남긴다.
- `checkpoints/actor-source-NNNN.pt`: 완료분 중 최근 `--keep-actor-sources`개만 남긴다.
- `onnx/version-NNNNN/`: 완료분 중 최근 `--keep-onnx-versions`개만 남긴다. `current.json`이
  가리키는 버전은 항상 보존하며, manifest가 없거나 손상된 디렉터리는 건드리지 않는다.
- `onnx/actor-source-NNNN.onnx`: 완료분 중 최근 `--keep-onnx-actor-sources`개만 남긴다.
- `keep_shards: true`와 함께 사용하면 최근 shard를 남기며 운영할 수 있다.
- snapshot은 삭제하지 않는다.

먼저 삭제 후보를 확인한다(`--delete`가 없으면 dry-run).

```bash
python -m scripts.prune_klent_artifacts \
  --work-dir data/runpod/klent-strong-attn \
  --keep-shards 2 --keep-iterations 2 --keep-actor-sources 1 \
  --keep-onnx-versions 2 --keep-onnx-actor-sources 1 \
  --min-age-seconds 600
```

별도 터미널에서 10분마다 실제 정리:

```bash
source .venv/bin/activate
python -u -m scripts.prune_klent_artifacts \
  --work-dir data/runpod/klent-strong-attn \
  --keep-shards 2 --keep-iterations 2 --keep-actor-sources 1 \
  --keep-onnx-versions 2 --keep-onnx-actor-sources 1 \
  --min-age-seconds 600 \
  --interval-seconds 600 --loop --delete
```

- `--keep-shards 2`: 유효한 완료 shard 중 최신 2개를 보관한다. 0이면 완료 shard 전체가 후보다.
- `--keep-iterations 2`: 완료된 반복 체크포인트 중 최신 2개를 보관한다. `latest.pt`가 가리키는
  반복 체크포인트는 항상 보존한다.
- `--keep-actor-sources 1`: 완료된 `checkpoints/actor-source-*.pt` 중 최신 1개를 보관한다.
  진행 중인 반복의 actor 원본은 보존한다.
- `--keep-onnx-versions 2`: 완료된 `onnx/version-*` 디렉터리 중 최신 2개를 보관한다. 게시
  포인터가 가리키는 버전은 항상 보존한다.
- `--keep-onnx-actor-sources 1`: 완료된 `onnx/actor-source-*.onnx` 중 최신 1개를 보관한다.
- `--min-age-seconds`: shard·체크포인트·ONNX 모두 해당 시간 이상 지나야 삭제한다.
- `--loop`를 빼면 한 번만 처리한다. Ctrl+C로 worker를 종료한다.
- 최신 checkpoint가 아직 없거나 run ID가 다르면 loop는 해당 회차를 건너뛰고 오류를 기록한다.
- 메타데이터가 없거나 손상된 shard, manifest가 없는 ONNX 버전 디렉터리, 임시 파일, symlink는
  정리하지 않는다.
- snapshot은 삭제하지 않는다. 이 worker만으로 전체 디스크 사용량이 제한되지는 않는다.
- 진행 중인 단일 trainer와 함께 실행할 수 있다. 같은 디렉터리를 새 실행으로 초기화하거나
  백업에서 복원할 때는 먼저 worker도 중지한다.

### KLENT 주기적 checkpoint snapshot

[save_training_snapshots.sh](../scripts/save_training_snapshots.sh)는 KLENT의
`checkpoints/latest.pt`를 기본 원본으로 사용하므로 인자 없이 실행할 수 있다.

```bash
bash scripts/save_training_snapshots.sh
```

저장 디렉터리, 간격, 보관 개수를 바꾸려면 원본, 저장 디렉터리, 저장 간격(초), 최대 보관
개수 순서로 인자를 지정한다. 아래 예시는 약 10분마다 복사하고 최근 48개만 보관한다.

```bash
bash scripts/save_training_snapshots.sh \
  data/runpod/klent-strong-attn/checkpoints/latest.pt \
  data/runpod/klent-strong-attn/checkpoints/snapshots \
  600 48
```

실제 간격에는 복사·안정성 확인 시간이 추가된다. 네 번째 인자 0 또는 생략은 무제한 보관이다.

저장 이름은 스크립트 공통 형식인 `training-latest-YYYYMMDD-HHMMSS.pt`지만 내용은
KLENT checkpoint다. 모델·optimizer·scaler·iteration·run ID가 함께 복사된다.
파일 변경 여부와 관계없이 주기적으로 복사하므로 같은 학습 상태가 여러 snapshot에 들어갈 수 있다.
학습 중간 배치를 저장하는 기능은 아니며, 가장 최근 완료된 checkpoint를 복사한다.
첫 checkpoint가 없으면 재시도 후 해당 회차를 건너뛴다.

학습, artifact worker, snapshot 스크립트를 각각 별도 터미널에서 실행하면 된다.
snapshot 개수 제한은 snapshot 스크립트가 담당하며 artifact worker는 snapshot을 건드리지 않는다.

특정 snapshot으로 되돌릴 때는 새로운 `work_dir`의 `checkpoints/latest.pt`에 복사하고,
그 경로를 지정한 YAML로 기본 resume를 실행한다. 새 경로에는 이전 `run.json`이나 더 최신
iteration checkpoint를 섞지 않는다. 목표 `--iterations`는 snapshot에 저장된 iteration보다
크게 지정해야 추가 학습한다. 원본 snapshot과 당시 YAML은 별도로 보관한다.

## 6. Arena와 matrix 평가

아래 평가는 KLENT와 기존 `strong_attn`을 **같은 Gumbel 탐색 조건**에 연결해 기력을
비교한다. KLENT 학습 Actor의 검색 없는 정책 성능과는 별도 지표다.
저장소 루트에서 `.venv`를 활성화하고 실행한다. GPU 메모리와 처리량을 확보하려면
학습과 평가 시간을 나누고, 평가마다 모델 파일과 설정을 고정한다.
**`.pt` 단일 arena는 6.2절, 폴더 matrix는 6.5절을 사용한다. ONNX export는 필요 없다.**
6.3~6.4절은 ONNX backend를 선택할 때의 대안이다.

### 6.1. 평가 입력과 지원 범위

기존 평가 로더는 KLENT와 기존 `strong_attn`의 `.pt`를 모두 지원한다. checkpoint의
`model_config`와 가중치를 복원하며 optimizer/scheduler는 생성하지 않는다. 기존 모델은
EMA가 저장되어 있으면 기본적으로 EMA를 사용한다. KLENT Q-head 모델은 `forward`에서
합법수 정책의 Q 기댓값을 상태가치로 계산하므로 같은 Gumbel arena에 연결할 수 있다.
가중치 누락/불일치는 엄격하게 검사한다.

| 도구/입력 | 지원 방법 |
|---|---|
| `great-kingdom-evaluate --backend pytorch` | KLENT와 기존 모델 `.pt`를 양쪽에 직접 지정 |
| `run_pt_folder_matrix_arena.py` | 폴더 안의 KLENT/기존 `.pt` 혼합 또는 KLENT끼리 matrix |
| `run_candidate_pairwise_matrix.py`, `find_strongest_candidate.py` | 공통 평가 로더 사용. 임의 이름의 snapshot은 `--candidates`로 명시 |
| `run_gumbel_setting_arena.py` | KLENT `.pt`를 `--checkpoint`에 직접 지정 가능 |
| `great-kingdom-evaluate --backend onnx` | 기존 ONNX 또는 KLENT `eval.onnx` 사용 |

`.pt` 평가에는 기본 backend인 `pytorch`를 사용한다. ONNX backend의 `.pt` 자동 export는
기존 학습 checkpoint용이므로 KLENT는 전용 export로 먼저 `eval.onnx`를 만들어야 한다.
`actor.onnx`는 학습 Actor용 3출력 모델이며 아래 ONNX arena 입력으로 사용하지 않는다.

### 6.2. `.pt` 단일 arena: KLENT vs 기존 모델 또는 KLENT vs KLENT

저장소 루트에서 실행한다. 아래 `/path/to/...`는 실제 checkpoint 경로로 바꾼다.
평가 중 원본이 갱신되어도 입력이 고정되도록 별도 폴더에 복사한다. 이미 고정된 snapshot이
있으면 복사 없이 해당 경로를 `--candidate`, `--best`에 직접 지정해도 된다.

```bash
source .venv/bin/activate
mkdir -p data/runpod/klent-arena-pt-v1/models
cp /path/to/klent-snapshot.pt data/runpod/klent-arena-pt-v1/models/klent.pt
cp /path/to/strong_attn.pt data/runpod/klent-arena-pt-v1/models/strong-attn.pt
```

평가 실행마다 `klent-arena-pt-v1`을 새 이름으로 바꿔 모델과 보고서를 보관한다.
`--candidate`는 승률을 확인할 모델, `--best`는 비교 상대다. `best`라는 이름 때문에
기존 학습 방식이나 특정 프리셋을 요구하지 않는다. **양쪽 모두 KLENT `.pt`여도 된다.**
`--backend pytorch`로 checkpoint를 직접 로드하며 ONNX 변환은 수행하지 않는다.

먼저 32 simulations / 최대 후보 8개로 선후공 한 판씩 실행한다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/klent-arena-pt-v1/models/klent.pt \
  --best data/runpod/klent-arena-pt-v1/models/strong-attn.pt \
  --backend pytorch --config configs/runpod/fast-matrix.yaml \
  --games 2 --batch-size 2 --device cuda \
  --report data/runpod/klent-arena-pt-v1/arena-fast.json
```

GPU 없는 로컬에서는 위 명령의 `--device cuda`만 `--device cpu`로 바꾼다.
2판은 실행 확인과 빠른 선별용이다. 이 설정은 오프닝 노이즈도 0이므로 seed나 판수만
늘려도 같은 전개가 반복될 수 있다.

본 평가는 오프닝 탐색이 있는 `arena.yaml`로 총 200판을 진행한다. 현재 기본 탐색은
64/8이며 첫 8턴 Gumbel scale 1.0, 이후 0.0이다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/klent-arena-pt-v1/models/klent.pt \
  --best data/runpod/klent-arena-pt-v1/models/strong-attn.pt \
  --backend pytorch --config configs/runpod/arena.yaml \
  --games 200 --batch-size 32 --device cuda --seed-start 0 \
  --report data/runpod/klent-arena-pt-v1/arena-200.json

python scripts/analyze_arena_report.py \
  data/runpod/klent-arena-pt-v1/arena-200.json
```

`--games`는 선후공을 합친 총 판수다. 두 YAML의 `paired_seeds: true`에 따라 같은 seed로
선후공을 교대하므로 짝수로 지정한다. `--batch-size`는 동시에 진행할 대국 수이며
총 판수와 별개다. GPU 메모리가 부족하면 `--batch-size`와 `--leaf-batch-size`를 줄인다.

KLENT snapshot끼리 비교하려면 `--best /path/to/other-klent-snapshot.pt`로 바꾸고
새 report 경로를 지정한다. 탐색 조건을 바꾸려면 예를 들어
`--gumbel-simulations 32 --gumbel-max-considered-actions 8`을 추가한다.
이 탐색 옵션은 양쪽 모델에 동일하게 적용된다.

보고서의 candidate 전체 승률과 Blue/Orange별 승률을 함께 확인한다. 분석기는 종료 원인,
대국 길이와 95% Wilson 구간도 출력한다. 선후공 쌍과 반복 전개의 상관 때문에 구간만으로
우열을 단정하지 말고, 최종 후보는 `--seed-start 1000` 등 다른 구간과 새 report 경로로
재평가한다. `summary.promoted`는 승률 문턱 충족 여부이며 최강 모델 보장이 아니다.
위 명령에는 `--promote`가 없으므로 상대 checkpoint를 교체하지 않는다.

Gumbel 탐색 파라미터 중 `gumbel_c_scale`은 실제 착수 결정에 직접 작용하며, 실측 벤치마크상
`1.0`이 `0.1` 대비 81.7%의 압도적인 우세를 보였다. 자세한 분석은
[Gumbel c_scale 벤치마크](gumbel-c-scale-benchmark.md)를 참고한다.

### 6.3. ONNX backend용 고정 모델 준비 (선택)

가장 간단한 방법은 이미 공개된 eval ONNX를 별도 평가 디렉터리에 복사하는 것이다.
아래 `eval-v1`은 평가 실행마다 새 이름을 사용한다. `current.json`은 한 번만 읽고,
그 포인터가 가리키는 버전의 모델과 manifest를 보관한다.

```bash
python - <<'PY'
from pathlib import Path
import shutil

from great_kingdom_ai.klent.publish import load_klent_onnx_pointer

root = Path('data/runpod/klent-strong-attn/eval-v1')
root.mkdir(parents=True, exist_ok=False)
pointer = load_klent_onnx_pointer('data/runpod/klent-strong-attn')
shutil.copy2(pointer.eval_path, root / 'klent.onnx')
shutil.copy2(pointer.manifest_path, root / 'klent-manifest.json')
print('model version:', pointer.model_version, 'run:', pointer.run_id)
PY

# 실제 비교할 기존 모델의 고정 ONNX 경로로 변경한다.
cp /path/to/strong_attn.onnx data/runpod/klent-strong-attn/eval-v1/strong-attn.onnx
```

Runpod 기본 공개 ONNX는 FP16이다. CPU에서는 [제출용 export 절차](klent-submission-export.md)의
FP32 모델을 사용한다. `latest.pt`나 특정 snapshot을 평가할 때도 해당 문서의 원본 복사,
`export_klent_checkpoint_to_onnx(..., kind='eval')`, parity 검증 절차를 따른다.
matrix용 snapshot은 서로 다른 iteration을 골라 각각 별도 이름의 평가용 ONNX로 만든다.
주기적 snapshot에는 같은 학습 상태가 중복될 수 있다.

기존 `strong_attn`이 `.pt`만 있다면 아래처럼 별도로 export한다. `checkpoint_path`는
실제 고정한 기존 checkpoint로 바꾼다. 이 함수에 KLENT checkpoint를 넣지 않는다.

```bash
python - <<'PY'
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx

export_checkpoint_to_onnx(
    '/path/to/strong_attn.pt',
    'data/runpod/klent-strong-attn/eval-v1/strong-attn.onnx',
    device='cpu', precision='fp16', prefer_ema=True,
)
PY
```

기존 모델은 저장된 EMA가 있으면 EMA를 사용한다. EMA/raw 선택, 원본 경로와 해시,
코드 커밋, ONNX 정밀도를 평가 기록에 남기고 두 모델의 정밀도를 맞춘다.
이미 만들어진 `.onnx`에 CLI의 `--onnx-precision`을 주어도 파일이 재변환되지는 않는다.

### 6.4. ONNX 단일 arena: 빠른 확인 → 본 평가

먼저 [fast-matrix.yaml](../configs/runpod/fast-matrix.yaml)로 선후공 한 판씩 확인한다.
현재 기본값은 32 simulations / 최대 후보 8개, 총 2판이며 오프닝을 포함해 노이즈가 없다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/klent-strong-attn/eval-v1/klent.onnx \
  --best data/runpod/klent-strong-attn/eval-v1/strong-attn.onnx \
  --backend onnx --config configs/runpod/fast-matrix.yaml \
  --report data/runpod/klent-strong-attn/eval-v1/fast-arena.json
```

2판은 실행 확인과 빠른 선별용이다. 노이즈가 0이면 seed만 바꾸거나 판수만 늘려도 같은
전개가 반복될 수 있다. 본 평가에서는 [arena.yaml](../configs/runpod/arena.yaml)의
오프닝 탐색을 사용한다. 현재 기본값은 64/8, 첫 8턴 Gumbel scale 1.0, 이후 0.0이다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/klent-strong-attn/eval-v1/klent.onnx \
  --best data/runpod/klent-strong-attn/eval-v1/strong-attn.onnx \
  --backend onnx --config configs/runpod/arena.yaml \
  --games 200 --batch-size 32 --device cuda --seed-start 0 \
  --report data/runpod/klent-strong-attn/eval-v1/arena-200.json

python scripts/analyze_arena_report.py \
  data/runpod/klent-strong-attn/eval-v1/arena-200.json
```

`games`는 양쪽 색을 합친 총 판수이며 짝수로 지정한다. `paired_seeds: true`는 같은
seed에서 candidate의 선후공을 교대한 쌍을 만든다. `batch-size: 32`는 GPU 메모리를
줄이기 위한 시작값이며 처리량 최적값은 아니다. OOM이면 batch size와 `--leaf-batch-size`를
줄인다. 로컬 CPU 확인은 FP32 ONNX에 `--device cpu --games 2 --batch-size 2`를 사용한다.

전체 승률과 함께 Blue/Orange별 승률, 종료 원인, 대국 길이, 분석기가 출력하는 95% Wilson
구간을 확인한다. 선후공 쌍과 반복 전개에는 상관이 있으므로 이 구간만으로 유의성을
단정하지 않는다. 최종 후보는 다른 seed 구간에서도 재평가한다. 예를 들어 위 명령의
`--seed-start 1000`과 새 report 경로를 사용한다.
`summary.promoted`는 설정된 승률 문턱 충족 여부이며 통계적 검증이나 최강 모델 보장이 아니다.
위 명령은 `--promote`를 지정하지 않아 상대 모델 파일을 교체하지 않는다.

### 6.5. `.pt` 폴더 matrix: KLENT와 기존 모델 혼합

평가할 checkpoint만 별도 폴더에 복사한다. 학습 중 갱신되는 `latest.pt` 대신 고정한
snapshot을 사용하고, 파일명은 모델마다 다르게 지정한다. 아래 원본 경로는 실제 파일로
바꾼다. KLENT끼리 비교하려면 기존 모델 대신 다른 KLENT snapshot을 넣으면 된다.

```bash
mkdir -p data/runpod/klent-eval-pt/models
cp /path/to/klent-snapshot-a.pt data/runpod/klent-eval-pt/models/klent-a.pt
cp /path/to/klent-snapshot-b.pt data/runpod/klent-eval-pt/models/klent-b.pt
cp /path/to/strong_attn.pt data/runpod/klent-eval-pt/models/strong-attn.pt

python scripts/run_pt_folder_matrix_arena.py \
  data/runpod/klent-eval-pt/models \
  --arena-config configs/runpod/fast-matrix.yaml \
  --device cuda \
  --output-dir data/runpod/klent-eval-pt/matrix-fast
```

ONNX 변환 없이 모든 `.pt` 조합을 평가한다. 모델이 N개면 N(N−1)/2개 조합이므로
10개 모델 × 조합당 2판은 총 90판이다. 기본 검색은 폴더 바로 아래의 `*.pt`이며
하위 폴더까지 포함하려면 `--recursive`를 사용한다. `--max-checkpoints N`은 파일명
정렬 후 마지막 N개를 고르므로 학습 iteration 순서와 일치하는 이름을 사용한다.

출력 디렉터리에 각 조합의 `pair-*-arena.json`과 `summary.json`이 생성된다.
summary의 `matrix`는 행 모델이 열 모델에 이긴 비율이고, `ranking`에는 평균/최저/최고
상대 승률과 선후공별 집계가 들어간다. 대각선 0.5는 자기 대국 결과가 아닌 표시값이다.
`winner`는 이 집계의 1위이며 통계적으로 검증된 최강 모델을 뜻하지 않는다.

빠른 선별 후 상위 후보만 별도 폴더에 모아 본 평가한다.

```bash
python scripts/run_pt_folder_matrix_arena.py \
  data/runpod/klent-eval-pt/models \
  --arena-config configs/runpod/arena.yaml \
  --device cuda --games 200 --seed-start 0 \
  --output-dir data/runpod/klent-eval-pt/matrix-full
```

`--games`는 조합당 총 판수다. 전체 판수 예산을 정하려면 대신 `--auto-games-total 600`을
사용한다. 조합 수로 나누고 올림하며 paired-seed 조건도 적용하므로 실제 총 판수는 예산보다
커질 수 있다. 두 옵션은 동시에 사용할 수 없다. matrix 스크립트에는 `--batch-size`가
없으므로 메모리를 줄이려면 arena YAML을 복사해 `batch_size: 32` 등으로 조절한 뒤 지정한다.

기존 report가 있으면 기본적으로 재사용한다. 모델 내용, 설정, seed가 바뀌면 새 출력
디렉터리를 사용하거나 `--force`로 다시 실행한다. 캐시가 입력 변경을 자동 검증한다고
가정하지 않는다. 원본 checkpoint, 설정 사본, 코드 커밋을 결과와 함께 보관한다.

단일 모델 쌍만 평가하려면 6.2절의 `.pt` arena 명령을 사용한다.

### 6.6. sim / max_consider 프리셋

| 사용 목적 | 추천 sim ($n$) | 추천 max_consider ($m$) | 선정 이유 |
| :--- | :---: | :---: | :--- |
| **자원 제약 환경 / 초고속 자가대진** | **16** | **4** | $C=2$ 정수 분할 ($4 \times 2 \times 2 = 16$). 4개 후보를 2페이즈로 압축 탐색하여 연산 효율 극대화. |
| **경량 강화학습 (Atari / MinAtar)** | **32** | **8** | $C=1.33$ 또는 $n=24(C=1), 48(C=2)$. 작은 액션 공간에서 $m=8$이면 3페이즈 할빙이 안정적. |
| **범용 표준 훈련 (Board Games / 로봇)** | **64** | **16** | $C=1$ 이론적 완전 분할 ($16 \times 4 = 64$). 탐색 폭(16개)과 안정적인 절반 탈락을 보장하는 최소 최적값. |
| **고성능 훈련 (AlphaZero급 고기력)** | **192 ~ 200** | **16** | $C=3$ 완벽 분할 ($16 \times 4 \times 3 = 192$). 논문에서 검증된 최적의 성능/비용 밸런스. |
| **토너먼트 / 최종 평가 (Evaluation)** | **800** | **32** | $C=5$ 완벽 분할 ($32 \times 5 \times 5 = 800$). 대형 후보군(32개)을 후보당 5회 이상 검증하며 정밀 탐색. |

## 7. 오류 대응과 검증 범위

| 증상 | 확인할 사항 |
|---|---|
| `onnx-cuda feature is required` | 설치 스크립트로 CUDA 기능을 포함한 Rust 확장 재빌드 |
| cuDNN/CUDA 공유 라이브러리 로드 오류 | venv activation, 설치 스크립트의 라이브러리 탐색 결과 |
| CUDA OOM | 학습 중이면 `batch_size`, 수집 중이면 `rust_self_play_batch_size`를 줄여 재검증 |
| `min_transitions` 미달 | `max_games_per_iteration`을 늘리거나 사전 실행 수집량을 낮춤 |
| `max_turns` 초과 | 로그와 대국 상태 확인. 미종료 대국을 정상 타깃으로 저장하지 않는 보호 동작 |
| parity 실패 | FP32 설정·export 결과 확인. 비교 실패를 끄고 정상 검증으로 취급하지 않음 |
| FP16 로드 시 `/Cast_1_cast_to_q_values` 등에서 float16/float 타입 불일치 | FP16 변환 수정이 포함된 코드로 업데이트한 뒤 아래 절차로 재export |

### FP16 ONNX Cast 타입 오류에서 재실행

`Type (tensor(float16)) ... does not match expected type (tensor(float))`는 ONNX 그래프의
타입 불일치다. FP16 변환 과정에서 기존 `Cast(to=FLOAT)`와 내부 타입 표시가 어긋나거나,
외부 출력인 policy/Q를 내부 value 계산에서도 참조하면서 생길 수 있다. GPU 메모리 부족이나
학습 loss 오류와는 다르다. 수정된 exporter는 공유 출력을 분리하고 Cast 타입을 맞추며,
FP16 저장 전에 `onnx.checker.check_model(..., full_check=True)`로 검사한다.

초기 `klent-smoke-fp16/onnx/actor-source-0000.onnx`에서 실패했다면 수정된 코드를 반영한 뒤
같은 명령을 다시 실행한다. 기본 설정에서 actor-source ONNX는 매 수집 시작 시 재export하므로
깨진 파일이나 FP32 성공 결과를 삭제할 필요가 없다. `--no-resume`도 추가하지 않는다.

```bash
source .venv/bin/activate
set -o pipefail
python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/smoke-fp16.yaml --iterations 2 \
  2>&1 | tee data/runpod/klent-configs/smoke-fp16-retry.log
```

별도로 `actor_onnx_path`를 지정했다면 해당 입력 파일은 자동 재export되지 않으므로 같은
원본 checkpoint에서 다시 export한다. 기존에 배포한 ONNX도 코드 수정만으로 바뀌지는 않는다.
CPU에서 FP16 그래프 로드·추론을 검증해도 CUDA FP16/AMP 검증은 별도로 수행한다.

로컬 CPU에서는 관련 테스트 200개와 실제 `strong_attn_klent` + Rust Actor의 2회 수집/학습/
FP32 공개가 통과했다. 이는 Runpod의 기본 배치 크기, CUDA AMP/FP16 안정성, 장시간 메모리,
처리량 또는 기력을 검증한 결과는 아니다.
GPU 검증에서는 최대 VRAM, 유한 loss, 수집 transition 수, 전체 반복 시간을 기록한다.
현재 CLI는 epoch metrics와 self-play 지표(3.1절)는 보고하지만 단계별 시간·최대 VRAM은
자동 보고하지 않는다. 비유한 진단값은 JSON으로 내보내지 않고 예외로 중단한다.
로컬 pytest에서 확인된 ONNX exporter 관련 deprecation warning의 발생 환경, 영향 및
후속 대응은 [Known KLENT pytest Warnings](klent-pytest-warnings.md)에 기록한다.
검색 없는 평가와 Gumbel Arena의 기력 비교는 별도 평가 단계이며 학습 명령만으로 수행되지 않는다.
6절은 PyTorch checkpoint 및 ONNX arena 경로를 사용하는 운영 절차다. 실제 학습 모델의 Runpod 대국 결과나
32/8 대비 32/4의 우위를 검증한 결과를 의미하지 않는다.

## 8. 순수 정책 무탐색 평가

학습된 policy 자체의 기력을 비교하려면 `--action-selection policy`를 사용한다.
이 경로는 PyTorch backend를 지원하며 KLENT 및 기존 checkpoint를 모두 로드한다.
합법수 중 **원본 policy logit이 가장 큰 수**를 선택하고 동점은 작은 action index로
해결한다. 공통 추론에서 Q/value가 계산될 수 있지만 착수에는 사용하지 않는다.
KLENT의 Q 결합 정책 `pi'`나 Gumbel 탐색을 사용하지 않는다.

```bash
great-kingdom-evaluate \
  --candidate /workspace-global/checkpoints/snapshots/training-latest-20260917-100951.pt \
  --best /workspace-global/checkpoints/snapshots/training-latest-20260917-082900.pt \
  --backend pytorch --config configs/runpod/policy-arena.yaml \
  --device cuda \
  --report data/runpod/klent-arena-pt-v1/policy-100951-vs-082900-seed1000.json

python scripts/analyze_arena_report.py \
  data/runpod/klent-arena-pt-v1/policy-100951-vs-082900-seed1000.json
```

- 기본 첫 8턴은 모델과 무관하게 합법수 전체(합법이면 pass 포함)에서 균등 추출한다.
  `--policy-opening-turns N`으로 조절한다. `N=0`이면 시작부터 policy argmax지만
  seed를 바꾸거나 판수를 늘려도 같은 대국이 반복될 수 있다.
- `paired_seeds: true`일 때 같은 seed의 선후공 두 판은 동일한 오프닝을 공유한다.
  모델이나 batch size를 바꿔도 오프닝은 유지된다. 짝수 판수로 비교한다.
  오프닝 중 대국이 종료되면 그 결과도 포함되므로 매우 긴 오프닝은 피한다.
- `seed_start`와 `policy_opening_turns`가 보고서 config에 기록되고 오프닝 착수도
  moves에 포함된다. 프리셋의 seed는 1000이며 새 표본은 `--seed-start 2000` 등으로 만든다.
- policy 모드에서는 `gumbel_simulations: 0`을 허용한다. sim=0만으로 모드가
  자동 변경되지는 않는다. Gumbel 관련 설정은 policy 착수에 영향을 주지 않는다.
- Python에서 게임 상태를 관리하고 활성 게임을 모델별로 묶어 추론한다.
  GPU 없는 로컬에서는 `--device cpu --games 2 --batch-size 2`로 실행할 수 있다.
- 기존 Gumbel arena의 noisy opening과 이 균등 오프닝은 서로 다르다.
  두 평가의 승률 차이를 Q/value만의 효과로 단정하지 않는다. 두 모드 모두
  최신/과거 모델의 상대 성능을 보되, 원인 분리는 동일 오프닝 통제 실험이 추가로 필요하다.
- ONNX backend는 이 모드를 지원하지 않으며 명시적으로 오류를 낸다.

### 기존 work_dir에서 LR override로 재개

학습 프로세스를 Ctrl+C로 중지하고 동일한 YAML에 `--learning-rate`를 추가한다.

```bash
python -u -m great_kingdom_ai.klent.cli \
  --config data/runpod/klent-configs/train.yaml --loop --learning-rate 0.0001
```

이 옵션은 새 학습뿐 아니라 resume의 optimizer 복원 **이후**에도 적용된다.
모델, Adam moment, scaler, iteration, work_dir는 유지하며 모든 parameter group의
LR만 변경한다. 최신 복구 후보가 `iteration-*.pt`여도 적용된다.
옵션을 생략하면 기존대로 checkpoint의 LR을 복원한다.
다음 완료 checkpoint에 변경된 LR이 저장된다. 그 전에 중단했다면 재실행 때도
같은 옵션을 지정해야 한다. YAML 자체는 자동 수정하지 않는다.
YAML에서 지정하려면 `learning_rate: 0.0001`과 `override_learning_rate: true`를 함께 쓴다.
