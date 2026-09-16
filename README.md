# Great Kingdom AI

Great Kingdom 보드게임을 위한 self-play 학습 실험 저장소입니다.

Rust 규칙 엔진과 Gumbel search를 기반으로 하되, 현재 권장 학습 경로는 **KLENT**입니다.
KLENT는 하나의 동기식 프로세스에서 zero-search self-play 수집 → 학습 → 체크포인트·ONNX
공개를 반복합니다. 학습된 Q-head 네트워크는 기존 고성능 Rust Gumbel MCTS Arena 평가
엔진과 호환되는 2출력 ONNX로 export됩니다.

- 운영 매뉴얼: [docs/runpod-klent-training.md](docs/runpod-klent-training.md)
- 제출용 ONNX export: [docs/klent-submission-export.md](docs/klent-submission-export.md)
- 구현 계획: [docs/klent-implementation-plan.md](docs/klent-implementation-plan.md)

## 구조

```text
.
├── python/great_kingdom_ai/      # KLENT 학습/평가/export 코드
├── rust/great_kingdom_core/      # Rust 규칙 엔진, Gumbel/KLENT search, ONNX evaluator
├── configs/runpod/              # KLENT 학습 및 Arena/matrix 설정
├── scripts/                     # setup, KLENT smoke, artifact 정리, Arena 도구
├── tests/                       # Python 테스트
└── data/                        # 실행 중 생성되는 shard/checkpoint/ONNX 산출물
```

### 모델 구조 및 프리셋

Great Kingdom AI는 전통적인 residual CNN backbone과 기하학적 어텐션 메커니즘을 융합한 하이브리드 네트워크 아키텍처를 지원합니다.

* **CNN Backbone**: 3x3 Conv-BN-ReLU Stem과 다수의 `ResidualBlock`으로 구성된 깨끗한 CNN backbone.
* **Board Self-Attention**: 토큰화된 9x9 보드 공간에 대해 scaled dot-product attention 연산을 수행하는 커스텀 `BoardSelfAttentionBlock`을 Backbone 뒤쪽에 배치 가능.
  - **Full 2D Relative Position Bias**: D4 대칭 궤도별로 공유되는 상대 위치 bias 테이블을 적용.
  - **LayerScale**: Self-Attention 및 FFN 잔차 경로(residual branch)에 초기값 `1e-2` 크기의 LayerScale 파라미터를 적용하여 학습 초기 수렴성 극대화.
* **Spatial Value Head**: 글로벌 풀링(GAP) 방식 대신 $1 \times 1$ Conv와 Flatten-Linear 구조를 채택하여 보드 공간 내 밀집 배치에 강건하게 대응.
* **Action Value (Q) Head**: KLENT용 프리셋은 `action_value_head=True`로 82-action Q 값을 함께 출력하고, 합법수 정책 기댓값을 상태가치로 사용합니다.

KLENT 학습 프리셋은 `strong_attn_klent`(운영 기본값)와 `small_klent`(smoke/테스트)입니다.
기존 state-value 프리셋(`strong_attn`, `strong_clean` 등)은 Arena 비교 baseline으로 유지됩니다.

## 환경

로컬 개발은 CPU 노트북 기준입니다. 규칙, 데이터 구조, 작은 smoke run, 테스트를 확인하는
용도입니다.

본격 학습은 다음 Runpod 환경을 기준으로 합니다.

- GPU: RTX 3090 24GB
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Python은 venv 사용

## 로컬 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ai]'
```

Rust 규칙 엔진을 Python 확장으로 빌드합니다.

```bash
cd rust/great_kingdom_core
../../.venv/bin/python -m maturin develop
cd ../..
```

기본 검증:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

## KLENT 학습

설정 원본은 [configs/runpod/klent-train.yaml](configs/runpod/klent-train.yaml)입니다.
전체 운영 절차와 배치 선정 근거는 [Runpod KLENT 매뉴얼](docs/runpod-klent-training.md)을
따릅니다.

빠른 CPU smoke:

```bash
python -m scripts.run_klent_smoke \
  --work-dir data/test-smoke --iterations 1 \
  --max-games 4 --min-transitions 4
```

반복 학습 (`--loop`는 중단할 때까지 계속하며 완료 상태에서 재개):

```bash
python -m great_kingdom_ai.klent.cli \
  --config configs/runpod/klent-train.yaml --loop
```

콘솔 entry point:

```bash
great-kingdom-klent --config configs/runpod/klent-train.yaml --loop
```

주기적 checkpoint snapshot과 artifact pruning worker는 별도 터미널에서 실행합니다.

```bash
bash scripts/save_training_snapshots.sh

python -m scripts.prune_klent_artifacts \
  --work-dir data/runpod/klent-strong-attn \
  --loop --delete
```

## Runpod 설치

Runpod에서는 기본 PyTorch/CUDA 설치를 유지하기 위해 setup script를 사용합니다.

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
```

스크립트는 venv 생성, 개발 의존성 설치, CUDA 확인, Rust PyO3 확장 빌드, Rust/Python 테스트를
수행합니다. 학습 중 CPU/RAM/GPU 상태는 아래로 관찰합니다.

```bash
./scripts/monitor.sh 10
```

## 평가와 플레이

Arena 평가는 KLENT와 기존 baseline, 또는 KLENT snapshot끼리 **같은 Gumbel 탐색 조건**에서
기력을 비교합니다. `.pt`는 `--backend pytorch`, ONNX는 `--backend onnx`를 사용합니다.
자세한 절차는 [매뉴얼 6절](docs/runpod-klent-training.md)을 참고합니다.

단일 `.pt` arena (KLENT vs baseline):

```bash
great-kingdom-evaluate \
  --candidate data/runpod/klent-strong-attn/checkpoints/snapshots/klent.pt \
  --best data/runpod/klent-strong-attn/checkpoints/snapshots/baseline.pt \
  --backend pytorch --config configs/runpod/arena.yaml \
  --games 200 --batch-size 32 --device cuda --seed-start 0 \
  --report data/runpod/klent-strong-attn/reports/arena-200.json

python scripts/analyze_arena_report.py \
  data/runpod/klent-strong-attn/reports/arena-200.json
```

폴더 matrix (ONNX 변환 없이 모든 `.pt` 조합):

```bash
python scripts/run_pt_folder_matrix_arena.py \
  data/runpod/klent-eval-pt/models \
  --arena-config configs/runpod/fast-matrix.yaml \
  --device cuda \
  --output-dir data/runpod/klent-eval-pt/matrix-fast
```

직접 대국과 수순 재생:

```bash
great-kingdom-play \
  --model-checkpoint data/runpod/klent-strong-attn/checkpoints/latest.pt \
  --human-player blue --device cpu --model-simulations 64

great-kingdom-play --replay-actions '20,68,77'
```

CLI 입력:

- `A1`부터 `I9`: 해당 좌표에 현재 플레이어 성 놓기
- `5 5`: 행/열 숫자로 착수
- `p` 또는 `pass`: 패스
- `l` 또는 `legal`: 현재 합법 수 출력
- `b` 또는 `board`: 보드 다시 출력
- `i <0-81>`: raw action index 입력
- `q` 또는 `quit`: 종료

## ONNX export

KLENT `latest.pt`/snapshot을 기존 Gumbel 엔진용 2출력 `eval.onnx`로 변환하고 parity를
검증하는 절차는 [docs/klent-submission-export.md](docs/klent-submission-export.md)에
정리되어 있습니다. 학습 중에는 trainer가 `actor.onnx`(3출력, zero-search actor용)와
`eval.onnx`(2출력)를 자동 공개합니다.

기존 state-value checkpoint를 직접 export하려면:

```bash
great-kingdom-export-onnx \
  --checkpoint /path/to/strong_attn.pt \
  --output /path/to/strong_attn.onnx \
  --precision fp16 --check-parity
```

CPU serving용 selective QDQ S8S8 quantization은
`great_kingdom_ai.onnx_quantization.quantize_onnx_s8s8_qdq`로 수행하며, 실제 replay
feature를 calibration data로 사용하는 것을 권장합니다.

## 개발 원칙

- 테스트하기 쉬운 구조를 유지합니다.
- 수정하기 쉬운 작은 모듈로 나눕니다.
- 로컬 CPU 환경에서는 빠른 검증을, Runpod GPU 환경에서는 본격 학습을 수행합니다.
- 규칙 엔진 동작이 의심될 때는 수동 CLI와 규칙 문서를 확인합니다.
