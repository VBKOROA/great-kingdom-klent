# KLENT latest.pt → 제출용 ONNX

이 문서는 KLENT로 학습한 `latest.pt`를 **기존 Gumbel 엔진에서 사용하는 2출력 ONNX**로
변환하고 검증하는 방법이다. 학습·snapshot 운영은 [Runpod 매뉴얼](runpod-klent-training.md)을 참고한다.
제출 플랫폼의 압축 형식·파일명·실행 프로그램 규약은 별도이며, 여기서는 이 저장소 엔진의
모델 입출력 계약을 기준으로 한다.

## 1. 제출할 모델과 출력 계약

- 학습 원본: `data/runpod/klent-strong-attn/checkpoints/latest.pt`
- snapshot을 고르려면 아래 `source`만 해당 `.pt` 경로로 바꾼다.
- export 종류: `kind="eval"`
- 기본 정밀도: CPU에서도 검증·실행할 수 있는 FP32

| 이름 | 타입·형태 | 의미 |
|---|---|---|
| 입력 `features` | float32 `[B, 11, 9, 9]` | 저장소와 동일한 feature planes |
| 출력 `policy_logits` | float32 `[B, 82]` | 81개 위치 + pass의 원본 정책 로짓 |
| 출력 `value` | float32 `[B]` | 합법수 정책의 Q 기댓값 `Σ π(a|s) Q(s,a)` |

`B`는 가변 batch 크기다. Gumbel 엔진은 정책 로짓과 상태가치를 받는다.
학습 Actor용 `kind="actor"`는 여기에 `q_values`까지 출력하므로 제출 예제에서는 사용하지 않는다.
`value`는 별도 value head를 새로 학습하는 것이 아니라 ONNX 그래프 내부에서 계산한다.

## 2. 원본 고정 → export → 검증

저장소 루트에서 venv를 활성화하고 CLI를 실행한다. Runpod에서는
`scripts/setup_runpod.sh`로 준비한 환경을 사용한다.

```bash
source .venv/bin/activate
python -m great_kingdom_ai.klent.submission_export \
  --checkpoint data/runpod/klent-strong-attn/checkpoints/latest.pt \
  --output-dir data/submissions/klent-v2
```

패키지를 업데이트하여 설치했다면 `great-kingdom-klent-export` 명령으로도 동일하게 실행할 수 있다.
`--output-dir`에는 매번 새로운 후보 디렉터리를 지정한다. 기존 디렉터리는 덮어쓰지 않는다.
아래 예시의 `klent-v2`도 이미 존재한다면 새 이름을 사용한다.

CLI는 원본을 `source.pt`로 고정한 뒤 CPU FP32 `kind="eval"`로 export하고,
ONNX 구조·출력 이름과 PyTorch/ONNX Runtime의 출력 수치·형태를 검증한다.
학습 중 `latest.pt`가 갱신되어도 모든 검증은 고정한 복사본을 사용한다.

기본 검증은 batch `1, 3, 8` × seed `0, 1, 2`의 9개 조합이며,
policy/value 최대 절대오차 허용치는 `1e-4`다. 이는 제출 CLI의 기본값이며
기존 학습·공개 코드의 parity 기본값은 변경하지 않는다. Runpod에서 관측한
FP32 policy 로짓 오차 약 `4e-5`를 고려한 값으로, 모든 모델의 정확성을 보장하는 기준은 아니다.
다음 옵션으로 검증 범위를 조정할 수 있다.

```bash
python -m great_kingdom_ai.klent.submission_export \
  --checkpoint data/submissions/klent-v2/source.pt \
  --output-dir data/submissions/klent-v3 \
  --tolerance 1e-4 --batch-sizes 1 3 8 --seeds 0 1 2
```

실패한 후보의 모델을 그대로 다시 export하려면 위처럼 해당 `source.pt`를 지정한다.
`latest.pt`를 다시 지정하면 학습이 진행된 다른 모델일 수 있다.

parity 실패 시에도 나머지 조합을 모두 검사하고 종료 코드 1을 반환한다.
`source.pt`, `model.pending.onnx`, 실패 결과를 포함한 `export-report.json`을 보관하며
`model.onnx`로 승격하지 않는다. export 자체가 실패했다면 pending 파일은 없거나 불완전할 수 있다.
보고서에는 batch/seed별 오차·허용치, iteration, run ID, 파일 해시, 라이브러리 버전을 기록한다.
parity 검사는 무작위 입력의 수치·형태 비교이므로 실제 대국 평가를 대신하지 않는다.

성공 시:

```text
data/submissions/klent-v2/
├── model.onnx          # Gumbel 엔진에 전달할 FP32 모델
├── source.pt           # 동일 모델의 재export·추적용 원본
└── export-report.json  # iteration, run ID, 해시, 라이브러리 버전, parity 결과
```

현재 export는 기본적으로 하나의 ONNX 파일을 생성한다. 별도 external-data 파일이 생기도록
export 방식을 변경했다면 그 파일도 함께 배포해야 한다. `source.pt`와 보고서는 보관용이며,
실제 제출물 구성은 제출 시스템의 규약에 맞춘다.

## 3. 실제 Rust evaluator 추론 확인

같은 feature 계약으로 실행할 Rust 확장이 설치된 환경에서 확인한다.

```bash
python - <<'PY'
import math
import great_kingdom_core as core

evaluator = core.OnnxEvaluator(
    'data/submissions/klent-v2/model.onnx', device='cpu', max_batch_size=8,
)
request = core.EvalRequest.from_feature_rows([core.GameState().feature_planes()])
logits, values = evaluator.evaluate(request)
assert len(logits) == 1 and len(logits[0]) == 82
assert len(values) == 1
assert all(math.isfinite(x) for x in logits[0]) and math.isfinite(values[0])
print('Rust inference OK; value =', values[0])
PY
```

이후 Gumbel 실행부의 모델 경로를 `model.onnx`로 지정한다. 탐색 횟수·시간 제한 등은 모델
파일에 포함되지 않으므로 실행부에서 설정한다. `latest.pt`는 가장 최근 모델이며 최강 모델을
보장하지 않는다. snapshot끼리 기력 평가 후 후보를 고르는 경우에도 같은 export 절차를 쓴다.

## 4. 제출 환경이 GPU일 때

FP16은 제출 환경에서 CUDA 추론을 지원하고 실제 추론 오차·속도를 확인한 경우 선택한다.
같은 `source.pt`에서 `export_klent_checkpoint_to_onnx(..., kind="eval", precision="fp16")`로
별도 `model.fp16.onnx`를 만들 수 있다. `device="cpu"`는 export를 수행하는 장치이며,
`precision="fp16"`과 별개의 옵션이다. FP16 변환도 외부 입출력은 float32로 유지한다.

위 CPU FP32 parity 결과를 FP16 파일의 검증 결과로 재사용하지 않는다. FP16은 제출 환경의
실제 CUDA evaluator로 로드·추론하고 FP32 대비 정책·가치 오차 및 대국 동작을 확인해야 한다.
학습용으로 이미 공개된 `eval.onnx`를 복사할 수도 있지만 Runpod 기본 공개 정밀도는 FP16이므로,
CPU 제출용은 위 절차대로 원본 체크포인트에서 FP32로 export한다.
