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

저장소 루트에서 설치된 venv를 활성화한다. Runpod에서는 `scripts/setup_runpod.sh`로
준비한 환경을 사용한다. 아래 경로의 `klent-v1`은 제출 후보별로 새로운 이름을 지정한다.
기존 후보를 덮어쓰지 않도록 이미 존재하는 디렉터리에서는 중단한다.

```bash
source .venv/bin/activate
python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

from great_kingdom_ai.klent.checkpoint import read_klent_checkpoint_metadata
from great_kingdom_ai.klent.export import (
    compare_klent_checkpoint_to_onnx,
    export_klent_checkpoint_to_onnx,
    onnx_output_names,
)
from great_kingdom_ai.replay.persistence import copy_file_atomic

source = Path('data/runpod/klent-strong-attn/checkpoints/latest.pt')
destination = Path('data/submissions/klent-v1')
if not source.is_file():
    raise FileNotFoundError(source)
destination.mkdir(parents=True, exist_ok=False)
checkpoint = destination / 'source.pt'
copy_file_atomic(source, checkpoint)
# 이후에는 고정한 복사본을 사용하므로 학습 중 latest.pt가 갱신돼도 비교 원본은 같다.
candidate = destination / 'model.pending.onnx'
export = export_klent_checkpoint_to_onnx(
    checkpoint, candidate, kind='eval', device='cpu', precision='fp32',
)
onnx.checker.check_model(str(candidate))
if onnx_output_names(candidate) != ['policy_logits', 'value']:
    raise RuntimeError('Unexpected output contract')

checks = []
for batch_size in (1, 3, 8):
    result = compare_klent_checkpoint_to_onnx(
        checkpoint, candidate, kind='eval', batch_size=batch_size,
    )
    checks.append(result.to_json_dict())
    if not result.passed:
        raise RuntimeError(f'Parity failed: {result.to_json_dict()}')

# 모든 검증에 성공했을 때만 최종 파일명을 사용한다.
output = destination / 'model.onnx'
candidate.replace(output)
metadata = read_klent_checkpoint_metadata(checkpoint)
report = {
    'source': str(source), 'checkpoint': str(checkpoint),
    'output': str(output), 'kind': export.kind, 'precision': export.precision,
    'opset': export.opset_version, 'iteration': metadata.iteration,
    'total_steps': metadata.total_steps, 'run_id': metadata.run_id,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
    'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
    'onnx_sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest(),
    'parity_checks': checks,
}
(destination / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
```

검증 실패 시 `model.onnx`는 생성되지 않는다. 원본·설정·오차를 확인한 뒤 새 후보 디렉터리에서
다시 실행한다. parity 검사는 무작위 입력의 수치·형태 비교이므로 실제 대국 평가를 대신하지 않는다.

성공 시:

```text
data/submissions/klent-v1/
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
    'data/submissions/klent-v1/model.onnx', device='cpu', max_batch_size=8,
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
