# 알려진 KLENT pytest 경고 목록

기록일: 2026-09-18. 상태: 문서화 완료; 익스포터 마이그레이션은 보류됨.

## 확인된 환경 및 결과

로컬 환경 점검은 프로젝트의 `.venv` 환경에서 **PyTorch `2.11.0+cpu`**를 사용하여 수행되었습니다:

```bash
.venv/bin/pytest -q \
  tests/test_klent_metrics.py \
  tests/test_klent_loss.py \
  tests/test_klent_trainer.py \
  tests/test_klent_loop.py
```

결과: **57 passed, 34 warnings**. 이 경고는 `tests/test_klent_trainer.py` 실행 중에 발생한 아래 두 가지 Deprecation 경고가 각각 17회씩 발생한 것입니다.
이 수치는 해당 특정 테스트 실행에 대한 결과이며 전체 리포지토리의 테스트 스위트 전체를 대변하는 것은 아닙니다. 실행 대상 테스트 및 설치된 PyTorch 버전에 따라 달라질 수 있습니다.

이는 로컬 CPU 환경에서의 관찰 결과입니다. 실제 Runpod 가상 환경 내부의 PyTorch 버전과 경고 발생 동작은 여기에서 직접 검증되지 않았습니다.

## 경고 상세 내용

### 1. 레거시 TorchScript 기반 ONNX 익스포터

- 분류(Category): `DeprecationWarning`
- 보고된 위치: `python/great_kingdom_ai/klent/export.py:189`, `torch.onnx.export(...)` 호출부
- 메시지 시작부: `You are using the legacy TorchScript-based ONNX export.`
- 의미: PyTorch는 레거시 익스포터를 지원 중단(deprecate)하고 `torch.export` 기반 익스포터로 전환하는 중입니다. 경고문에는 PyTorch 2.9부터 최신 익스포터가 기본값으로 적용되었다는 내용도 포함되어 있습니다.
- 현재 프로젝트 동작: KLENT는 `dynamo=False`를 명시적으로 전달하므로, 설치된 버전에서 최신 경로가 기본값이어도 레거시 경로를 계속 선택하여 사용합니다.

이 경고가 설치된 PyTorch 버전이 너무 낮거나 익스포트에 실패했음을 의미하지는 않습니다. 선택된 로컬 테스트들은 해당 경고가 발생하면서도 모두 성공적으로 완료되었습니다.

### 2. 지원 중단된 ONNX 로깅 헬퍼 함수

- 분류(Category): `DeprecationWarning`
- 보고된 위치: `.venv/lib/python3.11/site-packages/torch/onnx/utils.py:218`, `setup_onnx_logging(verbose)` 호출부
- 메시지: `The feature will be removed. Please remove usage of this function`
- 의미: 레거시 익스포트 경로 내부에서 PyTorch 내의 지원 중단 예정인 헬퍼 함수를 호출하고 있습니다. 프로젝트 코드가 이 헬퍼를 직접 호출하지는 않습니다.

이 경고를 숨기기 위해 설치된 `site-packages` 내부의 헬퍼 코드를 직접 패치하지 마십시오. 이는 모델이나 학습의 별도 결함으로 취급하기보다는 향후 익스포터 마이그레이션 과정의 일부로 평가해야 합니다.

## 현재 대응 방안

이번 진단 기능 수정 단계에서는 기존 익스포터와 PyTorch 환경을 그대로 유지합니다. 두 경고 중 어느 것도 단독으로 PyTorch 업그레이드나 다운그레이드를 강제하지 않습니다. 이는 손실(loss), 그래디언트 또는 모델 정확성 오류가 아니라 익스포트 API 유지보수에 관련된 사안입니다. 또한 CPU 테스트 통과가 CUDA FP16이나 AMP 호환성을 보장하는 것은 아닙니다.

본 문서에 의해 경고 필터를 추가하거나 의존성을 변경하지 않습니다. 경고는 눈에 보이도록 유지하십시오. 향후 필터링이 불가피해지더라도 모든 deprecation 경고를 무차별 억제하지 말고, 확인된 특정 메시지와 카테고리만을 정확히 매칭해야 합니다. 필터링은 로그 소음을 줄여줄 뿐 지원 중단된 의존성 호출 문제를 해결하지 못합니다.

Runpod 환경 구성은 `--system-site-packages` 가상 환경을 통해 베이스 이미지의 PyTorch/CUDA를 의도적으로 보존합니다. `pyproject.toml`의 선택적 `ai` 의존성 그룹에 `torch>=2.6`이 명시되어 있지만, `scripts/setup_runpod.sh`는 베이스 이미지의 PyTorch가 덮어씌워지지 않도록 `.[dev]`와 ONNX 패키지를 분리하여 설치합니다. 템플릿 이름만으로는 실행 중인 작업에서 실제로 임포트되는 버전을 보장할 수 없습니다. 활성 환경은 다음 명령어로 확인하십시오:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 후속 조치: 익스포터 마이그레이션

`dynamo=False`를 변경하거나 PyTorch 버전을 변경하는 작업은 별도의 검증 절차를 거치는 변경 사항으로 취급해야 합니다. 익스포터 변경은 연산 그래프 구조, 동적 배치(dynamic batch) 차원, 런타임 호환성에 영향을 줄 수 있습니다. 대체 익스포터를 배포하기 전 다음 사항을 수행하십시오:

1. 둘 이상의 배치 크기에서 출력 이름, 텐서 형태, 동적 배치 동작을 포함하여 `actor.onnx`와 `eval.onnx`를 모두 검증합니다.
2. 기존 패리티 검사를 사용하여 동일한 PyTorch 체크포인트와 익스포트된 출력을 비교하고, FP16 변환 경로를 실행해 봅니다.
3. CUDA FP16을 포함하여 Runpod에서 Rust ONNX Runtime 액터를 통한 로딩 및 추론을 검증한 뒤, 별도의 작업 디렉터리에서 AMP를 적용하여 짧은 수집/학습/게시 작업을 수행해 봅니다.
4. PyTorch 버전을 변경할 경우 체크포인트 이어받기(resume)가 정상 작동하는지 확인합니다. 정확한 의존성 버전과 남아있는 경고를 기록합니다.

익스포터 지원 사양이 변경되거나, 익스포트 테스트가 실패하거나, 의존성 업그레이드가 계획될 때 이 항목을 다시 검토하십시오. 확인된 경고문에는 해당 API들의 최종 제거 버전이 명시되어 있지 않으므로, PyTorch 2.9에서 기본값이 변경되었다는 사실만으로 제거 시점을 임의로 단정하지 마십시오.

## 참고 문서

- [Runpod 학습 및 검증 가이드](runpod-klent-training.md)
- [학습 진단 지표 구현 계획](klent-training-metrics-plan.md)
- [PyTorch 공식 ONNX 익스포트 튜토리얼](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
