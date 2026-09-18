# Gumbel MCTS c_scale 기력 비교 벤치마크 및 분석

이 문서는 Great Kingdom의 Gumbel MCTS Arena 평가에서 `gumbel_c_scale` 파라미터(1.0 vs 0.1)가 실제 모델의 착수 및 기력에 미치는 영향을 실측하고 분석한 결과이다.

---

## 1. 핵심 요약 (Executive Summary)

동일한 최신 KLENT 모델([model.onnx](../model.onnx))을 양측에 두고 `c_scale: 1.0`과 `c_scale: 0.1`의 맞대결 60게임(선후공 30쌍, Paired Seeds)을 실측한 결과:

> **`c_scale: 1.0`이 `c_scale: 0.1`을 상대로 49승 11패 (승률 81.7%)를 기록하며 압도적으로 더 강한 기력을 보였다.**

| 세팅 | 총 전적 | 승률 | Blue(선공) 승률 | Orange(후공) 승률 |
| :--- | :--- | :--- | :--- | :--- |
| **`c_scale = 1.0`** | **49승 11패** | **81.7%** | **28 / 30 (93.3%)** | **21 / 30 (70.0%)** |
| **`c_scale = 0.1`** | **11승 49패** | **18.3%** | 9 / 30 (30.0%) | 2 / 30 (6.7%) |

- 선공 유리 게임 특성에도 불구하고, `c_scale = 1.0`은 후공(Orange)을 잡고도 70.0%의 높은 승률을 달성했다.
- 반면 `c_scale = 0.1`은 선공에서도 30.0%에 그쳤으며, 후공에서는 6.7%(2승 28패)로 완패했다.
- 따라서 향후 Arena 평가 및 실전 모델 배포 시 **`c_scale: 1.0`을 유지하는 것이 기력 확보에 필수적**이다.

---

## 2. 실험 환경 및 세팅

- **대상 모델**: 최신 KLENT 훈련 체크포인트 기반 [model.onnx](../model.onnx) (누적 1.28억 transition 이상 학습된 단일 모델)
- **대국 방식**: 30개 시드 쌍(Paired Seeds), 총 60게임
  - 동일 시드에 대해 선후공을 맞바꾼 2게임씩 진행하여 진영 편차 상쇄
- **탐색 파라미터**:
  - `gumbel_simulations`: 32
  - `gumbel_max_considered_actions`: 8
  - `gumbel_c_visit`: 50.0
  - `opening_gumbel_turns`: 8 (`opening_gumbel_scale`: 1.0)
  - `gumbel_scale`: 0.0
  - `max_turns`: 112
- **비교 변수**:
  - Side A: `gumbel_c_scale = 1.0`
  - Side B: `gumbel_c_scale = 0.1`

---

## 3. 세부 통계 및 분석

### 3.1. 대국 통계
- **평균 대국 길이**: 48.5턴
- **종료 원인 (End Reasons)**:
  - 상대 성 파괴 (Opponent Castle Destroyed): 50판 (83.3%)
  - 연속 패스 판정승 (Consecutive Passes): 8판 (13.3%)
  - 본인 성 파괴 (Own Castle Destroyed): 2판 (3.3%)

대부분의 승부가 성 파괴(전술적 돌파)로 결정되었으며, 적극적인 수읽기를 구사한 `c_scale = 1.0` 측이 상대의 방어 허점을 파고들어 성을 함락시켰다.

### 3.2. MCTS 정책망 일치율 실측치
실제 대국 68개 국면에서 정책망 사전 확률(`log_prior`) 1위 수와 MCTS 탐색 후 최종 선택 수 간의 일치 여부를 측정한 결과:

- **`c_scale = 1.0`**:
  - 정책망 1위 수 유지율: **29.4%**
  - MCTS가 정책 1위 수를 뒤집은(Overrule) 비율: **70.6%**
  - 뒤집었을 때 선택된 순위: 2위뿐 아니라 3위, 4위, 5위, 7위 등 깊은 가치 평가에 기반한 전술 수 채택
- **`c_scale = 0.1`**:
  - 정책망 1위 수 유지율: **55.9%**
  - MCTS가 정책 1위 수를 뒤집은 비율: **44.1%**
  - 뒤집었을 때 선택된 순위: 거의 대부분 2위 수로 한정

---

## 4. 메커니즘 분석: 왜 1.0이 0.1보다 강한가?

Rust 엔진의 Sequential Halving 및 착수 점수 공식([rust/crates/great_kingdom_gumbel/src/selection.rs](../rust/crates/great_kingdom_gumbel/src/selection.rs)):

$$\text{ranking\_score} = \text{log\_prior} + \text{bonus}$$
$$\text{bonus} = (c_{visit} + \max N) \times c_{scale} \times \frac{q - q_{min}}{q_{max} - q_{min}}$$

1. **`c_scale = 0.1`의 한계 (보수적 탐색)**:
   - $c_{visit} = 50.0, c_{scale} = 0.1$일 때 1회 방문 시 보너스 최대치는 약 **$5.1$점**이다.
   - 이로 인해 정책망 1~2위 후보 간의 미세한 순위 변경만 일어날 뿐, 직관을 벗어난 깊은 전술적 수순을 발굴하지 못하고 소극적인 행마에 그치게 된다.
2. **`c_scale = 1.0`의 강점 (적극적 수읽기 돌파)**:
   - 보너스 최대치가 **$51.0$점**으로 확장된다.
   - 단 한 수로 성이 파괴되는 Sudden Death 성격의 Great Kingdom 규칙에서, 직관(1위 수)보다 트리 탐색을 통해 발견된 치명적인 약점 타격 수순(3~5위 수)에 과감히 가중치를 실어 승리를 쟁취한다.
3. **Anchor 상대 승률 하락의 원인 규명**:
   - 과거 Gumbel MCTS 모델인 Anchor(`latest-best.onnx`)를 상대로 `c_scale = 0.1` 적용 시 승률이 13~21%로 떡락했던 원인은, Anchor의 강함 때문이 아니라 **Candidate 모델 자체가 0.1 환경에서 기력이 심각하게 붕괴(18.3%)했기 때문**이다.

---

## 5. 착수 결정 파라미터 주의사항

Rust 엔진([rust/crates/great_kingdom_gumbel/src/policy.rs](../rust/crates/great_kingdom_gumbel/src/policy.rs#L105)) 구조상:

- **`gumbel_c_scale` (`c_scale`)**: 실제 대국 중 착수할 액션을 결정하는 `root_selected_action`에 쓰이는 **유일한 스케일 파라미터**이다.
- **`policy_target_c_scale`**: Self-play 데이터를 수집할 때 학습 라벨(`policy_target`) 분포를 계산하는 데만 쓰이며, **대국 착수에는 일절 관여하지 않는다.**

따라서 Arena 평가 및 기력 튜닝 시에는 반드시 **`gumbel_c_scale` (또는 `--gumbel-c-scale`)을 조절**해야 한다.
