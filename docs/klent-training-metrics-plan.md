# KLENT Training Diagnostics Implementation Plan

Status: Implemented and CPU-validated; Runpod GPU acceptance pending (2026-09-18).

## 1. Purpose and Scope

The current `klent_iteration` log reports collection counts, combined `epoch_losses`,
and artifact paths. Since self-play data changes every iteration, an increase in
combined loss cannot distinguish changes in policy targets from increased model
prediction error. Add the following high-priority metrics to the existing JSON log:

1. Policy loss and Q loss.
2. Target policy entropy and KL divergence from the target policy to the model policy.
3. Mean game length, termination reason rates, and pass rate.

Preserve the alternating collection/training procedure in the
[implementation plan](klent-implementation-plan.md) and the fresh iteration buffer
and default single epoch described in the [Runpod guide](runpod-klent-training.md).
This work adds observability without changing the training objective, targets,
optimizer, or collection conditions. Timing metrics, gradient/AMP diagnostics,
automated arena evaluation, dashboards, and automatic stopping criteria are deferred.
These metrics alone cannot establish whether playing strength has improved.

## 2. Existing Code and Available Data

- `python/great_kingdom_ai/klent/loss.py` already computes policy cross-entropy,
  selected-action Q MSE, and combined loss.
- `python/great_kingdom_ai/klent/trainer.py`: `fit_klent_model` currently returns only
  the unweighted mean of batch total losses for each epoch, alongside the step count.
  `KlentIterationSummary.to_dict()` constructs iteration log fields.
- `python/great_kingdom_ai/klent/cli.py`: `--loop` prints one JSON line after each
  completed iteration. Finite runs place the same summaries in an `iterations` array.
- Both Python and Rust actors provide `end_reason` and per-move `action` through
  `TrajectoryEpisode`. Compute game statistics in the shared Python path;
  no Rust interface or shard format extension is required.

## 3. Metric Definitions

### 3.1 Per-Epoch Training Metrics

Reuse the existing forward pass for each training batch. Do not add inference or
post-training reevaluation. These metrics are **epoch aggregates of values measured
immediately before each optimizer update**, rather than fixed evaluation losses for
the final checkpoint. The target is the stored analytical policy `pi'`; the model
policy is `pi_theta`, the softmax over legal actions using the existing loss mask.
Use natural logarithms for information metrics (nats).

| Field | Per-sample definition |
| --- | --- |
| `policy_loss` | `-sum(pi' * log(pi_theta))` |
| `q_loss` | `(Q_theta(s, action) - lambda_return)^2` |
| `target_policy_entropy` | `-sum(pi' * log(pi'))` |
| `policy_kl` | `KL(pi' || pi_theta) = policy_loss - target_policy_entropy` |
| `total_loss` | Current objective: `policy_loss + q_loss` |

Aggregation contract:

- Aggregate new metrics as `sum(sample_weight * metric) / sum(sample_weight)`.
  Weight each batch mean by its sum of sample weights so that a smaller final batch
  contributes correctly.
- Treat zero-probability terms in target entropy as zero. Construct safe logarithm
  inputs first so that even intermediate calculations avoid `0 * -inf`.
  Preserve the existing masking contract for illegal actions.
- Compute diagnostics in FP32 using detach/no-grad. Do not change the backward graph
  or training targets.
- Compute KL as the difference between CE and entropy using identical aggregation.
  Allow small negative values caused by floating-point error within test tolerances;
  do not unconditionally clamp logged values to zero and hide anomalies.
- Preserve the existing unweighted mean of batch means for `epoch_losses` to retain
  compatibility with historical logs. It may differ from the new `total_loss`
  depending on the final batch size or sample weights.
- Emit one entry per epoch. With the same buffer and sample weights, target entropy
  should agree across epochs within numerical tolerance. D4 augmentation also
  preserves entropy.

### 3.2 Collected Game Statistics

Compute these once per iteration immediately after collection, using completed
episodes before augmentation. Transitions include the final move that terminates
the game and any passes.

| Field | Definition / denominator |
| --- | --- |
| `mean_game_length` | Total transitions / completed games |
| `pass_count` | Number of transitions with `action == features.PASS_ACTION` |
| `pass_rate` | pass_count / total transitions |
| `end_reason_counts` | Completed game count for each termination reason |
| `end_reason_rates` | Game count for each reason / completed games |

Use the following names based on the existing engine codes and `docs/rule-spec.md`.
Verify that the Python and Rust code mappings agree during implementation.

| Code | Log key | Meaning |
| --- | --- | --- |
| 1 | `opponent_castle_destroyed` | Opponent's castle destroyed |
| 2 | `own_castle_destroyed` | Own castle destroyed |
| 3 | `consecutive_passes` | Consecutive passes |

Emit known reasons even when their counts are zero. Preserve unknown codes under
`unknown_<code>` keys so that game counts and rates still sum correctly. Exceeding
the game length limit remains an error, not a normal termination category.
Reject empty episode lists or zero transitions explicitly.
The pass rate is the **fraction of all moves that are passes**, not the mean of
per-game pass rates.

## 4. JSON Extension and Compatibility

Keep existing fields and add the following. This example shows only the new fields.

```json
{
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

The example corresponds to 768 games and 36,864 transitions.

- Use the same summary schema for `--loop` and finite runs.
- Emit the metrics with the existing command and YAML; no additional flag is required.
- Treat missing new fields in historical logs as unmeasured values.
- Preserve checkpoint and shard formats so existing checkpoints remain resumable.
  Do not backfill training metrics for past iterations.
- Do not serialize non-finite values as normal diagnostics. Raise an explicit error
  identifying the measurement location if they occur.

## 5. Implementation Structure and Sequence

1. **Separate the metrics module.** Add typed epoch aggregation, pure episode
   statistics functions, and JSON conversion to a new `klent/metrics.py`.
   Avoid concentrating calculation logic in the already large `trainer.py`.
2. **Extend loss diagnostics.** Reuse existing calculations and masks in `loss.py`
   to provide target entropy and KL. Preserve the optimization objective `total`.
3. **Connect the training loop.** Add epoch metrics to an internal training result.
   Inspect existing callers and preserve the public `fit_klent_model` as a thin
   wrapper returning `(epoch_losses, steps)`. Use the detailed result inside the
   trainer without duplicating training logic.
4. **Connect the summary.** Compute game statistics immediately after collection
   and emit them alongside epoch metrics. Check compatibility with existing summary
   construction sites and tests.
5. **Update the operations guide.** Add log examples, definitions, and the aggregation
   difference between `epoch_losses` and the new `total_loss` to
   `runpod-klent-training.md`.

Accumulate only a small number of detached scalars. Aggregate on the device and
retrieve them together at the end of each epoch to avoid extra CPU synchronization
per batch. Do not accumulate GPU tensors or episode histories across iterations;
long-running loops must not exhibit continuously growing memory use.

## 6. Validation Plan

Use the project's `.venv` and small fixtures that require no GPU for local development.

- **Formulas:** Compare CE, entropy, and KL against hand calculations for known
  distributions. Cover identical distributions with KL approximately zero, one-hot
  targets, pass as the only legal action, zero probabilities, and extreme logits.
- **Aggregation:** Compare against a reference calculation over all samples with
  unequal batch sizes and nonuniform sample weights. Verify
  `policy_loss = target_policy_entropy + policy_kl` and
  `total_loss = policy_loss + q_loss` within numerical tolerance.
- **Training equivalence:** Confirm that existing loss and gradients are preserved
  for identical initial parameters and batches.
- **Game statistics:** Cover different game lengths, every termination reason,
  games with and without passes, and unknown termination codes. Reason counts must
  sum to games; mean game length multiplied by games must equal transitions.
  Verify rate denominators and empty-input handling.
- **Output and compatibility:** Use small CPU training runs to check JSON in both
  CLI modes, array lengths with multiple epochs, preservation of existing fields,
  and resumption from existing checkpoints.
- **Regression:** Focus on `test_klent_loss.py`, `test_klent_trainer.py`,
  `test_klent_loop.py`, and a new `test_klent_metrics.py`. Follow existing execution
  requirements for Rust integration checks.
- **Runpod acceptance:** Run a short job in a separate work_dir with the Rust CUDA
  actor and AMP. Verify finite metrics, consistent counts and rates, and one-line
  JSON output. Record local CPU validation and hardware validation separately.

## 7. Completion Criteria and Interpretation

- The existing `python -u -m great_kingdom_ai.klent.cli --config ... --loop` command
  emits every high-priority metric in each iteration's JSON.
- Measurement requires no extra forward pass and preserves existing training,
  resume, and persistence contracts.
- CPU validation passes, and any areas not validated on Runpod are documented.
- When policy loss increases, the log distinguishes increased target entropy from
  increased KL. Q loss can be tracked separately and compared with changes in game
  length, termination reasons, and pass rate.
- Do not introduce a fixed threshold for normal loss. KL and Q loss also measure
  different data each iteration, so these logs alone cannot establish improvement
  or regression in playing strength.
