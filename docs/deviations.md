# Deviations from the analysis plan

The analysis plan (`docs/analysis_plan.md`) was committed in `927e44b` on 2 November 2026 at
11:50 EST, before any test-set result existed. It is not edited afterwards. Every later change,
clarification or addition is recorded here with its date and reason, and reported in the paper.

## D1 · 2026-11-02 · The matched-dose comparison is the guard against the between-family confound

**What changes.** The plan lists the ΔE-matched comparison between families at equal dose as a
secondary analysis (§9.5). It is now reported as the guard against the between-family confound, beside
the pooled comparison (§9.3). Nothing new is computed; only its role and wording change.

**Why.** The within-family dose–response check (§9.2) was meant to guard against a monitor that only
separates kinds of shift (e.g. structural vs colour) without tracking harm. The plan itself notes that
this check cannot discriminate, because dose orders harm almost perfectly inside every family (pilot ρ
0.96–1.0). That leaves the pooled cross-family comparison, which is exactly where the confound lives.
Holding the magnitude fixed removes it. At each common-grid dose (ΔE 2, 4, 6, 8, 10), the families have
the same colour difference but very different harm. In the pilot at ΔE 10 the AUROC drops were blur
0.285, JPEG 0.136, stain 0.041, striping 0.036 and clipped 0.035. So the question becomes: at fixed
magnitude, which measure still orders the families by harm?

**How it is reported.**
- At each of the five common-grid doses: the per-family ΔG (with patient-clustered intervals) and each
  predictor's value, and the order each predictor puts the families in. With five families per dose, this
  is **descriptive**: orderings and values are shown, not correlation coefficients or tests.
- ΔE is constant across the families at a matched dose **by construction**, so it cannot order them. That
  is a property of the design, not an empirical finding, and is stated as such. The informative comparison
  is between the monitor signals and 1 − SSIM, which does vary across families at a fixed ΔE.

**Raised by:** career-ops; recorded by the research chat. No test result had been seen when this was written.

## D2 · 2026-11-02 · Which statistic is "the" signal, and a tractable computation (specified before any result)

**What changes.** The plan lists two statistics per tier (§6) but does not say which is the continuous
signal used in H1–H3, nor how the monitors are computed at this scale. Both are fixed here, before any test
result exists.

- **Primary continuous signal per tier:** Tier 1a and Tier 1b, MMD² between reference and deployment
  features; Tier 2, mean Mahalanobis distance. Secondary: mean Wasserstein distance (Tier 1), mean energy
  (Tier 2).
- **Features:** each tier's features are standardised with the reference's mean and SD. Tier 1b (1,024-d)
  and Tier 2 (2,048-d) are then projected onto the reference's first 50 principal components (fitted on the
  reference only). Tier 1a (23-d) is used as is. This keeps Tier 2's covariance estimable: with more
  dimensions than reference tiles, the monitor would fall back to a diagonal covariance.
- **Tiles per slide:** at most 8 tiles per slide (seed 0) for reference and deployment alike, in every monitor
  computation — the same rule as the real-tile null check (ISS-039). Without it, each condition needs a
  permutation test on a ~9,000 × 9,000 kernel, for 36 doses × 5 folds × 4 conditions.
- **Clusters:** slide ids, for every permutation and bootstrap inside the monitor (ISS-039).

**Why.** Without this, the choice of statistic could be made after seeing which one looks better. Fixing it
first removes that freedom.

**Recorded by:** the research chat. No test result had been seen when this was written.

## D3 · 2026-11-02 · Alarms on the real arm; Tier 2 null check (specified before any monitor result)

**What changes.**
1. For each fold, the alarm decision of each tier and of the AND rule on `test_cross` (α = 0.05) is
   reported beside that fold's ΔG. The plan reports alarm behaviour for the synthetic arm only (§9.6) and
   continuous signals for the real arm (§9.4). A fold whose ΔG interval includes 0 is a real-data
   **harmless shift** — a different laboratory, and for some hospitals a different scanner type, at no
   measurable cost — so an alarm there is a false alarm with respect to harm, and silence is the correct
   output. A fold whose ΔG interval excludes 0 is a real-data harmful shift.
2. The standing null check (§9.7) for Tier 2 uses the E1 seed-42 model of each fold. Tier 2 depends on the
   model, and repeating it for all 60 runs adds nothing to a calibration check.

**Why.** Fold 0's real-arm ΔG (run log) includes 0: a harmless shift on real data, the counterpart of the
synthetic harmless doses. Raised by career-ops; specified here before any monitor statistic on a test tile
had been computed.

## D4 · 2026-11-02 · Which Tier 1 enters the AND rule (specified before any monitor result)

**What changes.** The plan's AND rule (§9.6) combines "Tier 1" and Tier 2 without saying which Tier 1. Fixed
here: the **primary AND rule is Tier 1b (DenseNet-121) AND Tier 2**; Tier 1a AND Tier 2 is reported as
secondary. A tier flags when either of its two statistics is significant at α/2 (MMD², Wasserstein for
Tier 1; Mahalanobis, energy for Tier 2), as implemented in wsidrift with slide clusters.

**Why.** DR-004 and DR-018 make Tier 1b the primary input-level tier; the H3 prediction expects Tier 1a to
over-weight stain, so making it primary for alarms would build the expected failure into the headline
rule. Recorded by the research chat before any monitor statistic on a test tile was computed.

## D5 · 2026-11-02 · How folds are labelled by their ΔG, and how alarms on them are scored (before folds 1–4)

**What changes.** D3 called a fold whose ΔG interval includes 0 a "harmless shift". With 4–7 tumour patients
in `test_cross`, that only means **no harm detected at this sample size**, and the percentile bootstrap may
understate the uncertainty (run log, 2 Nov). Each fold is therefore labelled from its seed-averaged ΔG(AUROC)
point estimate and 95% interval, with these thresholds, fixed before folds 1–4 were scored:

| Label | Rule | Alarm on this fold is scored as |
|---|---|---|
| harm detected | interval lower bound > 0 | detection (silence = miss) |
| no harm detected — clean | point ≤ 0.01 **and** interval upper bound ≤ 0.03 | false alarm (silence = correct) |
| no harm detected — inconclusive | anything else | ambiguous: reported, not counted as false alarm or detection |

Every fold's table row shows its label together with the point estimate and interval, so a reader can see
what the label rests on. The label is an inferential state, never a property of the shift.

**Applied to fold 0 (already scored):** seed-averaged ΔG ≈ −0.019; every seed's point estimate is
negative and every upper bound ≤ +0.009 → "no harm detected — clean"; an alarm there would be a false alarm.

**Raised by:** career-ops. Recorded by the research chat before folds 1–4 had any result.
