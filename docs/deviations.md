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
