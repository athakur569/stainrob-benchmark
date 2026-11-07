# Sealed timestamp — TCGA-BRCA ER plan

The TCGA-BRCA confirmatory study (ER status) is developed on a separate branch, `tcga`. So that its plan can be shown
to precede its results whatever that branch's visibility, the SHA-256 of the locked plan and of the design files it
fixes is recorded here, on the main branch.

| File (on branch `tcga`) | Commit | Committed | SHA-256 of the file at that commit |
|---|---|---|---|
| `docs/tcga/plan.md` (the locked plan) | `e5ea567603251bb28a2ec732f187cc45a0bca7af` | 2026-11-06T20:43:38-05:00 | `e0019b37ae4c818ab03e40ffcbd8a3811332c8b1a08d7a34dfd35d5ed163337d` |
| `data/tcga_brca/design_er/patients.csv` (each patient's site, pool and role; no label) | `e5ea567603251bb28a2ec732f187cc45a0bca7af` | 2026-11-06T20:43:38-05:00 | `770c3f41ee828ccd39184c8428a5bfd7d154d1daee3ca09ecbb503be5b5c45f9` |
| `data/tcga_brca/design_er/pools.csv` (site → held-out pool) | `e5ea567603251bb28a2ec732f187cc45a0bca7af` | 2026-11-06T20:43:38-05:00 | `784262e0f44004cfb97929104691dcda4160463bf49dff18a06ea69792409616` |
| `data/tcga_brca/design_er/halvings.csv` (the halvings of the in-domain role) | `e5ea567603251bb28a2ec732f187cc45a0bca7af` | 2026-11-06T20:43:38-05:00 | `b5cd41daa9c23a18628123f856ae9361f4de5deb8070d5237eae479db4f50f16` |

Check: `git show <commit>:<path> | sha256sum`. GitHub recorded the push of that commit at 2026-11-07T01:43:39Z. The
design files rebuild byte for byte from `scripts/tcga_design.py` and the label and header tables committed at that
commit. Before the lock, no calibration-role, in-domain or held-out-pool patient had been scored, no pool had been
scored as held out, and no monitor had been run on TCGA; the plan's status paragraph lists what was done. Changes after
the lock are numbered deviations in `docs/tcga/deviations.md` on the same branch, and each is sealed here when it is made.
