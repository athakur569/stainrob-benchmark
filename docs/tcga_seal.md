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

## OSF registration (added 7 Nov 2026)

The locked plan is registered on OSF as a Secondary Data Preregistration: **https://osf.io/mz3ph**, registered on
7 Nov 2026 (10:01 as OSF displays it) and pending OSF's 48-hour approval when this entry was written. It is embargoed
until 2028-05-07 at the latest and opens earlier, with the results paper. OSF keeps the uploaded files in an
associated project, https://osf.io/a45gx. The five attached files:

| Attached file | SHA-256 |
|---|---|
| `plan.md` (the locked plan, identical to `docs/tcga/plan.md` above) | `e0019b37ae4c818ab03e40ffcbd8a3811332c8b1a08d7a34dfd35d5ed163337d` |
| `plan.pdf` (a rendering of `plan.md`) | `4d3ef77277b997fd119c1aa214fa0cd3f5a841ed6e80ad7d49516e06e6c28de9` |
| `seal.txt` (the upload set's seal record) | `9d1e27bd3218f766f265efd36392f9a9945f29232220205a57c053c1fd464efe` |
| `design_files.zip` (`design_pools.csv`, `design_patients.csv`, `design_halvings.csv`) | `fcdfebc90ad72f6afe40cf1e5d516c0f8df8651686215e3ebf3585e4d4a7dffd` |
| `slides_and_doses.zip` (`manifest_dx.csv`, `doses.txt`) | `47a03c6874f8a822d5e52a7d5a378a1bada4974ca4e978f5f03d6b29fffc213d` |

OSF allowed five files, so the design files and the slide and dose lists were attached as two archives. Each archive
was extracted on 7 Nov: the three design files are byte-identical to the files sealed above, and every member of both
archives matches the SHA-256 listed in `seal.txt` (`manifest_dx.csv` is `data/tcga_brca/manifest_dx.csv` at the lock
commit, the slide file list; `doses.txt` lists the 17 doses with their committed strengths).

## Deviations and clarifications (on branch `tcga`)

| Entry | File | Commit | Committed | SHA-256 of the file at that commit |
|---|---|---|---|---|
| D1, a clarification: the batch design, fixed before any monitor run, and before any model was trained on the study's roles | `docs/tcga/deviations.md` | `21984c111b35896f5124e0789d8694b3ab48c2a2` | 2026-11-07T10:18:35-05:00 | `26bd224e551a8c64953cfe1db4da58e3a356430a5f507d5553a4ee5b7797181a` |
