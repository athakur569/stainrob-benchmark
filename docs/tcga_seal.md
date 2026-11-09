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
| D1 and D2, a clarification: the stream design, fixed before any stream run, and before any model was trained on the study's roles | `docs/tcga/deviations.md` | `990fe878cd24ea5b84e9618a037d9c20fb6de876` | 2026-11-07T10:41:00-05:00 | `3a7b6ffc0b96ac466a502bd83ddd5e6af88876594cbe95da42b02ee53ce73539` |
| D1 to D3, a clarification: the common operating point, H3's display, per-site readings, the permutation unit, refusals and the wsidrift pin, fixed before any monitor run and before any model was trained on the study's roles | `docs/tcga/deviations.md` | `dde788a3514e6c71447ececfac65871218ced693` | 2026-11-07T10:58:21-05:00 | `710ffcacb20c9e471bc87ae4412d2c5e04822c1363c79d617bed6d4bab48d24b` |
| D1 to D4, a clarification: the definitions behind the descriptive readings and the A4 report, fixed before any monitor run and before any model was trained on the study's roles | `docs/tcga/deviations.md` | `bed648047c6f235609e741b8557b0e54472af437` | 2026-11-07T11:13:52-05:00 | `62cfee731d856ed10d598598bd56cecf6932096cf4e535b7a4d1bd38d8d7620e` |
| D1 to D5, a clarification: the open choices in the readings beside the hypotheses (with the same-slide control's weighting named as a departure from the rehearsal's arithmetic), fixed before any monitor run and before any model was trained on the study's roles | `docs/tcga/deviations.md` | `77113ddcb04ed0e6bb8067760f3e8ffb14c4018a` | 2026-11-07T12:06:23-05:00 | `eedc93e7f093822aae5f454026e08811f339a24fc9eb5b869375a29ed395618a` |
| D1 to D6, a clarification: Tier 1a moved to just after training and before any reading, and the rules for any change after the first ER result, fixed while the models were training and before any model was graded | `docs/tcga/deviations.md` | `b43f4fb3b26b9135d7a0739e4c15010d53d2bc34` | 2026-11-07T18:38:07-05:00 | `1e77819d840f479d8cab5f0d58c6d29c2f8dafc1071044e0475261a93a18dd52` |
| D1 to D7, **after a result** (the gate count, 2026-11-08 02:25:54): each arm's run staged so its evaluability read is checked before any H1 output is opened, display only | `docs/tcga/deviations.md` | `333c01c054157a8dba980a465e914e930bf98a15` | 2026-11-08T02:33:15-05:00 | `bbca4c4eaf426d472267eaff8ea6fcaaa09a1ad4962dbe7deaefb54e09a2f2ad` |
| The run's record (D7): the seven operational scripts' last-modified times and SHA-256, the scripts committed unedited beside it | `docs/tcga/run/README.md` | `333c01c054157a8dba980a465e914e930bf98a15` | 2026-11-08T02:33:15-05:00 | `f56dc47b8a8e70f6c79eb9c717959d895c4446c8d34b37a676b447b4a07d41a1` |
| D1 to D8, **after a result**: each arm's dose scoring limited to that arm's own runs, with a completeness check by slide id after scoring | `docs/tcga/deviations.md` | `8fb30e985a5fabd3c08d9b6bdabfc3c894c67ce8` | 2026-11-09T05:36:52-05:00 | `e2aa56dd5f5eda78e4d595bc7eda950814fc2b960449fe9c98c884382353979e` |
| The run's record (D8): er_back.sh changed, score_check.py added | `docs/tcga/run/README.md` | `8fb30e985a5fabd3c08d9b6bdabfc3c894c67ce8` | 2026-11-09T05:36:52-05:00 | `f0aa591b7a833775174844751546bc3ee933153469a265f59a30f739554db1e6` |
| D1 to D9, **after a result**, descriptive only: the exactly paired view of doses that QC empties in part, and the removed slides' ER and type mix | `docs/tcga/deviations.md` | `c01afdd604ec38e6cff2f1f8b23f4199e8d7b289` | 2026-11-09T06:04:24-05:00 | `414f236c0fbec45c087828ff0ad9272bc0e9a121b18b036597638dfcba480792` |
| The run's record (D9): er_back.sh's readings stage calls the paired view | `docs/tcga/run/README.md` | `c01afdd604ec38e6cff2f1f8b23f4199e8d7b289` | 2026-11-09T06:04:24-05:00 | `476fe2b76bbc9661d501fdc36deac08a991eada5a9f2df90d6bb0bdc6e7482f3` |
| D1 to D10, **after a result**: the A3 stage checks that doses caught by QC in full have no graded rows | `docs/tcga/deviations.md` | `537adc94f3dffbb7ebb51ddf073f94fab6f2e77e` | 2026-11-09T06:07:16-05:00 | `176c67f86e9f7d8e39dd286326a4bbb48fcea1d311ff1dcaf4d4fc8f33cfb247` |
| The run's record (D10): a3_summary.py changed | `docs/tcga/run/README.md` | `537adc94f3dffbb7ebb51ddf073f94fab6f2e77e` | 2026-11-09T06:07:16-05:00 | `4c9c9e6c2b596b04b5910e4b590f2dd8bd5a18ad2e5d16a16306c62f395f5b67` |
| D1 to D11, **after a result**: the run wrapper's false failure on a silent step, fixed; the arm resumed at grading | `docs/tcga/deviations.md` | `fe569134399f2fe117d739c3d0634ab5b68b5fef` | 2026-11-09T08:38:49-05:00 | `01455bac135473f0ac54627df769d4a344eb09fb6c08707b59742059d1f1bab5` |
| The run's record (D11): er_back.sh changed; the wrapper as fixed and as it ran | `docs/tcga/run/README.md` | `fe569134399f2fe117d739c3d0634ab5b68b5fef` | 2026-11-09T08:38:49-05:00 | `2c61b1b42ab6b2b81c37a9190f8a58caf2fa6eab1eca2661e8c9e730d392b895` |
| D11's rerun check: the first grading's file hashes, taken before the rerun | `docs/tcga/run/d11_rerun_check.txt` | `fe569134399f2fe117d739c3d0634ab5b68b5fef` | 2026-11-09T08:38:49-05:00 | `64fee83383a587b7bc7633caa06d89fcbfa01cbdfe983d5a14ed08e7b7649a4c` |
| D11's rerun check, completed: all 17 files identical after the rerun | `docs/tcga/run/d11_rerun_check.txt` | `aa0a68610cf643fb5921d36c685019368a5bc77f` | 2026-11-09T08:56:40-05:00 | `59ba55e4d90f8fd1ed6a4f09f9d366d6402b905007f0943132d244ddfbf9a0d4` |
| D1 to D12, **after a result**: every H1 label, the borderline reading's included, reads "not evaluable" on an arm whose H1 is not evaluable, and the A4 report marks that arm's "better than S1" readings as descriptive | `docs/tcga/deviations.md` | `69c792c570a0078be2d7a834a74c0766e1e5aef0` | 2026-11-09T09:46:23-05:00 | `3fc633469e3ea9e3e01432f65a33a97f0b7736a9174219115d4755b053d75203` |
| The run's record (D12): er_back.sh changed; d12_rerun.sh and d12_compare.py added | `docs/tcga/run/README.md` | `69c792c570a0078be2d7a834a74c0766e1e5aef0` | 2026-11-09T09:46:23-05:00 | `9ba0b135fe51c28409c2528c34f986ec50d5ad3df110e43cf8b7e28b31c903a6` |
| D12's rerun check: the ResNet-50 arm's readings and A4 report file hashes, taken before the rerun | `docs/tcga/run/d12_rerun_check.txt` | `69c792c570a0078be2d7a834a74c0766e1e5aef0` | 2026-11-09T09:46:23-05:00 | `b297d7e51d079640b56494493805f76664f8de358787c14ec77048ad916b8604` |
| D12's rerun check, completed: 24 of 31 files identical; the other 7 differ only in D12's labels and marks | `docs/tcga/run/d12_rerun_check.txt` | `939948824b76f4c4212c887b2736c85412ced38c` | 2026-11-09T10:14:29-05:00 | `c482eb51e856fae2d2f9e51f1c320adce891fa0478d2dcaf5da4440410daa235` |
| The run's record (D12): the comparison's first draft would have passed a rerun that changed nothing; the end-state check that closed it | `docs/tcga/run/README.md` | `e67a514e78ce971c418384800603056a45e3d744` | 2026-11-09T10:17:21-05:00 | `8fe79683902fc0ebc263921a072bd8474aae665acf56fd52e39a7d661e5f33b7` |

## Batch files (fixed by D1, generated before any monitor run)

Generated by `scripts/v02_design.py` at tcga `b86d5ea` from the sealed design files, checked (non-IHC patients only in
fold 1's held-out batches; no AO patient in the Virchow file; every case-mix resample's class counts equal the share ×
47 or × 19, rounded; 47 / 19 patients; a full batch and 20 + 20 resamples in each of the 225 fold × halving × target
cells) and regenerated byte for byte from a scratch copy.

| File (on branch `tcga`) | Commit | Committed | SHA-256 of the file at that commit |
|---|---|---|---|
| `data/tcga_brca/design_er/batches.csv` (ResNet-50 arm) | `3e342fd2fc54127d6568b6544cb0731460eaaf70` | 2026-11-07T10:22:11-05:00 | `37e396be9e4cc1c299f75f648b075e98b211df03f71c7f1f750f2fc84285aee0` |
| `data/tcga_brca/design_er/batches_virchow.csv` (Virchow arm, AO removed before drawing) | `3e342fd2fc54127d6568b6544cb0731460eaaf70` | 2026-11-07T10:22:11-05:00 | `f8d975c22cc72e38a46b44d8e39919bdb8c460a222893ed4cdbd78a8b708eccc` |

## Per-site batch files (fixed by D3, generated before any monitor run)

Generated by `scripts/v02_design.py --tss-batches` at tcga `59020cb` from the sealed design files alone (the patients
and halvings files; no label is read), checked (the sites with at least 20 primary patients in a pool: 14, and 13 on
the Virchow arm, where AO is removed before drawing; each full batch holds all of its site's patients in that pool,
non-IHC included (AN's 12); 20 resamples of 19 patients for every site and 20 of 47 for the sites with at least 47
patients, 9 and 8 on the Virchow arm; all of it in each of the 5 halvings of the fold that holds the site's pool out)
and regenerated byte for byte from a scratch copy that holds no label file. These batches are descriptive (D3): they
feed the per-site readings beside the pools and no hypothesis.

| File (on branch `tcga`) | Commit | Committed | SHA-256 of the file at that commit |
|---|---|---|---|
| `data/tcga_brca/design_er/batches_tss.csv` (ResNet-50 arm) | `67d8cbeef540cbc0a5c54c547dd2b14752c29420` | 2026-11-07T11:29:13-05:00 | `7f7d249c2ba04b7ad61423c93ba6611853b9c5974a9c4f1e0f0eaf94d54db2c7` |
| `data/tcga_brca/design_er/batches_tss_virchow.csv` (Virchow arm, AO removed before drawing) | `67d8cbeef540cbc0a5c54c547dd2b14752c29420` | 2026-11-07T11:29:13-05:00 | `f197717e138cb0240bb584665c8f9f9f6d0d2c75d1c799e1b0651c71e40fe7c3` |
