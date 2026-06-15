# Experiment log

Every non-trivial training run gets a file here: `YYYY-MM-DD-<short-name>.md`. Use `_template.md` as the starting point. The collection of these files becomes the evidence trail for the thesis defense.

**Rule:** if a run was meaningful enough to spend GPU money on, it gets logged here — including failures. "I tried X and it didn't work because Y" is defensible thesis material.

## Index

Most recent first. `WER` columns are FLEURS yo_ng (n=200) unless noted; `—` means not applicable (no model produced); `TBD` means the entry exists but the number hasn't been backfilled yet.

| Date | Trial | Status | Base WER | Run WER | Notes |
|---|---|---|---|---|---|
| _live_ | [`_active.md`](_active.md) | 🟡 in flight | — | — | Working notes for the trial currently in progress. Promoted to a dated file on completion. |
| 2026-06-15 | [v4 OOD eval on `yoruba_male_dataset`](2026-06-15-v4-ood-male-dataset.md) | ⚠️ useful, not shipped | 32.77% Y-CER-perm | **19.30% Y-CER-perm OOD / 5.64% Y-CER-perm holdout** (N=100 OOD, N=50 holdout) | Paired baseline-vs-v4 on Drive-resident merged checkpoint from `2026-06-14T17-23-20Z` training run. v4 wins on every YASR-Bench metric vs baseline; +25 pp Y-WER-perm and +13.5 pp Y-CER-perm on OOD. Diacritic gap stable at ~26 pp across holdout and OOD → evidence the fine-tune has internalised a canonical diacritization convention. Same-author OOD caveat: next round must use OpenSLR86 or FLEURS test for strict-OOD triangulation. v4 NOT pushed to HF; `M1_HF_MODEL` remains v3. |
| 2026-06-11 | [data-mix expansion → v3](2026-06-11-data-mix-v3.md) | ✅ shipped + flipped (2026-06-12) | 88.6% | **61.53% FLEURS (cloud, N=200, beams=5)** — v1=65.92%, v2=66.09%, v3=61.53%. MPS replication (2026-06-12): v1=69.24%, v2=68.54%, v3=63.71%. | Pushed as `devalade/whisper-large-v3-yoruba-v3`. v3 beats v2 by 4.56 pts (cloud) / 4.83 pts (MPS) on FLEURS by varying only the data axis (Hidi-agili + chukypedro demographic shards + FLEURS yo_ng train). Directionally validates the diversity hypothesis on both protocols but lands 1.5 pts short of the ≤ 60% gate. v2 ↔ v1 ordering flips inside noise → consistent with the single-speaker-overfit diagnosis. `M1_HF_MODEL` flipped to v3 in `config.py` on 2026-06-12; held-out + convo A/B parked as immediate follow-up. |
| 2026-06-09 | [3-epoch ft, BATCH=12, bf16 verified → v2](2026-06-09-3epoch-v2.md) | ✅ shipped | 88.6% | 65.6% FLEURS (n=25 greedy) / 49.2% on `Hidi-agili` eval split | Pushed as `devalade/whisper-large-v3-yoruba-v2`. v2 − v1 = −0.5 pts on FLEURS despite −23.8 pts on training-distribution split → single-speaker overfit confirmed. `M1_HF_MODEL` not flipped. Next trial = data-mix expansion, not more epochs. |
| 2026-06-06 | [small-run loop + convo eval set](2026-06-06-small-run-holdout.md) | ⚠️ tooling | — | — | Infrastructure for faster trials; no checkpoint produced. |
| 2026-06-05 | [first full ft: `devalade/whisper-large-v3-yoruba`](2026-06-05-large-v3-yoruba-ft.md) | ✅ shipped | TBD | TBD | Replaced Rafat baseline as M1's HF backend (v1). |
| 2026-06-05 | [pipeline-validation smoke test](2026-06-05-smoke-test.md) | ✅ pipeline ok | — | 88.4% (meaningless) | 50 samples / 1 epoch. Value was catching 5 env bugs. |

### Status legend

- ✅ **shipped** — model was pushed to HF and is in production use (M1 backend or otherwise).
- ⚠️ **useful, not shipped** — the run produced learning (negative result, tooling, methodology) but no model.
- ❌ **failed** — the run did not produce the expected output. Log the why anyway.
- 🟡 **in flight** — currently being run or recently finished; numbers not yet in.

## How to log a new trial

1. Copy `_template.md` to `YYYY-MM-DD-<short-name>.md`.
2. Fill in setup, results, what worked, what didn't, decision.
3. Add a row at the top of the Index table above.
4. Commit with the model/code change in the same commit so the doc and the code stay in sync.

If you're starting a trial and don't yet know how it'll go, work in `_active.md` and promote it when finished — see the top of that file for the convention.
