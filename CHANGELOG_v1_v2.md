# Interface changes between v1 and v2

P1–P9 saw v1. P10 and P11 saw v2. No participant saw both. The changes were
made after the thesis, at the supervisor's request, before the two follow-up
sessions. The code for each version is at the `interface-v1` and
`interface-v2` tags of this repository and of
[`xai_web`](https://github.com/sribadal/xai_web). See the Versions section of
the README for what the tags in this repository cover.

## Changes

| Area | v1 (P1–P9) | v2 (P10–P11) |
|---|---|---|
| FACE counterfactual (`face.py`) | Target was the cohort median risk. Lab values were freely mutable, and treatment was not required to differ. | Target is this patient's own prediction plus a margin, so the retrieved case does better than this patient rather than better than average. Lab values are held near-fixed by an aggregate tolerance budget in clinical units (1.0, escalating to 3.0; no counterfactual is returned above 5.0). At least one treatment flag must differ. The top 10 candidates are re-verified after the immutability rules are applied, and registry sentinel values (such as 9999) are stripped. |
| Feature-mutability policy (`feature_policy.py`) | `had_surg`, `had_syst`, `impulse_surg` and `impulse_syst` were immutable. | Those four are treated as binary treatment flags that may change. `CF_FREE_FLAGS`, `LAB_TOLERANCE` and `SENTINEL_VALUES` were added. This policy is shared by every counterfactual method, so the whole counterfactual family was re-audited (see the paper's Supplement 5). |
| Attribution panel: spacing (`xai_web` `6421dda`) | Bar width depended on the number of prediction horizons (three for MA-RSF, five for ConDOR), so the same panel looked different for the two models. | The same width for both models. |
| Attribution panel: labels (`3a6a83d`) | For LIME (the ConDOR attribution method), each label carried the split condition after the variable name, for example "Surgery (≤ 0)". | Variable name only. TreeSHAP (MA-RSF) labels never had a condition, so they are unchanged. |
| Counterfactual panel: changed features (`5976dfe`) | A feature was highlighted as changed whenever its value in the comparison differed from the patient's, even if the method had not changed it. Registry codes for unknown values (such as 9999) could be flagged as implausible. | Only the features the method changed are highlighted. Registry codes for unknown values are not flagged as implausible. |
| Counterfactual panel: shared comparison (`606babf`) | No note. | When the same comparison patient is the closest match at other horizons, a note above the panel says so: "The same comparable patient was also the closest match at 3 and 6 months, so this comparison is identical there. Only the predicted difference changes." (English and Dutch.) |
| Value labels (`feature-value-encoder.ts`, `eeead9c`) | A TNM value missing from the label map was shown with a stage prefix, so an encoded value appeared as a stage that does not exist (clinical T stage shown as cT7; the registry value was cT4). Basis-of-diagnosis codes 1 and 2 had their labels swapped. The T and N labels mixed size and node definitions from TNM 7 and TNM 8. Seven morphology codes, and a metastasis count of 0, had no label. | A TNM value missing from the map shows "—". Basis-of-diagnosis labels follow the registry data dictionary, with code 0 (death certificate only) added. T and N labels show the bare stage (cT2, cN1). The missing morphology codes and the count of 0 have labels. |

All `xai_web` changes were committed on 31 July, before P11's session that day. The FACE and feature-policy changes in `xai_py` were made before the follow-up sessions and committed later (25 November).

P8 (5 June) saw the interface one commit before the other v1 participants: the practice video was replaced on 11 June. The vignettes were the same.

## Unchanged in both versions

- **Season of diagnosis.** One value that should have shown as winter was
  mislabelled. Several participants noticed it.
- **Attribution values on hover.** Patient values in the attribution panel are
  shown only on hover, not on the axis. Most participants asked for them to be
  shown directly.

## Effect on the selected methods

The method selection did not change. FACE remained the selected
counterfactual for both models: it is still the only ConDOR counterfactual
that is not dropped, and still the highest-scoring counterfactual on the
MA-RSF. Its scores rose on both models. Both audits are in the paper's
Supplement 5.