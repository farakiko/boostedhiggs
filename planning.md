# HIG-24-008 — Referee response action plan (Farouk side)

Source of comments: `HIG-24-008_review.pdf` (JHEP_259P_0326 referee report).
Paper: arXiv:2603.22233. Branch focus: 1ℓ ggF + 1ℓ VBF SRs (Farouk-led).

Conventions used below:
- **Owner**: who the referee tag points to (Farouk-only, or shared).
- **Type**: `text` (review/polish only), `lookup` (pull a number from existing fits/tables), `empirical` (need a fresh plot/check from `events_dict` or combine output).
- **Status**: `todo` / `in-progress` / `done`.
- Region shorthand mirrors `binder/hists_plots.ipynb`: `THWW`, `n_bjets_T`, `mjj`, `deta`, `NumOtherJets`, `VH_fj_VScore`, `rec_higgs_pt`, `rec_higgs_m`.

---

## Comments tagged "Farouk"

### (3) Cross-check Table 5 (high-pT H complementarity)
- Owner: Farouk + Mohammad.
- Type: lookup.
- Ask: confirm the per-channel expected significance numbers and the simplified quadratic combination in the new Table 5 added to the paper.
- Plan: pull current AN/PAS values for boosted Hbb (Run 2 + Run 3 if cited), Hττ, Hcc, and our HWW expected significance from the latest combine output; reproduce the quadratic sum row.
- Status: todo.

### (8) Lepton ID/iso scale-factor validity for in-jet leptons
- Owner: Christina/Farouk/Corrine/Mohammad.
- Type: text + lookup.
- Ask: polish the response that "signal and background leptons have identical isolation efficiency"; quote uncertainty magnitudes on the isolation efficiency.
- Plan:
  - Confirm with `boostedhiggs/corrections.py` what SF source is applied (POG vs custom).
  - For 1ℓ, retrieve isolation SF uncertainty magnitudes from POG JSONs (currently consumed by the processor) — quote min/max %.
  - Optionally compare ID/iso efficiency on prompt prompt-isolated CR (Z→ll-like) vs on signal-jet leptons in MC; show that they agree within the assigned unc.
- Status: todo.

### (9) Mass sculpting from the THWW tagger
- Owner: Christina/Farouk.
- Type: empirical (key check).
- Ask: prove the tagger does not sculpt the background mass to mimic a 125 GeV bump after the SR THWW cut.
- Plan in `binder/hists_plots.ipynb`:
  1. In a background-enriched region (W+jets CR or pre-tagger preselection on data), plot `rec_higgs_m` (and also `fj_mass`/`fj_ParT_mass`) for several THWW slices: `<0.5`, `0.5–0.75`, `0.75–0.9`, `0.9–0.93`, `>0.93`. Look for an artificial enhancement near 125 GeV.
  2. Same with QCD MC and W+jets MC, separately.
  3. Quote the relative shape change in the 110–140 GeV window vs sidebands and confirm it stays smooth.
- Deliverable: 1 figure (data) + supporting MC, plus one-paragraph response.
- Status: todo.

### (10) LJP reweighting — robustness of W→qq (2-prong) → H→4q (4-prong) extrapolation
- Owner: Farouk.
- Type: text + empirical.
- Ask: defend that the SF derived on tt̄ semileptonic W jets (≤3 prongs in the validation) extrapolates to the 4-prong H→WW signal.
- Plan:
  - Re-state the LJP method's prong-by-prong factorization (each splitting reweighted independently → number of prongs is not an extrapolation parameter, only kinematic coverage of the LP density is).
  - Empirical support: in `lundplaneSF/LundReweighting/`, plot the LP density coverage of H→4q vs W→qq (or just the 1D Lund plane) and show H→4q lives inside the calibrated region of (kT, ΔR).
  - Cite the existing prong-mismodeling systematic (`LP_weight_prongs`) and the unclustered/distortion variations as the way prong-count differences are absorbed into the uncertainty.
- Status: todo.

### (11) Why ΔR=0.2 PF-constituent removal around the lepton in the LJP procedure
- Owner: Farouk.
- Type: text.
- Ask: explain that ΔR is measured from the lepton, and why we strip PF constituents in addition to the lepton itself.
- Response (to draft): the lepton's calorimeter footprint (and recovered photons/pile-up correction) leaves residual constituents in the AK8 jet that distort the LP density at small angles; removing all PF candidates within ΔR(lepton, constituent)<0.2 yields a clean hadronic LP density consistent with the W→qq calibration, mirroring the lepton-subtraction recipe used in W jet mass corrections.
- Plan: paste the relevant snippet from the LJP-input processor for confirmation; keep the response to 2–3 sentences.
- Status: todo.

### (12) LJP scale-factor text rephrasing — approval
- Owner: Farouk/Cristina/Corrinne.
- Type: text.
- Ask: approve the proposed rewordings of the "agreement between data and simulation…" and "scale factor is defined as…" sentences.
- Status: todo (low effort, text only).

### (19,i) Migration matrix: reco pjT bins → STXS pHT bins
- Owner: Christina/Farouk/Melisha.
- Type: empirical (small).
- Ask: report the migration matrix between our 3 reco pT(j) bins (250–350, 350–500, 500–Inf, plus VBF) and the STXS pT(H) bins. Either the diagonal numbers or the full matrix as a small table/equation.
- Plan in `binder/hists_plots.ipynb` using ggF and VBF MC only:
  1. Define reco bins from `rec_higgs_pt` per the SR definitions; define STXS pT(H) bins per the paper.
  2. Tabulate `nominal`-weighted yields in each (reco, STXS) cell, normalize per STXS column.
  3. Output as a 4×N LaTeX table or a numpy matrix that can be inlined as an equation. Save under `binder/efficiencies/`.
- Status: todo.

### (24) Definition of QCD-enriched samples for fake-factor measurement
- Owner: Christina/Farouk/Corrine/Mohammad.
- Type: text + lookup.
- Ask: how the QCD-enriched data sample for the fake-factor extraction was defined.
- Plan:
  - Pull the exact definition from `boostedhiggs/fakesprocessor.py` and the `binder/Dec20_fakeRate_extraction.ipynb` / `Dec20_promptRate_extraction.ipynb`.
  - Write it up: trigger paths, lepton-ID inversion (loose-not-tight), b-jet veto, anti-isolation in the AK8 region, MC subtraction of prompt contamination — quote what is actually in code.
- Status: todo.

### (25) W+jets relative-acceptance modelling uncertainty across ggF/VBF/VH SRs
- Owner: Christina/Farouk/Corrine/Mohammad.
- Type: empirical + text.
- Ask: are there modelling uncertainties on the W+jets relative acceptance between ggF / VBF / VH SRs (since the W+jets normalization is constrained from a CR shared across SRs)?
- Plan:
  - Inventory existing W+jets nuisances in `combine/systematics.py` and `combine/datacard_systematics.py` (PDF, αS, μF/μR scale envelope, PS ISR/FSR). NOTE: we switched to NLO W+jets samples and **do not apply** the Nick (V+jets NLO EWK/QCD k-factor) correction — do not list it as a coverage source.
  - For each ggF/VBF/VH SR, compute the variation of W+jets yield under (a) renorm/fact-scale envelope, (b) PDF+αS, (c) PS ISR/FSR. If the relative acceptance ratio (SR_X / CR) shifts > the assigned unc, propose adding a dedicated extrapolation uncertainty.
  - Check whether these nuisances are correlated across SRs (correct) or decorrelated (would be the natural answer here).
- Status: todo.

### (26) tt̄ relative-acceptance modelling uncertainty across ggF vs VBF SRs
- Owner: Christina/Farouk/Corrine/Mohammad.
- Type: empirical + text.
- Ask: same as (25) but for tt̄ and ggF vs VBF.
- Plan: mirror (25) — `weight_TopPtReweight`, ISR/FSR (`weight_PSFSR/PSISR`), μF/μR, hdamp; quantify the relative ggF/VBF acceptance change.
- Status: todo.

### (29) Impact plot — nuisance parameter ranking
- Owner: Farouk/Mohammad/Antonis.
- Type: empirical (combine).
- Ask: provide a NP impact/ranking plot (auxiliary material) demonstrating no large pulls and identifying the dominant uncertainties.
- Plan:
  - Use `combine` MultiDimFit + `combineTool.py -M Impacts` on the latest combined datacard (1ℓ ggF+VBF, optionally combined with 0ℓ).
  - Render with `plotImpacts.py`. Save PDF under `combine/templates/.../impacts/`.
  - Discuss with Mohammad/Antonis whether the auxiliary plot is per-channel or combined.
- Status: todo.

---

## Suggested execution order

1. **Quick wins (text only):** (11), (12) — minutes of work once we draft the wording.
2. **Single-file empirical checks in the notebook:** (9) mass sculpting, (19,i) migration matrix.
3. **Lookup + write-up:** (3) Table 5 cross-check, (24) fake QCD-enriched sample definition.
4. **Acceptance/modelling studies:** (25) W+jets, (26) tt̄.
5. **Combine deliverables:** (29) impact plot.
6. **LJP defense:** (8), (10) — natural pair; do (10) first (technical), then (8).

## Working notes / shortcuts

- All ad-hoc plots can be made by extending `binder/hists_plots.ipynb`'s `vars_to_plot` and `region_sel`/`categories_sel` dicts.
- Use `make_composition_table(events_dict, presel)` (defined in the notebook) to print yield tables for any region — directly useful for (24)/(25)/(26).
- For combine work, start from `combine/run.sh` to remember the latest datacard tag.
- Save figures to `binder/efficiencies/` (or a fresh `referee_response/` subfolder if we want them isolated).
