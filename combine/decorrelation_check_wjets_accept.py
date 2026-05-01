"""
Local Asimov check for referee comment (25): does decorrelating the
QCDscale_wjets_ACCEPT and PDF_wjets_ACCEPT 5% lnN's between the W+jets CR and
each SR materially change the fit sensitivity?

Strategy: build a tiny Asimov binned-Poisson likelihood from the v19_wjetscrbinned
templates (8 regions x 8 mass bins). Float r (signal strength), the three
W+jets pT-binned norms (with VBF as a fixed linear combination), and the ttbar
norm. Add the W+jets QCDscale + PDF lnN's in two scenarios:
  (A) correlated:    one theta_qcd + one theta_pdf, applied to W+jets in all 8 regions
  (B) decorrelated:  one theta per (nuisance, region) -> 16 nuisances total
Compute sigma(r) via the marginal Hessian and report the difference.

This is a flat-rate check by construction (the lnN doesn't shape m*), so it
won't reproduce combine to 3 decimals. It does answer the relative question:
once the floating CR norms absorb the common shift, do the residual per-SR
freedoms matter for sigma(r)?
"""
import pickle
import numpy as np
from scipy.optimize import minimize

PKL = "/Users/fmokhtar/projects/boostedhiggs/combine/templates/v19_wjetscrbinned/hists_templates_Run2.pkl"

REGIONS = [
    "VBF",
    "ggFpt250to350",
    "ggFpt350to500",
    "ggFpt500toInf",
    "WJetsCRpt250to350",
    "WJetsCRpt350to500",
    "WJetsCRpt500toInf",
    "TopCR",
]
SIG_SAMPLES = ["ggF", "VBF", "WH", "ZH", "ttH", "ggFpt200to300", "ggFpt300to450", "ggFpt450toInf", "mjj1000toInf"]
WJETS = "WJetsLNu"
TTBAR = "TTbar"
OTHER_BKG = ["SingleTop", "DYJets", "Diboson", "EWKvjets", "Fake"]

# VBF region's W+jets is a fixed linear combination of the three pT-bin norms.
# Coefficients copied from combined.txt (line for wjetsnormSFVBF_effect_VBF_wjets).
VBF_W_COEF = {"pt250": 0.19549559436291622, "pt350": 0.4639551601691901, "pt500": 0.3405492454678938}

LNN_QCD = 0.05
LNN_PDF = 0.05


def load_templates():
    with open(PKL, "rb") as f:
        h = pickle.load(f)
    n_reg, n_mass = len(REGIONS), 8
    S = np.zeros((n_reg, n_mass))
    Bw = np.zeros((n_reg, n_mass))
    Bt = np.zeros((n_reg, n_mass))
    Bo = np.zeros((n_reg, n_mass))
    for i, reg in enumerate(REGIONS):
        for s in SIG_SAMPLES:
            try:
                S[i] += h[{"Sample": s, "Region": reg, "Systematic": "nominal"}].values()
            except KeyError:
                pass
        Bw[i] = h[{"Sample": WJETS, "Region": reg, "Systematic": "nominal"}].values()
        Bt[i] = h[{"Sample": TTBAR, "Region": reg, "Systematic": "nominal"}].values()
        for s in OTHER_BKG:
            try:
                Bo[i] += h[{"Sample": s, "Region": reg, "Systematic": "nominal"}].values()
            except KeyError:
                pass
    # Clip to physical-positive
    return np.maximum(S, 0), np.maximum(Bw, 0), np.maximum(Bt, 0), np.maximum(Bo, 0)


# region index -> which W+jets norm(s)
def wjets_factor(i, mu_w):
    """mu_w = (mu_pt250, mu_pt350, mu_pt500)."""
    if i == 0:  # VBF
        return VBF_W_COEF["pt250"] * mu_w[0] + VBF_W_COEF["pt350"] * mu_w[1] + VBF_W_COEF["pt500"] * mu_w[2]
    if i in (1, 4):
        return mu_w[0]  # ggFpt250to350 / WJetsCRpt250to350
    if i in (2, 5):
        return mu_w[1]
    if i in (3, 6):
        return mu_w[2]
    return 1.0  # TopCR: W+jets not floated (negligible there anyway)


N_REG = len(REGIONS)


# Scenarios:
#   "A"  : 1 nuisance per ACCEPT (theta_qcd, theta_pdf), correlated across all 8 regions (current setup)
#   "B"  : per-region nuisances (8 thetas per ACCEPT) -> 16 in total (most aggressive)
#   "C"  : per-pT-bin (4 thetas per ACCEPT: pt250, pt350, pt500, VBF), shared between SR and matched CR
#   "D"  : 1 shared CR-side nuisance + 4 SR-only nuisances per ACCEPT (5 thetas per ACCEPT)
#
# region order: 0=VBF, 1..3=ggF SRs (pt250/350/500), 4..6=W+jets CRs (pt250/350/500), 7=TopCR
def theta_map(scenario):
    """Return list-of-arrays mapping each region to its (qcd, pdf) nuisance index, plus n_thetas per ACCEPT."""
    if scenario == "A":
        idx = [0] * N_REG
        return idx, 1
    if scenario == "B":
        idx = list(range(N_REG))
        return idx, N_REG
    if scenario == "C":
        # pt-bin bucket: VBF -> 3 (its own); ggFpt250+CRpt250 -> 0; pt350 -> 1; pt500 -> 2
        idx = [3, 0, 1, 2, 0, 1, 2, 0]  # TopCR shares with bucket 0 (W+jets is tiny there anyway)
        return idx, 4
    if scenario == "D":
        # bucket 0: shared "CR" nuisance for the 3 W+jets CRs and TopCR
        # buckets 1..4: per-SR for VBF, ggFpt250, ggFpt350, ggFpt500
        idx = [1, 2, 3, 4, 0, 0, 0, 0]
        return idx, 5
    raise ValueError(scenario)


def build_nll(S, Bw, Bt, Bo, scenario):
    N_obs = S + Bw + Bt + Bo
    idx, n_thetas = theta_map(scenario)
    n_theta_total = 2 * n_thetas  # qcd + pdf
    n_params = 1 + 3 + 1 + n_theta_total

    def unpack_thetas(p):
        tq_vals = p[5 : 5 + n_thetas]
        tp_vals = p[5 + n_thetas : 5 + 2 * n_thetas]
        tq = np.array([tq_vals[idx[i]] for i in range(N_REG)])
        tp = np.array([tp_vals[idx[i]] for i in range(N_REG)])
        return tq, tp

    def predict(p):
        r = p[0]
        mu_w = p[1:4]
        mu_t = p[4]
        tq, tp = unpack_thetas(p)
        pred = np.zeros_like(N_obs)
        for i in range(N_REG):
            wf = wjets_factor(i, mu_w)
            wunc = np.exp(LNN_QCD * tq[i] + LNN_PDF * tp[i])
            pred[i] = r * S[i] + wf * wunc * Bw[i] + mu_t * Bt[i] + Bo[i]
        return np.maximum(pred, 1e-12)

    def nll(p):
        pred = predict(p)
        pois = 2 * np.sum(pred - N_obs + N_obs * np.log(N_obs / pred))
        gauss = np.sum(p[5 : 5 + 2 * n_thetas] ** 2)
        return pois + gauss

    return nll, n_params, predict


def numerical_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm = x.copy()
            xpm[i] += eps
            xpm[j] -= eps
            xmp = x.copy()
            xmp[i] -= eps
            xmp[j] += eps
            xmm = x.copy()
            xmm[i] -= eps
            xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H, f0


def fit_and_sigma(S, Bw, Bt, Bo, scenario):
    nll, n, _ = build_nll(S, Bw, Bt, Bo, scenario)
    x0 = np.zeros(n)
    x0[0:5] = 1.0
    res = minimize(
        nll, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 200000, "adaptive": True}
    )
    if not res.success:
        res = minimize(nll, res.x, method="Powell", options={"xtol": 1e-9, "ftol": 1e-9, "maxiter": 200000})
    x_hat = res.x
    H, f0 = numerical_hessian(nll, x_hat, eps=1e-4)
    cov = np.linalg.inv(H)
    sig_r = float(np.sqrt(cov[0, 0]))
    return x_hat, sig_r, f0


def main():
    print("Loading templates...")
    S, Bw, Bt, Bo = load_templates()

    print("\nPer-region yields (sum over m* bins, Asimov r=1):")
    print(f"{'region':22s}  {'S':>8s}  {'B_w':>9s}  {'B_t':>9s}  {'B_o':>9s}  {'B_tot':>9s}  W%   S/sqrt(B)")
    Btot_all = Bw + Bt + Bo
    for i, reg in enumerate(REGIONS):
        s_i = S[i].sum()
        bw_i = Bw[i].sum()
        bt_i = Bt[i].sum()
        bo_i = Bo[i].sum()
        bt_tot = bw_i + bt_i + bo_i
        wpct = 100 * bw_i / max(bt_tot, 1e-9)
        soverb = s_i / max(np.sqrt(bt_tot), 1e-9)
        print(f"{reg:22s}  {s_i:8.2f}  {bw_i:9.2f}  {bt_i:9.2f}  {bo_i:9.2f}  {bt_tot:9.2f}  {wpct:4.1f}  {soverb:7.4f}")

    scenarios = {
        "A": "1 nuisance, fully correlated across all 8 regions (current setup)",
        "B": "per-region nuisances (8 per ACCEPT, 16 total) - most aggressive",
        "C": "per-pT-bin (4 per ACCEPT, 8 total): SR and matched CR share within a pT bin",
        "D": "1 shared CR + 4 per-SR nuisances (5 per ACCEPT, 10 total)",
    }
    results = {}
    for sc, descr in scenarios.items():
        print(f"\n--- Scenario {sc}: {descr} ---")
        x, sig, nll0 = fit_and_sigma(S, Bw, Bt, Bo, sc)
        results[sc] = (x, sig, nll0)
        print(f"  r_hat = {x[0]:.6f}   sigma(r) = {sig:.6f}   nll_min = {nll0:.3e}")
        print(f"  mu_w (pt250, pt350, pt500) = {x[1]:.4f}, {x[2]:.4f}, {x[3]:.4f}   mu_ttbar = {x[4]:.4f}")

    print("\n=== Comparison: relative change in sigma(r) vs scenario A ===")
    sig_A = results["A"][1]
    print(f"  A (correlated, baseline) sigma(r) = {sig_A:.6f}")
    for sc in ("B", "C", "D"):
        s = results[sc][1]
        rel = (s - sig_A) / sig_A * 100
        print(f"  {sc}: sigma(r) = {s:.6f}   delta = {rel:+.3f}%")


if __name__ == "__main__":
    main()
