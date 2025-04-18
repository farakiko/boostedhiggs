#!/usr/bin/python

import json
import os
import pickle as pkl
import warnings

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import onnx
import onnxruntime as ort
import scipy

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")


# df = 1

# mapping_dict = {
#     "mjj350-700": (df["VBF"]["lep"]["STXS_finecat"] % 100 == 17) | (df["VBF"]["lep"]["STXS_finecat"] % 100 == 18),
#     "mjj700-1000": (df["VBF"]["lep"]["STXS_finecat"] % 100 == 19) | (df["VBF"]["lep"]["STXS_finecat"] % 100 == 20),
#     "mjj1000-1500": (df["VBF"]["lep"]["STXS_finecat"] % 100 == 21) | (df["VBF"]["lep"]["STXS_finecat"] % 100 == 22),
#     "mjj1500plus": (df["VBF"]["lep"]["STXS_finecat"] % 100 == 23) | (df["VBF"]["lep"]["STXS_finecat"] % 100 == 24),
# }


combine_samples_by_name = {
    "GluGluHToWW_Pt-200ToInf_M-125": "ggF",
    "GluGluHToWW_Pt-200ToInf_M-125_Rivet": "ggF",
    "GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8": "ggF",  # inclusive sample
    "VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil": "VBF",
    "VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet": "VBF",
    "ttHToNonbb_M125": "ttH",
    "HWminusJ_HToWW_M-125": "WH",
    "HWplusJ_HToWW_M-125": "WH",
    "HZJ_HToWW_M-125": "ZH",
    "GluGluZH_HToWW_M-125_TuneCP5_13TeV-powheg-pythia8": "ZH",
    "GluGluHToTauTau": "HTauTau",
}

combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # bkg
    "QCD_Pt": "QCD",
    "TT": "TTbar",
    "WJetsToLNu_HT": "WJetsLNu_LO",
    "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr": "WJetsLNu",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "EWK": "EWKvjets",
    "DYJets": "DYJets",
    "JetsToQQ": "WZQQ",
    # # NLO:
    # "WJetsToLNu_1J": "WJetsLNu_NLO",
    # "WJetsToLNu_2J": "WJetsLNu_NLO",
    # "WJetsToLNu_LHEFilterPtW": "WJetsLNu_NLO",
    # "WJetsToLNu_HT": "LO HT binned",
    # "WJetsToLNu_1J": "WJetsLNu NLO LHEFilterPtW",
    # "WJetsToLNu_2J": "WJetsLNu NLO LHEFilterPtW",
    "WJetsToLNu_1J": "WJetsLNu NLO MatchEWPDG20",
    "WJetsToLNu_2J": "WJetsLNu NLO MatchEWPDG20",
    "WJetsToLNu_LHEFilterPtW": "WJetsLNu NLO LHEFilterPtW",
    "_MatchEWPDG20": "WJetsLNu NLO MatchEWPDG20",
}

signals = ["VBF", "ggF", "WH", "ZH", "ttH"]


def get_common_sample_name(sample):

    # first: check if the sample is in one of combine_samples_by_name
    sample_to_use = None
    for key in combine_samples_by_name:
        if key in sample:
            sample_to_use = combine_samples_by_name[key]
            break

    # second: if not, combine under common label
    if sample_to_use is None:
        for key in combine_samples:
            if key in sample:
                sample_to_use = combine_samples[key]
                break
            else:
                sample_to_use = sample
    return sample_to_use


def get_sum_sumgenweight(pkl_files, year, sample):
    """Load and sum the sumgenweight of each pkl file."""

    sum_sumgenweight = 0
    for ifile in pkl_files:
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]

    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    """Get xsec-weight and scales events by lumi/sumgenweights."""

    # if "WJetsToLNu_1J" in sample:
    #     sample = "WJetsToLNu_1J"
    # if "WJetsToLNu_2J" in sample:
    #     sample = "WJetsToLNu_2J"

    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


# ---------------------------------------------------------
# TAGGER STUFF
def get_finetuned_score(data, model_path):

    input_dict = {
        "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
    }

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(
        model_path,
        providers=["AzureExecutionProvider"],
    )
    outputs = ort_sess.run(None, input_dict)

    return scipy.special.softmax(outputs[0], axis=1)[:, 0]


# ---------------------------------------------------------

# PLOTTING UTILS
color_by_sample = {
    "ggF": "lightsteelblue",
    "VBF": "peru",
    # signal that is background
    "WH": "tab:brown",
    "ZH": "yellowgreen",
    "ttH": "tab:olive",
    # background
    "QCD": "tab:orange",
    "Fake": "orange",
    "Fake_EWK_SF_Up": "orange",
    "Fake_EWK_SF_Down": "orange",
    "Fake_FR_stat_Up": "orange",
    "Fake_FR_stat_Down": "orange",
    # "Fake": "navajowhite",
    "WJetsLNu": "tab:green",
    "WJetsLNu_LO": "springgreen",
    "TTbar": "tab:blue",
    "Diboson": "orchid",
    "SingleTop": "tab:cyan",
    "EWKvjets": "tab:grey",
    "DYJets": "tab:purple",
    "WZQQ": "khaki",
    # "WZQQorDYJets": "khaki",
    # wjets matched and unmatched
    "WJetsLNu_unmatched": "lightgreen",
    "WJetsLNu_matched": "tab:green",
    # ttbar matched and unmatched
    "TTbar_allmatched": "tab:blue",
    "TTbar_unmatched": "lightskyblue",
    # ttbar LP
    "TTbar_is_top_lq": "lightskyblue",
    "TTbar_is_top_lqq": "tab:blue",
    "TTbar_others": "darkblue",
    "TTbar_LP_is_top_lq": "lightskyblue",
    "TTbar_LP_is_top_lqq": "tab:blue",
    "TTbar_LP_others": "darkblue",
    "TTbar_LP": "tab:blue",
    # split WJets by b/c hads
    "WJetsLNu_nB0": "tab:green",
    "WJetsLNu_nB1": "springgreen",
    "WJetsLNu_nB2": "greenyellow",
    "WJetsLNu_nBother": "lime",
    "WJetsLNu_nC0": "tab:green",
    "WJetsLNu_nC1": "springgreen",
    "WJetsLNu_nC2": "greenyellow",
    "WJetsLNu_nCother": "lime",
    "WJetsLNu NLO LHEFilterPtW": "greenyellow",
    "WJetsLNu NLO MatchEWPDG20": "springgreen",
    "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr": "lime",
}

plot_labels = {
    "ggF": "ggF",
    "WH": "WH",
    "ZH": "ZH",
    "VH": "VH",
    # "VH": "VH(WW)",
    # "VBF": r"VBFH(WW) $(qq\ell\nu)$",
    "VBF": r"VBF",
    # "ttH": "ttH(WW)",
    "ttH": r"$t\bar{t}$H",
    "QCD": "Multijet",
    "Fake": "Fake",
    "Fake_EWK_SF_Up": "Fake_EWK_SF_Up",
    "Fake_EWK_SF_Down": "Fake_EWK_SF_Down",
    "Fake_FR_stat_Up": "Fake_FR_stat_Up",
    "Fake_FR_stat_Down": "Fake_FR_stat_Down",
    "Diboson": "VV",
    "WJetsLNu": r"W$(\ell\nu)$+jets",
    "WJetsLNu_LO": r"W$(\ell\nu)$+jets (LO HT binned)",
    "SingleTop": r"Single T",
    "EWKvjets": "EWK VJets",
    "DYJets": r"Z$(\ell\ell)$+jets",
    "WZQQ": r"V$(qq)$",
    # "WZQQorDYJets": r"W$(qq)$/Z(inc.)+jets",  # TODO: make sure it's WZQQ is NLO in next iteration
    # wjets matched and unmatched
    "WJetsLNu_unmatched": r"W$(\ell\nu)$+jets unmatched",
    "WJetsLNu_matched": r"W$(\ell\nu)$+jets matched",
    # ttbar matched and unmatched
    "TTbar_allmatched": r"$t\bar{t}$+jets matched",
    "TTbar_unmatched": r"$t\bar{t}$+jets unmatched",
    "TTbar": r"$t\bar{t}$+jets",
    "TTbar_is_top_l": r"$t^{\ell}$",
    "TTbar_is_top_lq": r"$t^{{\ell}q}$",
    "TTbar_is_top_lqq": r"$t^{{\ell}qq}$",
    "TTbar_is_top_lb": r"$t^{{\ell}b}$",
    "TTbar_is_top_lbq": r"$t^{{\ell}bq}$",
    "TTbar_is_top_lbqq": r"$t^{{\ell}bqq}$",
    "TTbar_others": r"$t\bar{t}$+jets (other)",
    "TTbar_LP": r"$t\bar{t}$+jets (after LP reweighting)",
    "TTbar_LP_is_top_l": r"$t^{\ell}$ (after LP reweighting)",
    "TTbar_LP_is_top_lq": r"$t^{{\ell}q}$ (after LP reweighting)",
    "TTbar_LP_is_top_lqq": r"$t^{{\ell}qq}$ (after LP reweighting)",
    "TTbar_LP_is_top_lb": r"$t^{{\ell}b}$ (after LP reweighting)",
    "TTbar_LP_is_top_lbq": r"$t^{{\ell}bq}$ (after LP reweighting)",
    "TTbar_LP_is_top_lbqq": r"$t^{{\ell}bqq}$ (after LP reweighting)",
    "TTbar_LP_others": r"$t\bar{t}$+jets (other)",
    # split WJets by b/c hads
    "WJetsLNu_nB0": r"W$(\ell\nu)$+jets (nB=0)",
    "WJetsLNu_nB1": r"W$(\ell\nu)$+jets (nB=1)",
    "WJetsLNu_nB2": r"W$(\ell\nu)$+jets (nB=2)",
    "WJetsLNu_nBother": r"W$(\ell\nu)$+jets (other)",
    "WJetsLNu_nC0": r"W$(\ell\nu)$+jets (nC=0)",
    "WJetsLNu_nC1": r"W$(\ell\nu)$+jets (nC=1)",
    "WJetsLNu_nC2": r"W$(\ell\nu)$+jets (nC=2)",
    "WJetsLNu_nCother": r"W$(\ell\nu)$+jets (other)",
    "WJetsLNu NLO LHEFilterPtW": r"W$(\ell\nu)$+jets (NLO LHEFilterPtW)",
    "WJetsLNu NLO MatchEWPDG20": r"W$(\ell\nu)$+jets (NLO MatchEWPDG20)",
    "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr": r"W$(\ell\nu)$+jets (NLO 012JetsNLO 34JetsLO)",
}

label_by_ch = {"mu": "Muon", "ele": "Electron"}


def get_axis(var, massbin=5):
    axis_dict = {
        "dR_genlep_recolep": hist2.axis.Regular(
            35, 0, 0.02, name="var", label=r"$\Delta R(\ell_{gen}, \ell_{reco})$", overflow=True
        ),
        "Zmass": hist2.axis.Regular(40, 30, 450, name="var", label=r"Zmass [GeV]", overflow=True),
        "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
        "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
        "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
        "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
        "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
        "rec_higgs_etajet_m": hist2.axis.Variable(
            list(range(50, 240, 20)), name="var", label=r"PKU definition Higgs reconstructed mass [GeV]", overflow=True
        ),
        "rec_higgs_pt": hist2.axis.Variable(
            [250, 350, 500, 1000], name="var", label=r"Higgs reconstructed $p_T$ [GeV]", overflow=True
        ),
        "fj_pt_over_lep_pt": hist2.axis.Regular(35, 1, 10, name="var", label=r"$p_T$(Jet) / $p_T$(Lepton)", overflow=True),
        "rec_higgs_pt_over_lep_pt": hist2.axis.Regular(
            35, 1, 10, name="var", label=r"$p_T$(Recontructed Higgs) / $p_T$(Lepton)", overflow=True
        ),
        "golden_var": hist2.axis.Regular(35, 0, 10, name="var", label=r"$p_{T}(W_{l\nu})$ / $p_{T}(W_{qq})$", overflow=True),
        "rec_dphi_WW": hist2.axis.Regular(
            35, 0, 3.14, name="var", label=r"$\left| \Delta \phi(W_{l\nu}, W_{qq}) \right|$", overflow=True
        ),
        "fj_ParT_mass": hist2.axis.Variable(
            list(range(65, 240, massbin)), name="var", label=r"ParT regressed mass [GeV]", overflow=True
        ),
        # VBF
        "deta": hist2.axis.Regular(35, 0, 7, name="var", label=r"$\left| \Delta \eta_{jj} \right|$", overflow=True),
        "mjj": hist2.axis.Regular(35, 0, 2000, name="var", label=r"$m_{jj}$", overflow=True),
        "fj_genjetpt": hist2.axis.Regular(30, 200, 600, name="var", label=r"Gen Jet $p_T$ [GeV]", overflow=True),
        "jet_resolution": hist2.axis.Regular(
            30, -3, 3, name="var", label=r"(Gen Jet $p_T$ - Jet $p_T$)/Gen Jet $p_T$", overflow=True
        ),
        "nj": hist2.axis.Regular(40, 0, 10, name="var", label="number of jets outside candidate jet", overflow=True),
        "inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
        "THWW": hist2.axis.Regular(25, 0, 1, name="var", label=r"$T_{HWW}$", overflow=True),
        # "THWW": hist2.axis.Regular(25, 0.75, 1, name="var", label=r"$T_{HWW}$", overflow=True),  # preselection
        # "THWW": hist2.axis.Regular(8, 0.9, 1, name="var", label=r"$T_{HWW}$", overflow=True),
        "fj_ParT_inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"ParT-Finetuned score", overflow=True),
        "fj_ParT_all_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
        # AN
        "FirstFatjet_pt": hist2.axis.Regular(30, 250, 600, name="var", label=r"Leading AK8 jet $p_T$ [GeV]", overflow=True),
        "SecondFatjet_pt": hist2.axis.Regular(
            30, 250, 600, name="var", label=r"Sub-Leading AK8 jet $p_T$ [GeV]", overflow=True
        ),
        "fj_pt": hist2.axis.Regular(30, 250, 600, name="var", label=r"Higgs candidate jet $p_T$ [GeV]", overflow=True),
        # "fj_pt": hist2.axis.Regular(20, 250, 600, name="var", label=r"Higgs candidate jet $p_T$ [GeV]", overflow=True),
        "lep_pt": hist2.axis.Regular(40, 30, 400, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
        "pt_ratio": hist2.axis.Regular(40, 0, 2, name="var", label=r"Lepton $p_T$ / Jet $p_T$", overflow=True),
        "rho": hist2.axis.Regular(50, -1, -10, name="var", label=r"Rho", overflow=True),
        "lep_eta": hist2.axis.Regular(35, -2.5, 2.5, name="var", label=r"Lepton $\eta$", overflow=True),
        "NumFatjets": hist2.axis.Regular(5, 0.5, 5.5, name="var", label="Number of AK8 jets", overflow=True),
        "NumOtherJets": hist2.axis.Regular(
            7, 0.5, 7.5, name="var", label="Number of AK4 jets (non-overlapping with AK8)", overflow=True
        ),
        "lep_fj_dr": hist2.axis.Regular(
            35, 0.03, 0.8, name="var", label=r"$\Delta R(\ell, \mathrm{Higgs \ candidate \ jet})$", overflow=True
        ),
        "met_pt": hist2.axis.Regular(40, 20, 250, name="var", label=r"MET [GeV]", overflow=True),
        # "met_pt": hist2.axis.Regular(20, 20, 250, name="var", label=r"MET [GeV]", overflow=True),
        "met_phi": hist2.axis.Regular(40, -3.14, 3.14, name="var", label=r"MET $\Phi$", overflow=True),
        "met_fj_dphi": hist2.axis.Regular(
            35,
            0,
            1.57,
            name="var",
            label=r"$\left| \Delta \phi(MET, \mathrm{Higgs \ candidate \ jet}) \right|$",
            overflow=True,
        ),
        "lep_met_mt": hist2.axis.Regular(35, 0, 160, name="var", label=r"$m_T(\ell, p_T^{miss})$ [GeV]", overflow=True),
        "ht": hist2.axis.Regular(30, 400, 1400, name="var", label=r"ht [GeV]", overflow=True),
        "rec_W_qq_m": hist2.axis.Regular(40, 0, 160, name="var", label=r"Reconstructed $W_{qq}$ mass [GeV]", overflow=True),
        "rec_higgs_m": hist2.axis.Variable(
            list(range(75, 255, massbin)), name="var", label=r"Higgs reconstructed mass [GeV]", overflow=True
        ),
        "rec_W_lnu_m": hist2.axis.Regular(
            40, 0, 160, name="var", label=r"Reconstructed $W_{\ell \nu}$ mass [GeV]", overflow=True
        ),
        "fj_mass": hist2.axis.Variable(
            list(range(40, 255, massbin)), name="var", label=r"Higgs candidate soft-drop mass [GeV]", overflow=True
        ),
        "fj_lsf3": hist2.axis.Regular(35, 0, 1, name="var", label=r"Higgs candidate jet LSF$_3$", overflow=True),
        # lepton isolation
        "lep_isolation": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Lepton PF isolation", overflow=True),
        "lep_isolation_ele": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Electron PF isolation", overflow=True),
        "lep_isolation_ele_highpt": hist2.axis.Regular(
            35, 0, 0.5, name="var", label=r"Electron PF isolation (high $p_T$)", overflow=True
        ),
        "lep_isolation_ele_lowpt": hist2.axis.Regular(
            35, 0, 0.15, name="var", label=r"Electron PF isolation (low $p_T$)", overflow=True
        ),
        "lep_isolation_mu": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Muon PF isolation", overflow=True),
        "lep_isolation_mu_highpt": hist2.axis.Regular(
            35, 0, 0.5, name="var", label=r"Muon PF isolation (high $p_T$)", overflow=True
        ),
        "lep_isolation_mu_lowpt": hist2.axis.Regular(
            35, 0, 0.15, name="var", label=r"Muon PF isolation (low $p_T$)", overflow=True
        ),
        "lep_misolation": hist2.axis.Regular(35, 0, 0.2, name="var", label=r"Lepton mini-isolation", overflow=True),
        "lep_misolation_mu": hist2.axis.Regular(35, 0, 0.2, name="var", label=r"Muon mini-isolation", overflow=True),
        "lep_misolation_mu_highpt": hist2.axis.Regular(
            35, 0, 0.2, name="var", label=r"Muon mini-isolation (high $p_T$)", overflow=True
        ),
        "lep_misolation_mu_lowpt": hist2.axis.Regular(
            35, 0, 0.2, name="var", label=r"Muon mini-isolation (low $p_T$)", overflow=True
        ),
        "lep_misolation_ele": hist2.axis.Regular(35, 0, 0.2, name="var", label=r"Electron mini-isolation", overflow=True),
        "lep_misolation_ele_highpt": hist2.axis.Regular(
            35, 0, 0.2, name="var", label=r"Electron mini-isolation (high $p_T$)", overflow=True
        ),
        "lep_misolation_ele_lowpt": hist2.axis.Regular(
            35, 0, 0.2, name="var", label=r"Electron mini-isolation (low $p_T$)", overflow=True
        ),
    }
    return axis_dict[var]


def plot_hists(
    hists,
    years,
    channels,
    vars_to_plot,
    add_data,
    logy,
    mult,
    outpath,
    text_="",
    blind_region=None,
    save_as=None,
    plot_Fake_unc=None,
    plot_syst_unc=None,
    add_soverb=False,
    legend_ncol=2,
    seperate_Fake_unc=False,
    plot_totsig=True,
    plot_sig_components=False,
):
    # luminosity
    luminosity = 0
    for year in years:
        lum = 0
        for ch in channels:
            with open("../fileset/luminosity.json") as f:
                lum += json.load(f)[ch][year] / 1000.0

        luminosity += lum / len(channels)

    for var in vars_to_plot:

        if var not in hists.keys():
            print(f"{var} not stored in hists")
            continue

        print(f"Will plot {var} histogram")

        # get histograms
        h = hists[var]

        if h.shape[0] == 0:  # skip empty histograms (such as lepton_pt for hadronic channel)
            print("Empty histogram ", var)
            continue

        # get samples existing in histogram
        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
        signal_labels = [label for label in samples if label in signals]
        bkg_labels = [label for label in samples if (label and label not in signal_labels and (label not in ["Data"]))]

        # get total yield of backgrounds per label
        # (sort by yield in fixed fj_pt histogram after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            if "fj_pt" in hists.keys():
                order_dic[plot_labels[bkg_label]] = hists["fj_pt"][{"samples": bkg_label}].sum()
            else:
                order_dic[plot_labels[bkg_label]] = hists[var][{"samples": bkg_label}].sum()

        if add_data:
            data = h[{"samples": "Data"}]
        bkg = [h[{"samples": label}] for label in bkg_labels]
        signal = [h[{"samples": label}] for label in signal_labels]

        # scale signal for non-log plots
        if logy:
            mult_factor = 1
        else:
            mult_factor = mult
        signal_mult = [s * mult_factor for s in signal]

        # ----------------------------- initialize a figure
        if add_soverb:
            if add_data and data and len(bkg) > 0:
                if len(signal) > 0:
                    fig, (ax, rax, sax, dax) = plt.subplots(
                        nrows=4,
                        ncols=1,
                        figsize=(9, 9),
                        gridspec_kw={"height_ratios": (4, 1, 1, 1), "hspace": 0.07},
                        sharex=True,
                    )
            else:
                fig, (ax, sax, dax) = plt.subplots(
                    nrows=3,
                    ncols=1,
                    figsize=(9, 9),
                    gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.07},
                    sharex=True,
                )
                rax = None
        else:
            if add_data and data and len(bkg) > 0:
                fig, (ax, rax) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    figsize=(8, 8),
                    gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
                    sharex=True,
                )
                sax = None
                dax = None
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                rax = None
                sax = None
                dax = None

        if seperate_Fake_unc:
            errps = {
                "hatch": "////",
                "facecolor": "none",
                "lw": 0,
                "color": "k",
                "edgecolor": (0, 0, 0, 0.5),
                "linewidth": 0,
                "alpha": 0.9,
            }
            errps_fake = {
                "hatch": "////",
                "facecolor": "tab:orange",
                "lw": 3,
                "color": "tab:orange",
                "linewidth": 0,
                "alpha": 0.5,
            }
        else:
            errps = {
                "hatch": "////",
                "facecolor": "none",
                "lw": 0,
                "color": "k",
                "edgecolor": (0, 0, 0, 0.5),
                "linewidth": 0,
                "alpha": 0.4,
            }

        # sum all of the background and get the stat. and the syst. unc.
        if len(bkg) > 0:
            tot = bkg[0].copy()
            for i, b in enumerate(bkg):
                if i > 0:
                    tot = tot + b

            tot_val = tot.values()
            tot_val_zero_mask = tot_val == 0  # check if this is for the ratio or not
            tot_val[tot_val_zero_mask] = 1

            tot_err_MC = np.sqrt(tot.variances())

            if plot_Fake_unc and ("Fake" in h.axes["samples"]):
                tot_err_Fake = h[{"samples": "Fake"}].values() * plot_Fake_unc

            if plot_syst_unc:
                tot_err_SYST_up = tot.values() * plot_syst_unc[0][var]
                tot_err_SYST_down = tot.values() * plot_syst_unc[1][var]

        if add_data and data:
            data_err_opts = {
                "linestyle": "none",
                "marker": ".",
                "markersize": 10.0,
                "elinewidth": 1,
            }

            if blind_region and (("rec_higgs" in var) or ("ParT_mass" in var) or ("fj_mass" in var)):
                if var == "fj_mass":
                    blind_region = [50, 130]
                else:
                    blind_region = [90, 160]

                massbins = data.axes[-1].edges
                lv = int(np.searchsorted(massbins, blind_region[0], "right"))
                rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)

                data.view(flow=True)[lv:rv].value = 0
                data.view(flow=True)[lv:rv].variance = 0

            tot_err_data = np.sqrt(data.values())
            hep.histplot(
                data,
                ax=ax,
                histtype="errorbar",
                color="k",
                capsize=4,
                yerr=tot_err_data,
                label="Data",
                **data_err_opts,
                flow="none",
            )

            if len(bkg) > 0:

                data_val = data.values()
                data_val[tot_val_zero_mask] = 1

                # from hist.intervals import ratio_uncertainty
                # yerr = ratio_uncertainty(data_val, tot_val, "poisson")
                yerr = np.sqrt(data_val) / tot_val

                hep.histplot(
                    data_val / tot_val,
                    tot.axes[0].edges,
                    yerr=yerr,
                    ax=rax,
                    histtype="errorbar",
                    color="k",
                    capsize=4,
                    flow="none",
                )

                # plot the total unc.
                unc_lab = "Stat."
                unc_up = (tot_err_MC / tot_val) ** 2
                unc_down = (tot_err_MC / tot_val) ** 2

                if plot_Fake_unc and ("Fake" in h.axes["samples"]):

                    if seperate_Fake_unc:
                        unc_fake_up = (tot_err_Fake / tot_val) ** 2
                        unc_fake_down = (tot_err_Fake / tot_val) ** 2
                    else:
                        unc_up += (tot_err_Fake / tot_val) ** 2
                        unc_down += (tot_err_Fake / tot_val) ** 2

                if plot_syst_unc:
                    unc_lab += " + Syst."
                    unc_up += (tot_err_SYST_up / tot_val) ** 2
                    unc_down += (tot_err_SYST_down / tot_val) ** 2

                unc_lab += " unc."

                rax.stairs(
                    values=1 + np.sqrt(unc_up),
                    baseline=1 - np.sqrt(unc_down),
                    edges=tot.axes[0].edges,
                    **errps,
                    label=unc_lab,
                )
                if seperate_Fake_unc:
                    rax.stairs(
                        values=(1 + np.sqrt(unc_up)) + np.sqrt(unc_fake_up),
                        baseline=(1 - np.sqrt(unc_up)) - np.sqrt(unc_fake_down),
                        edges=tot.axes[0].edges,
                        **errps_fake,
                        label="Fake normalization unc.",
                    )

                rax.axhline(1, ls="--", color="k")
                rax.set_ylim(0.2, 1.8)

                # chi-square test
                obs_ratio = data_val / tot_val  # Data/MC
                exp_ratio = 1  # we want to test against Data/MC=1

                msk = obs_ratio > exp_ratio  # if obs>exp then use sigma=unc_up else sigma=unc_down
                sigma = np.where(msk, np.sqrt(unc_up), np.sqrt(unc_down))

                chi_square = ((obs_ratio - exp_ratio) ** 2 / sigma**2).sum()

                num_dof = len(obs_ratio) - 1  # num bins - 1
                print(
                    f"#bins: {len(obs_ratio)}, chi_square: {chi_square:.3f}, reduced_chi_square: {(chi_square/num_dof):.3f}"
                )

        # plot the background
        if len(bkg) > 0:

            hep.histplot(
                bkg,
                ax=ax,
                stack=True,
                sort="yield",
                edgecolor="black",
                linewidth=1,
                histtype="fill",
                label=[plot_labels[bkg_label] for bkg_label in bkg_labels],
                color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
                flow="none",
            )

            # plot the total unc.
            unc_lab = "Stat."
            unc_up = tot_err_MC**2
            unc_down = tot_err_MC**2

            if plot_Fake_unc and ("Fake" in h.axes["samples"]):
                if seperate_Fake_unc:
                    unc_fake_up = (tot_err_Fake) ** 2
                    unc_fake_down = (tot_err_Fake) ** 2

                else:
                    unc_up += tot_err_Fake**2
                    unc_down += tot_err_Fake**2

            if plot_syst_unc:
                unc_lab += " + Syst."
                unc_up += tot_err_SYST_up**2
                unc_down += tot_err_SYST_down**2

            unc_lab += " unc."

            ax.stairs(
                values=tot.values() + np.sqrt(unc_up),
                baseline=tot.values() - np.sqrt(unc_down),
                edges=tot.axes[0].edges,
                **errps,
                label=unc_lab,
            )

            if seperate_Fake_unc:
                ax.stairs(
                    values=(tot.values() + np.sqrt(unc_up)) + np.sqrt(unc_fake_up),
                    baseline=(tot.values() - np.sqrt(unc_up)) - np.sqrt(unc_fake_down),
                    edges=tot.axes[0].edges,
                    **errps_fake,
                    label="Fake normalization unc.",
                )

        # plot the signal (times 10)
        if len(signal) > 0:
            tot_signal = None
            for i, sig in enumerate(signal_mult):
                if tot_signal is None:
                    tot_signal = signal[i].copy()
                else:
                    tot_signal = tot_signal + signal[i]

            # plot the total signal (w/o scaling)
            tot_signal *= mult_factor

            if mult_factor == 1:
                siglabel = r"ggF+VBF+WH+ZH+ttH"
            else:
                siglabel = r"ggF+VBF+WH+ZH+ttH $\times$" + f"{mult_factor}"

            if plot_totsig:
                hep.histplot(
                    tot_signal,
                    ax=ax,
                    label=siglabel,
                    linewidth=2,
                    color="tab:red",
                    flow="none",
                )
            # # add MC stat errors
            # ax.stairs(
            #     values=tot_signal.values() + np.sqrt(tot_signal.values()),
            #     baseline=tot_signal.values() - np.sqrt(tot_signal.values()),
            #     edges=sig.axes[0].edges,
            #     **errps_signal,
            # )

            if plot_sig_components:
                # plot individual signal
                for i, sig in enumerate(signal_mult):
                    hep.histplot(
                        sig,
                        ax=ax,
                        label=signal_labels[i] if mult_factor == 1 else signal_labels[i] + r"$\times$" + f"{mult_factor}",
                        linewidth=3,
                        color=color_by_sample[signal_labels[i]],
                        flow="none",
                    )

            if add_soverb:
                totsignal_val = tot_signal.values()
                # replace values where bkg is 0
                totsignal_val[tot_val == 0] = 0
                soverb_val = totsignal_val / (tot_val)

                tot_soversqrtb = ((totsignal_val[:4] / mult_factor).sum()) / np.sqrt(tot_val[:4].sum())
                # sax.legend(title=f"S/sqrt(B) in [75-155GeV]={tot_soversqrtb:.2f}")

                hep.histplot(
                    soverb_val,
                    tot_signal.axes[0].edges,
                    label="Total Signal",
                    ax=sax,
                    linewidth=3,
                    color="tab:red",
                    flow="none",
                )

                for i, sig in enumerate(signal_mult):
                    lab_sig_mult = f"{mult_factor} * {plot_labels[signal_labels[i]]}"
                    if mult_factor == 1:
                        lab_sig_mult = f"{plot_labels[signal_labels[i]]}"
                    hep.histplot(
                        sig,
                        ax=dax,
                        label=lab_sig_mult,
                        linewidth=3,
                        color=color_by_sample[signal_labels[i]],
                        flow="none",
                    )

        ax.set_ylabel("Events")
        if "Data" in samples:
            ax.set_xlabel("")  # because the bottom plot will have the label

        if rax is not None:
            rax.set_ylabel("Ratio", fontsize=20, labelpad=15)
            rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

        if add_soverb:
            if rax is not None:
                rax.set_xlabel("")
            sax.set_ylabel(r"S/B", fontsize=20, y=0.4, labelpad=0)
            dax.set_ylabel(r"S", fontsize=20, y=0.4, labelpad=8)
            dax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

        # get handles and labels of legend
        handles, labels = ax.get_legend_handles_labels()

        # append legend labels in order to a list
        summ = []
        for label in labels[: len(bkg_labels)]:
            summ.append(order_dic[label].value)
        # get indices of labels arranged by yield
        order = []
        for i in range(len(summ)):
            order.append(np.argmax(np.array(summ)))
            summ[np.argmax(np.array(summ))] = -100

        # plot bkg, then signal, then data
        hand = [handles[i] for i in order] + handles[len(bkg) : -1] + [handles[-1]]
        lab = [labels[i] for i in order] + labels[len(bkg) : -1] + [labels[-1]]

        lab_new, hand_new = [], []
        for i in range(len(lab)):
            lab_new.append(lab[i])
            hand_new.append(hand[i])

        if add_soverb:
            ax.legend(
                [hand[idx] for idx in range(len(hand))],
                [lab[idx] for idx in range(len(lab))],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

            # ax.legend(
            #     [hand_new[idx] for idx in range(len(hand_new))],
            #     [lab_new[idx] for idx in range(len(lab_new))],
            #     title=text_,
            #     ncol=legend_ncol,
            #     fontsize=14,
            #     loc="upper left",
            # )

        else:
            ax.legend(
                [hand_new[idx] for idx in range(len(hand_new))],
                [lab_new[idx] for idx in range(len(lab_new))],
                title=text_,
                ncol=legend_ncol,
                fontsize=14,
                loc="upper left",
            )

        _, a = ax.get_ylim()
        # if logy or ("isolation" in var) or ("lsf3" in var):
        if logy or ("isolation" in var) or ("lsf3" in var) or ("THWW" in var):
            ax.set_yscale("log")
            # ax.set_ylim(1e-1, a * 15.7)
            ax.set_ylim(1e-1, a * 105.7)
        else:
            ax.set_ylim(0, a * 1.7)

        if "Num" in var:
            ax.set_xlim(h.axes["var"].edges[0], h.axes["var"].edges[-1])
        else:
            ax.set_xlim(h.axes["var"].edges[0], h.axes["var"].edges[-1])

        if add_soverb:
            _, a = dax.get_ylim()
            dax.set_ylim(0, a + 4)

            if a > 1000:
                dax.set_yticks([800])
            if a > 800:
                dax.set_yticks([600])
            elif a > 700:
                dax.set_yticks([500])
            elif a > 500:
                dax.set_yticks([300])
            elif a > 300:
                dax.set_yticks([200])
            elif a > 300:
                dax.set_yticks([200])
            elif a > 180:
                dax.set_yticks([160])
            elif a > 140:
                dax.set_yticks([120])
            elif a > 100:
                dax.set_yticks([90])
            elif a > 40:
                dax.set_yticks([40])
            elif a > 20:
                dax.set_yticks([20])
            elif a > 10:
                dax.set_yticks([10])
            elif a > 5:
                dax.set_ylim(0, a + 2)
                dax.set_yticks([5])
            elif a > 3:
                dax.set_ylim(0, a + 1)
                dax.set_yticks([3])
            elif a > 2:
                dax.set_ylim(0, a + 1)
                dax.set_yticks([2])
            else:
                dax.set_ylim(0, a + 1)
                dax.set_yticks([1])

        hep.cms.lumitext("%.0f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
        hep.cms.text("Work in Progress", ax=ax, fontsize=15)

        # save plot
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if save_as:
            plt.savefig(f"{outpath}/{save_as}_stacked_hists_{var}.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"{outpath}/stacked_hists_{var}.pdf", bbox_inches="tight")
