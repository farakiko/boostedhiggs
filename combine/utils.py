#!/usr/bin/python


import json
import pickle as pkl
import warnings
from typing import List

import numpy as np
import scipy
from hist import Hist
from systematics import sigs

warnings.filterwarnings("ignore", message="Found duplicate branch ")

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
    "TT": "TTbar",
    "WJetsToLNu_HT": "WJetsLNu",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "EWK": "EWKvjets",
    "DYJets": "DYJets",
    "JetsToQQ": "WZQQ",
    "WJetsToLNu_1J": "WJetsToLNu_1J",
    "WJetsToLNu_2J": "WJetsToLNu_2J",
    "WJetsToLNu_LHEFilterPtW-250To400": "WJetsToLNu_LHEFilterPtW-250To400",
    "WJetsToLNu_LHEFilterPtW-400To600": "WJetsToLNu_LHEFilterPtW-400To600",
    "WJetsToLNu_LHEFilterPtW-600ToInf": "WJetsToLNu_LHEFilterPtW-600ToInf",    
}

# (name in templates, name in cards)
labels = {
    # sigs
    "ggF": "ggH_hww",
    "ggFpt200to300": "ggH_hww_200_300",
    "ggFpt300to450": "ggH_hww_300_450",
    "ggFpt450toInf": "ggH_hww_450_Inf",
    "VBF": "qqH_hww",
    "ttH": "ttH_hww",
    "WH": "WH_hww",
    "ZH": "ZH_hww",
    "mjj1000toInf": "qqH_hww_mjj_1000_Inf",
    # BKGS
    "WJetsLNu": "wjets",
    "TTbar": "ttbar",
    "SingleTop": "singletop",
    "Diboson": "diboson",
    "EWKvjets": "ewkvjets",
    "DYJets": "zjets",
    "WZQQ": "wzqq",
    "Fake": "fake",
}


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
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_sum_sumpdfweight(pkl_files, year, sample, sample_to_use):

    if (sample_to_use in sigs + ["WJetsLNu", "TTbar"]) and (sample != "ST_s-channel_4f_hadronicDecays"):
        sum_sumpdfweight = {}
        for key in range(103):
            sum_sumpdfweight[key] = 0

        for ifile in pkl_files:
            # load and sum the sumgenweight of each
            with open(ifile, "rb") as f:
                metadata = pkl.load(f)

            for key in range(103):
                sum_sumpdfweight[key] = sum_sumpdfweight[key] + metadata[sample][year]["sumpdfweight"][key]
        return sum_sumpdfweight
    else:
        return 1


def get_sum_sumscsaleweight(pkl_files, year, sample, sample_to_use):

    if (sample_to_use in sigs + ["WJetsLNu", "TTbar", "SingleTop"]) and (sample != "ST_s-channel_4f_hadronicDecays"):
        sum_sumlheweight = {}
        for key in [0, 1, 3, 5, 7, 8, 4]:
            sum_sumlheweight[key] = 0

        for ifile in pkl_files:
            # load and sum the sumgenweight of each
            with open(ifile, "rb") as f:
                metadata = pkl.load(f)

            for key in [0, 1, 3, 5, 7, 8, 4]:
                sum_sumlheweight[key] = sum_sumlheweight[key] + metadata[sample][year]["sumlheweight"][key]
        return sum_sumlheweight
    else:
        return 1


def get_xsecweight(pkl_files, year, sample, sample_to_use, is_data, luminosity):
    """
    Returns the xsec*lumi / [sumgenweight, sumlheweight, or sumpdfweight]
    """
    xsecweight = {}
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
        sumgenweights = get_sum_sumgenweight(pkl_files, year, sample)
        sumpdfweights = get_sum_sumpdfweight(pkl_files, year, sample, sample_to_use)
        sumscaleweights = get_sum_sumscsaleweight(pkl_files, year, sample, sample_to_use)

        xsecweight = (xsec * luminosity) / sumgenweights

        return xsecweight, sumgenweights, sumpdfweights, sumscaleweights

    else:
        return 1, 1, 1, 1


def shape_to_num(var, nom, clip=1.5):
    """
    Estimates the normalized rate from a shape systematic by integrating and dividing by the nominal integrated value.
    """
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)

    if var_rate == nom_rate:  # in cases when both are zero
        return 1

    if abs(var_rate / nom_rate) > clip:
        var_rate = clip * nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate / nom_rate


def get_template(h, sample, region):
    return h[{"Sample": sample, "Systematic": "nominal", "Region": region}]


def get_template_diffbins(h, sample):
    if sample not in h.axes["Sample"]:
        return 0

    return h[{"Sample": sample, "Systematic": "nominal"}]


def load_templates(years, lep_channels, outdir):
    """Loads the hist templates that were created using ```make_templates.py```."""

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(lep_channels) == 1:
        save_as += f"_{lep_channels[0]}_"

    with open(f"{outdir}/hists_templates_{save_as}.pkl", "rb") as f:
        hists_templates = pkl.load(f)

    return hists_templates


def blindBins(h: Hist, blind_region: List = [90, 160], blind_samples: List[str] = []):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_samples`` specified, only blind those samples, else blinds all.

    CAREFUL: assumes axis=0 is samples, axis=3 is mass_axis

    """

    h = h.copy()

    massbins = h.axes["mass_observable"].edges

    lv = int(np.searchsorted(massbins, blind_region[0], "right"))
    rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)
    if len(blind_samples) >= 1:
        for blind_sample in blind_samples:
            sample_index = np.argmax(np.array(list(h.axes[0])) == blind_sample)

            h.view(flow=True)[sample_index, :, :, lv:rv].value = 0
            h.view(flow=True)[sample_index, :, :, lv:rv].variance = 0

    else:
        h.view(flow=True)[:, :, :, lv:rv].value = 0
        h.view(flow=True)[:, :, :, lv:rv].variance = 0

    return h


def get_finetuned_score(data, model_path):
    import onnx
    import onnxruntime as ort

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
