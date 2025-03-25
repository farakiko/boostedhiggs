import glob
import json
import logging
import os
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


combine_samples = {
    "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr": "WJetsToLNu NLO 012JetsNLO 34JetsLO",
    "WJetsToLNu_HT": "WJetsToLNu LO (HT)",
    "WJetsToLNu_1J": "WJetsToLNu NLO (LHEFilterPtW)",
    "WJetsToLNu_2J": "WJetsToLNu NLO (LHEFilterPtW)",
    "WJetsToLNu_LHEFilterPtW": "WJetsToLNu NLO (LHEFilterPtW)",
    "_MatchEWPDG20": "WJetsToLNu NLO (MatchEWPDG20)",
}

# combine_samples = {
#     "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr": "WJetsToLNu NLO 012JetsNLO 34JetsLO",
#     "WJetsToLNu_HT-100To200": "WJetsToLNu_HT-100To200",
#     "WJetsToLNu_HT-200To400": "WJetsToLNu_HT-200To400",
#     "WJetsToLNu_HT-400To600": "WJetsToLNu_HT-400To600",
#     "WJetsToLNu_HT-600To800": "WJetsToLNu_HT-600To800",
#     "WJetsToLNu_HT-800To1200": "WJetsToLNu_HT-800To1200",
#     "WJetsToLNu_HT-1200To2500": "WJetsToLNu_HT-1200To2500",
#     "WJetsToLNu_HT-2500ToInf": "WJetsToLNu_HT-2500ToInf",
#     "WJetsToLNu_1J": "WJetsToLNu NLO (1,2 J)",
#     "WJetsToLNu_2J": "WJetsToLNu NLO (1,2 J)",
#     "WJetsToLNu_LHEFilterPtW-250To400": "WJetsToLNu_LHEFilterPtW-250To400",
#     "WJetsToLNu_LHEFilterPtW-400To600": "WJetsToLNu_LHEFilterPtW-400To600",
#     "WJetsToLNu_LHEFilterPtW-600ToInf": "WJetsToLNu_LHEFilterPtW-600ToInf",
#     "WJetsToLNu_Pt-100To250_MatchEWPDG20": "WJetsToLNu_Pt-100To250_MatchEWPDG20",
#     "WJetsToLNu_Pt-250To400_MatchEWPDG20": "WJetsToLNu_Pt-250To400_MatchEWPDG20",
#     "WJetsToLNu_Pt-400To600_MatchEWPDG20": "WJetsToLNu_Pt-400To600_MatchEWPDG20",
#     "WJetsToLNu_Pt-600ToInf_MatchEWPDG20": "WJetsToLNu_Pt-600ToInf_MatchEWPDG20",
# }


def get_common_sample_name(sample):

    for key in combine_samples:
        if key in sample:
            sample_to_use = combine_samples[key]
            break
        else:
            sample_to_use = sample
    return sample_to_use


def make_templates(years, channels, samples_dir, samples, presel):
    """
    Postprocess the parquets by applying preselection, saving a `nominal` weight column, and
    saving a THWW tagger score column, all in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [dict]: key=year, value=str pointing to the path of the parquets
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)

    Returns
        a dict() object events_dict[year][channel][samples] that contains a dataframe of procesed events.

    """

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.Variable(
            np.linspace(0, 1600, 60),
            name="gen_V_pt",
            label=r"Gen V pT [GeV]",
            overflow=True,
            underflow=True,
        ),
        hist2.axis.Variable(
            np.linspace(0, 1600, 60),
            name="LHE_Vpt",
            label=r"LHE V pT [GeV]",
            overflow=True,
            underflow=True,
        ),
        hist2.axis.Variable(
            np.linspace(0, 1600, 60),
            name="LHE_HT",
            label=r"LHE HT [GeV]",
            overflow=True,
            underflow=True,
        ),
        storage=hist2.storage.Weight(),
    )

    for year in years:
        for ch in channels:
            logging.info(f"Processing {year} {ch} channel")

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

                # get a combined label to combine samples of the same process
                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                parquet_files = glob.glob(f"{samples_dir[year]}/{sample}/outfiles/*_{ch}.parquet")
                pkl_files = glob.glob(f"{samples_dir[year]}/{sample}/outfiles/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue
                for parquet_file in parquet_files:
                    try:
                        data = pd.read_parquet(parquet_file)
                    except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                        continue

                    if len(data) == 0:
                        continue

                    if sample_to_use == "WJetsToLNu NLO (LHEFilterPtW)":
                        if ("WJetsToLNu_1J" in sample) or ("WJetsToLNu_2J" in sample):
                            data = data[data["LHE_Vpt"] < 250]
                        else:
                            data = data[data["LHE_Vpt"] > 250]

                    # if sample_to_use == "WJetsToLNu NLO (MatchEWPDG20)":
                    #     if ("WJetsToLNu_1J" in sample) or ("WJetsToLNu_2J" in sample):
                    #         data = data[data["gen_V_pt"] < 100]
                    #     else:
                    #         data = data[data["gen_V_pt"] > 100]

                    # get event_weight
                    try:
                        data["xsecweight"] = utils.get_xsecweight(pkl_files, year, sample, False, luminosity)
                    except EOFError:
                        continue
                    data["nominal"] = data["xsecweight"] * data[f"weight_{ch}"]

                    if sample in ["WJetsToLNu_1J, WJetsToLNu_2J"]:
                        data["nominal"] /= data["weight_ewkcorr"]

                    if "WJetsToLNu_HT-" in sample:
                        data["nominal"] /= data["weight_ewkcorr"]
                        data["nominal"] /= data["weight_qcdcorr"]

                    # apply preselection
                    for selection in presel[ch]:
                        data = data.query(presel[ch][selection])

                    hists.fill(
                        Sample=sample_to_use,
                        gen_V_pt=data["gen_V_pt"],
                        LHE_Vpt=data["LHE_Vpt"] if "LHE_Vpt" in data else 0,
                        LHE_HT=data["LHE_HT"] if "LHE_HT" in data else 0,
                        weight=data["nominal"],
                    )

    return hists
