"""
Computes the LateX cutflow tables in the AN.

It does so in a few steps,
    1. Will load the pkl files that contain the cutflows and the sumgenweight
    2. Will scale the events by the cross section
    3. Will save the yields in a dictionary called `cutflows -> Dict()`
    4. Will make the LateX table using the function `make_composition_table()`

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import pyarrow
import yaml

sys.path
sys.path.append("../python/")

import utils_diffBins
from utils_diffBins import get_xsecweight

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

pd.options.mode.chained_assignment = None


cut_to_label = {
    "sumgenweight": "sumgenweight",
    "HEMCleaning": "HEMCleaning",
    "Trigger": "Trigger",
    "METFilters": "METFilters",
    "OneLep": "n Leptons = 1",
    "NoTaus": "n Taus = 0",
    "AtLeastOneFatJet": r"n FatJets $>=$ 1",
    "CandidateJetpT": r"j $p_T > 250$GeV",
    "LepInJet": r"$\Delta R(j, \ell) < 0.8$",
    "JetLepOverlap": r"$\Delta R(j, \ell) > 0.03$",
    "dPhiJetMET": r"$\Delta \phi(\mathrm{MET}, j)<1.57$",
    "MET": r"$\mathrm{MET}>20$",
    "None": "None",
    "msoftdrop": r"j $\mathrm{softdrop} > 40$GeV",
    "THWW": r"$\ensuremath{T_{\text{HWW}}^{\ell\nu qq}} > 0.75$",
}

parquet_to_latex = {
    "WJetsLNu": "$\PW(\Pell\PGn)$+",
    "TTbar": "\\ttbar",
    "Others": "Other MC",
    "ggF": "ggF",
    "VBF": "VBF",
    "WH": "WH",
    "ZH": "ZH",
    "ttH": "$t\\bar{t}H$",
    "Data": "Data",
}


def get_lumi(years, channels):
    # get lumi
    with open("../fileset/luminosity.json") as f:
        luminosity = json.load(f)

    lum_ = 0
    for year in years:
        lum = 0
        for ch in channels:
            lum += luminosity[ch][year] / 1000.0

        lum_ += lum / len(channels)
    return lum_


def get_cutflow(pkl_files, year, ch, sample, is_data):
    """Get cutflow from metadata but multiply by xsec-weight."""

    with open("../fileset/luminosity.json") as f:
        luminosity = json.load(f)[ch][year]

    xsec_weight = get_xsecweight(pkl_files, year, sample, is_data, luminosity)

    cuts = [
        "sumgenweight",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "AtLeastOneFatJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMET",
        "MET",
    ]

    if year == "2018":
        cuts += ["HEMCleaning"]

    evyield = dict.fromkeys(cuts, 0)
    for ik, pkl_file in enumerate(pkl_files):
        with open(pkl_file, "rb") as f:
            metadata = pkl.load(f)

        cutflows = metadata[sample][year]["cutflows"][ch]

        for key in evyield.keys():

            if key == "sumgenweight":
                evyield[key] += metadata[sample][year][key] * xsec_weight
            else:
                evyield[key] += cutflows[key] * xsec_weight
    return evyield


def make_cutflow_dict(years, channels, samples_dir, samples):
    cutflows = {}
    for year in years:
        print(f"Processing year {year}")

        cutflows[year] = {}

        for ch in channels:
            print(f"  {ch} channel")
            cutflows[year][ch] = {}

            condor_dir = os.listdir(samples_dir[year])

            for sample in condor_dir:

                # first: check if the sample is in one of combine_samples_by_name
                sample_to_use = None
                for key in utils_diffBins.combine_samples_by_name:
                    if key in sample:
                        sample_to_use = utils_diffBins.combine_samples_by_name[key]
                        break

                # second: if not, combine under common label
                if sample_to_use is None:
                    for key in utils_diffBins.combine_samples:
                        if key in sample:
                            sample_to_use = utils_diffBins.combine_samples[key]
                            break
                        else:
                            sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                is_data = False
                if sample_to_use == "Data":
                    is_data = True

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if len(pkl_files) == 0:
                    continue

                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")

                try:
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:
                    # empty parquet because no event passed selection
                    #                 print(f"No parquet file for {sample}")
                    continue

                if len(data) == 0:
                    #                 print(f"Hi, No parquet file for {sample}")
                    continue

                if sample_to_use not in cutflows[year][ch].keys():
                    cutflows[year][ch][sample_to_use] = get_cutflow(pkl_files, year, ch, sample, is_data)
                else:
                    temp = get_cutflow(pkl_files, year, ch, sample, is_data)
                    for key in cutflows[year][ch][sample_to_use]:
                        cutflows[year][ch][sample_to_use][key] += temp[key]

        print("------------------------------------------")

    return cutflows


def add_cut_to_cutflow(cutflows, years, channels, samples, samples_dir, add_cuts, THWW_path):
    from make_stacked_hists import make_events_dict

    # dummy selection at first
    presel = {
        "mu": {},
        "ele": {},
    }

    events_dict = make_events_dict(years, channels, samples_dir, samples, presel, THWW_path)

    for cut, sel in list(add_cuts.items()):
        for year in years:
            for ch in channels:
                for sample in samples:

                    df = events_dict[year][ch][sample]
                    df = df.query(sel)

                    w = df["nominal"]

                    cutflows[year][ch][sample][cut] = w.sum()

    return cutflows


def combine_channels(cutflows):

    # combine both channels
    cutflows_new = {}
    for year in cutflows.keys():
        cutflows_new[year] = {}
        cutflows_new[year]["lep"] = {}

        for ch in ["mu", "ele"]:
            for sample in cutflows[year][ch]:

                if sample not in cutflows_new[year]["lep"]:
                    cutflows_new[year]["lep"][sample] = {}

                for cut in cutflows[year][ch][sample]:

                    if (year != "2018") and (cut == "HEMCleaning"):
                        continue

                    if cut not in cutflows_new[year]["lep"][sample]:
                        cutflows_new[year]["lep"][sample][cut] = cutflows[year][ch][sample][cut]
                    else:
                        cutflows_new[year]["lep"][sample][cut] += cutflows[year][ch][sample][cut]
        cutflows[year] = {**cutflows[year], **cutflows_new[year]}

    return cutflows


def combine_years(cutflows):
    """Will remove the HEM cleaning cutflow from 2018 first."""

    whatever_year = list(cutflows.keys())[0]
    channels = cutflows[whatever_year].keys()

    # combine all years
    cutflows_new = {}
    cutflows_new["Run2"] = {}

    for ch in channels:
        cutflows_new["Run2"][ch] = {}

        for year in cutflows:
            for sample in cutflows[year][ch]:

                if sample not in cutflows_new["Run2"][ch]:
                    cutflows_new["Run2"][ch][sample] = {}

                for cut in cutflows[year][ch][sample]:
                    if "HEM" in cut:
                        continue
                    if cut not in cutflows_new["Run2"][ch][sample]:
                        cutflows_new["Run2"][ch][sample][cut] = cutflows[year][ch][sample][cut]
                    else:
                        cutflows_new["Run2"][ch][sample][cut] += cutflows[year][ch][sample][cut]

    cutflows = {**cutflows, **cutflows_new}

    return cutflows


def combine_nondominant_bkg(cutflows):
    """Combines non-dominant backgrounds under key `Others`."""

    dominant_bkgs = ["WJetsLNu", "TTbar"]
    signals = ["ggF", "VH", "WH", "ZH", "ttH"]

    for year in cutflows:
        for ch in cutflows[year]:
            cutflows[year][ch]["Others"] = dict.fromkeys(cutflows[year][ch]["WJetsLNu"], 0)
            for sample in cutflows[year][ch]:
                if sample == "Data":
                    continue
                if sample not in signals + dominant_bkgs:
                    for cut in cutflows[year][ch][sample]:
                        cutflows[year][ch]["Others"][cut] += cutflows[year][ch][sample][cut]
    return cutflows


def make_latex_cutflow_table(cutflows, year, ch, cuts, add_data=False, add_sumgenweight=False):
    """Will use the cutflows dictionary to make the LateX table we have in the AN."""

    samples_bkg = ["WJetsLNu", "TTbar", "Others"]
    samples_sig = ["ggF", "VBF", "WH", "ZH", "ttH"]

    # backgrounds
    headers = [parquet_to_latex[s] for s in samples_bkg]

    textabular = f"l{'r'*len(headers)}"
    textabular += "|r"

    texheader = "\\textbf{Inclusive Selection}" + " & " + " & ".join(headers) + " & Total MC "
    if add_data:
        textabular += "|r"
        texheader += "& Data "
    texheader += "\\\\"
    texdata = "\\hline\n"

    data = dict()

    for cut in cuts:
        if (year != "2018") and (cut == "HEMCleaning"):
            continue

        if not add_sumgenweight and cut == "sumgenweight":
            continue

        data[cut] = []

        for sample in samples_bkg:
            data[cut].append(round(cutflows[year][ch][sample][cut]))

        totalmc = 0
        for sample in samples_bkg + samples_sig:
            totalmc += round(cutflows[year][ch][sample][cut])

        data[cut].append(totalmc)

        if add_data:
            data[cut].append(round(cutflows[year][ch]["Data"][cut]))

    for label in data:
        if label == "z":
            texdata += "\\hline\n"
        texdata += f"{cut_to_label[label]} & {' & '.join(map(str,data[label]))} \\\\\n"

    texdata += "\\hline\n"

    # signal
    headers2 = [parquet_to_latex[s] for s in samples_sig]
    texheader2 = " & " + " & ".join(headers2) + "\\\\"
    texdata2 = "\\hline\n"

    textabular2 = f"l{'r'*len(headers2)}"

    data = dict()
    for cut in cuts:
        if (year != "2018") and (cut == "HEMCleaning"):
            continue

        data[cut] = []

        for sample in samples_sig:
            data[cut].append(round(cutflows[year][ch][sample][cut]))

    for label in data:
        if label == "z":
            texdata += "\\hline\n"
        texdata2 += f"{cut_to_label[label]} & {' & '.join(map(str,data[label]))} \\\\\n"

    # make table
    print("\\begin{table}[!htp]")
    print("\\begin{center}")

    print("\\begin{tabular}{" + textabular + "}")
    print(texheader)
    print(texdata, end="")
    print("\\end{tabular}")

    print("\\begin{tabular}{" + textabular2 + "}")
    print(texheader2)
    print(texdata2, end="")
    print("\\end{tabular}")

    if ch == "lep":
        print(
            "\\caption{Event yield of "
            + year
            + " Monte Carlo samples normalized to "
            + str(round(get_lumi([year], [ch])))
            + "\\fbinv.}"
        )
    else:
        print(
            "\\caption{Event yield of "
            + ch
            + " channel "
            + year
            + " Monte Carlo samples normalized to "
            + str(round(get_lumi([year], [ch])))
            + "\\fbinv.}"
        )

    print("\\label{sel-tab-cutflow" + year + "}")
    print("\\end{center}")
    print("\\end{table}")


def main(args):

    years = args.years.split(",")
    channels = args.channels.split(",")

    # load the `events_dict_config.yml`
    with open("config_make_cutflow_table.yaml", "r") as stream:
        cutflow_config = yaml.safe_load(stream)

    cutflows = make_cutflow_dict(years, channels, cutflow_config["samples_dir"], cutflow_config["samples"])

    cuts = [
        "sumgenweight",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "AtLeastOneFatJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMET",
        "MET",
        "HEMCleaning",
    ]
    if args.add_cuts:
        cutflows = add_cut_to_cutflow(
            cutflows,
            years,
            channels,
            cutflow_config["samples"],
            cutflow_config["samples_dir"],
            cutflow_config["add_cuts"],
            cutflow_config["THWW_path"],
        )

        for cut in cutflow_config["add_cuts"]:
            cuts += [cut]

    if len(channels) > 1:
        cutflows = combine_channels(cutflows)

    cutflows = combine_years(cutflows)
    cutflows = combine_nondominant_bkg(cutflows)

    if len(channels) > 1:
        channel = "lep"
    else:
        channel = channels[0]

    if len(years) > 1:
        year = "Run2"
    else:
        year = years[0]

    print("---------------------------------------")
    print("---------------------------------------")

    make_latex_cutflow_table(
        cutflows,
        year,
        channel,
        cuts,
        add_data=cutflow_config["add_data"],
        add_sumgenweight=cutflow_config["add_sumgenweight"],
    )


if __name__ == "__main__":
    # e.g.
    # python make_cutflow_table.py --years 2017 --channels ele,mu --add_cuts

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument(
        "--add-cuts",
        dest="add_cuts",
        help="Add additional cuts to the cutflow table",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
