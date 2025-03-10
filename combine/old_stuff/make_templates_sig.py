"""
Follows a similar approach to ```make_templates.py``` except that
    (1) it doesn't care about any systematics except the nominal
    (2) the "Region" axis takes care of the regions to be tested for significance

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils_diffBins
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "QCD", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (fj_ParT_score>0.97)}`)
        model_path [str]: path to the ParT finetuned model.onnx

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (samples, systematic, ptbin, mass_observable)

    """
    lepiso_filter = {
        "lepiso": {
            "mu": [
                [("lep_pt", "<", 55), (("lep_isolation", "<", 0.15))],
                [("lep_misolation", "<", 0.2), ("lep_pt", ">=", 55)],
            ],
            "ele": [
                [("lep_pt", "<", 120), (("lep_isolation", "<", 0.15))],
                [("lep_pt", ">=", 120)],
            ],
        },
        "lepisoinv": {
            "mu": [
                [("lep_pt", "<", 55), (("lep_isolation", ">", 0.15))],
                [("lep_misolation", ">", 0.2), ("lep_pt", ">=", 55)],
            ],
            "ele": [
                [("lep_pt", "<", 120), (("lep_isolation", ">", 0.15))],
                [("lep_pt", ">=", 120)],
            ],
        },
    }

    presel = {
        "mu": {
            "lep_fj_dr003": "( ( lep_fj_dr>0.03) )",
            "lep_fj_dr08": "( ( lep_fj_dr<0.8) )",
            "fj_pt300": "( ( fj_pt>250) )",
            "dphi<1.57": "(abs_met_fj_dphi<1.57)",
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
            "MET>25": "met_pt>25",
        },
        "ele": {
            "lep_fj_dr003": "( ( lep_fj_dr>0.03) )",
            "lep_fj_dr08": "( ( lep_fj_dr<0.8) )",
            "fj_pt300": "( ( fj_pt>250) )",
            "dphi<1.57": "(abs_met_fj_dphi<1.57)",
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
            "MET>30": "met_pt>30",
        },
    }

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            list(range(50, 240, 20)), name="mass_observable", label=r"Higgs reconstructed mass [GeV]", overflow=True
        ),
        storage=hist2.storage.Weight(),
    )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):
                # if sample == "QCD_Pt_170to300":
                #     print(f"Skipping sample {sample}")
                #     continue

                if "DYJetsToLL_M-50_HT" in sample:
                    print(f"Skipping sample {sample}")
                    continue

                for key in utils_diffBins.combine_samples:  # get a combined label to combine samples of the same process
                    if key in sample:
                        sample_to_use = utils_diffBins.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                if "postprocess" in samples_dir[year]:
                    parquet_files = glob.glob(f"{out_files}/{ch}.parquet")
                else:
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files, filters=lepiso_filter["lepiso"][ch])
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["fj_ParT_score_finetuned"] = utils_diffBins.get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                data["abs_met_fj_dphi"] = np.abs(data["met_fj_dphi"])

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                # get event_weight
                if sample_to_use != "Data":
                    event_weight = utils_diffBins.get_xsecweight(pkl_files, year, sample, False, luminosity)

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    logging.info(f"Applying {region} selection on {len(data)} events")

                    if ("SR" in region) and ("QCD" in sample):  # literally a single qcd event
                        threshold = 30
                        data = data[(event_weight * data[f"weight_{ch}"]) < threshold]

                    df = data.copy().query(region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                    # nominal weight
                    if sample_to_use == "Data":  # for data (fill as 1)
                        hists.fill(
                            Sample=sample_to_use,
                            Systematic="nominal",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=np.ones_like(df["fj_pt"]),
                        )
                    else:
                        nominal = df[f"weight_{ch}"] * event_weight
                        hists.fill(
                            Sample=sample_to_use,
                            Systematic="nominal",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=nominal,
                        )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 0
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates_sig.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years, channels, config["samples"], config["samples_dir"], config["regions_sel"], config["model_path"]
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates_sig.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
