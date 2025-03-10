"""
Loads the config from `config_make_stacked_hists.yaml`, and postprocesses
the condor output to make stacked histograms

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
import onnx
import onnxruntime as ort
import pandas as pd
import pyarrow
import scipy
import utils_diffBins
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


sel = {
    "signal_region": {
        "mu": [[("lep_pt", "<", 55), (("lep_isolation", "<", 0.15))], [("lep_misolation", "<", 0.2), ("lep_pt", ">=", 55)]],
        "ele": [[("lep_pt", "<", 120), (("lep_isolation", "<", 0.15))], [("lep_pt", ">=", 120)]],
    },
    "qcd_region": {
        "mu": [[("lep_pt", "<", 55), (("lep_isolation", ">", 0.15))], [("lep_misolation", ">", 0.2), ("lep_pt", ">=", 55)]],
        "ele": [[("lep_pt", "<", 120), (("lep_isolation", ">", 0.15))], [("lep_pt", ">=", 120)]],
    },
}


input_feat = {
    "v2-1111-10noMass1": [
        "fj_pt",
        "fj_msoftdrop",
        "met_relpt",
        # "met_fj_dphi",
        "lep_fj_dr",
        "n_bjets_L",
        "n_bjets_M",
        "n_bjets_T",
        "lep_isolation",
        "lep_misolation",
    ],
    "v2_1-12": [
        "fj_pt",
        "fj_msoftdrop",
        "met_relpt",
        # "met_fj_dphi",
        "lep_fj_dr",
        "n_bjets_L",
        "n_bjets_M",
        "n_bjets_T",
        "lep_isolation",
        "lep_misolation",
    ],
    # "v2_10_5": [],
    # "v2_10_12": [],
    # "v2_nor1": [],
    "v2_nor2": [],
}


def make_events_dict(
    years, channels, samples_dir, samples, presel, logging_=True, region="signal_region", keep_weights=False
):
    """
    Postprocess the parquets by applying preselections, saving an `event_weight` column, and
    a tagger score column in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)

    Returns
        a dict() object events_dict[year][channel][samples] that contains big dataframes of procesed events

    """

    if logging_ is False:
        logger = logging.getLogger()
        logger.disabled = True

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            events_dict[year][ch] = {}

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            condor_dir = os.listdir(samples_dir)

            for sample in condor_dir:
                if sample == "DYJetsToLL_M-10to50":
                    continue  # because tagger didnt run for it for one of the years

                # get a combined label to combine samples of the same process
                for key in utils_diffBins.combine_samples:
                    if key in sample:
                        sample_to_use = utils_diffBins.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir}/{sample}/outfiles/"
                if "postprocess" in samples_dir:
                    parquet_files = glob.glob(f"{out_files}/{ch}.parquet")
                else:
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")

                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files, filters=sel[region][ch])

                except pyarrow.lib.ArrowInvalid:
                    # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                if not keep_weights:
                    data = data[data.columns.drop(list(data.filter(regex="weight_mu_")))]
                    data = data[data.columns.drop(list(data.filter(regex="weight_ele_")))]
                    data = data[data.columns.drop(list(data.filter(regex="L_btag")))]
                    data = data[data.columns.drop(list(data.filter(regex="M_btag")))]
                    data = data[data.columns.drop(list(data.filter(regex="T_btag")))]
                    data = data[data.columns.drop(list(data.filter(regex="veto")))]
                    data = data[data.columns.drop(list(data.filter(regex="fj_H_VV_")))]
                    data = data[data.columns.drop(list(data.filter(regex="_up")))]
                    data = data[data.columns.drop(list(data.filter(regex="_down")))]

                # get event_weight
                if sample_to_use != "Data":
                    event_weight = utils_diffBins.get_xsecweight(pkl_files, year, sample, False, luminosity)

                    logging.info("---> Using already stored event weight")
                    event_weight *= data[f"weight_{ch}"]

                else:
                    event_weight = np.ones_like(data["fj_pt"])

                data["event_weight"] = event_weight

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                logging.info(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
                logging.info(f"tot event weight {data['event_weight'].sum()} \n")

                # add finetuned tagger score
                for modelv, inp in input_feat.items():
                    PATH = f"../../weaver-core-dev/experiments_finetuning/{modelv}/model.onnx"

                    data["met_relpt"] = data["met_pt"] / data["fj_pt"]

                    if modelv in ["v2-1111-10noMass1", "v2_1-12"]:
                        input_dict = {
                            "basic": data.loc[:, inp].values.astype("float32"),
                            "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
                        }
                    else:
                        input_dict = {
                            # "basic": 0,
                            "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
                        }

                    onnx_model = onnx.load(PATH)
                    onnx.checker.check_model(onnx_model)

                    ort_sess = ort.InferenceSession(
                        PATH,
                        providers=["AzureExecutionProvider"],
                    )
                    outputs = ort_sess.run(None, input_dict)

                    if modelv == "v2-1111-10noMass1":
                        data["fj_ParT_score_finetuned"] = scipy.special.softmax(outputs[0], axis=1)[:, 0]
                    elif modelv == "v2_1-12":
                        data["fj_ParT_score_finetuned_HPO"] = scipy.special.softmax(outputs[0], axis=1)[:, 0]
                    else:
                        data[f"fj_ParT_score_finetuned_{modelv}"] = scipy.special.softmax(outputs[0], axis=1)[:, 0]

                # drop hidNeuron columns for memory
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # fill the big dataframe
                if sample_to_use not in events_dict[year][ch]:
                    events_dict[year][ch][sample_to_use] = data
                else:
                    events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data])

    return events_dict


def make_hists_from_events_dict(events_dict, samples_to_plot, vars_to_plot, selections):
    """
    Takes an `events_dict` object that was processed by `make_events_dict` and starts filling histograms.

    Args
        events_dict [dict]: see output of `make_events_dict()`
        samples_to_plot [list]: which samples to use when plotting
        vars_to_plot [list]: which variables to plot

    Returns
        hist.Hist object

    """

    hists = {}
    for var in vars_to_plot:
        hists[var] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            utils_diffBins.axis_dict[var],
        )

        for sample in samples_to_plot:
            for year in events_dict:
                for ch in events_dict[year]:
                    df = events_dict[year][ch][sample]

                    for sel, value in selections[ch].items():
                        df = df.query(value)

                    hists[var].fill(samples=sample, var=df[var], weight=df["event_weight"])

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp config_make_stacked_hists.yaml {args.outpath}/")

    # load config from yaml
    with open("config_make_stacked_hists.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.make_events_dict:
        events_dict = make_events_dict(
            years,
            channels,
            args.samples_dir,
            config["samples"],
            config["presel"],
        )
        with open(f"{args.outpath}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{args.outpath}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.info("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    if args.plot_hists:
        PATH = args.outpath + f"stacked_hists_{args.tag}"
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        os.system(f"cp config_make_stacked_hists.yaml {PATH}/")

        logging.info("##### SELECTIONS")
        for ch in config["sel"]:
            logging.info(f"{ch} CHANNEL")
            for sel, value in config["sel"][ch].items():
                logging.info(f"{sel}: {value}")
            logging.info("-----------------------------")

        hists = make_hists_from_events_dict(events_dict, config["samples_to_plot"], config["vars_to_plot"], config["sel"])

        utils_diffBins.plot_hists(
            years,
            channels,
            hists,
            config["vars_to_plot"],
            config["add_data"],
            config["logy"],
            config["add_soverb"],
            config["only_sig"],
            config["mult"],
            outpath=PATH,
        )


if __name__ == "__main__":
    # e.g.
    # python finetuned_make_stacked_hists.py --years 2017 --channels ele,mu --plot_hists --make_events_dict --tag v1
    # python finetuned_make_stacked_hists.py --years 2017 --channels ele,mu --plot_hists --tag v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--samples_dir", dest="samples_dir", default="../eos/Jul21_2017", help="path to parquets", type=str)
    parser.add_argument("--outpath", dest="outpath", default="hists/", help="path of the output", type=str)
    parser.add_argument("--tag", dest="tag", default="test/", help="path of the output", type=str)
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="Make events dictionary", action="store_true")
    parser.add_argument("--plot_hists", dest="plot_hists", help="Plot histograms", action="store_true")

    args = parser.parse_args()

    main(args)
