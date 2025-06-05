"""
Loads the config from `config_make_stacked_hists.yaml`, and postprocesses
the condor output to make stacked histograms.

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
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


ptbinning_trgSF = [2000, 200, 170, 150, 130, 110, 90, 70, 50, 30]
etabinning_trgSF = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

with open("../combine/trg_eff_SF_ARC.pkl", "rb") as f:
    TRIGGER_SF = pkl.load(f)


def get_nominal(df, year, ch, xsecweight, is_data):
    """Compute the nominal event weight."""

    if is_data:
        return np.ones_like(df["fj_pt"])

    nominal = df[f"weight_{ch}"] * xsecweight

    if ch == "ele":
        year_tag = f"UL{year[2:].replace('APV', '')}"
        sf_nominal = TRIGGER_SF[year_tag]["nominal"]

        lep_pt = df["lep_pt"].to_numpy()
        lep_eta = np.abs(df["lep_eta"].to_numpy())

        for i in range(len(ptbinning_trgSF) - 1):
            high_pt, low_pt = ptbinning_trgSF[i], ptbinning_trgSF[i + 1]
            msk_pt = (lep_pt >= low_pt) & (lep_pt < high_pt)

            for j in range(len(etabinning_trgSF) - 1):
                low_eta, high_eta = etabinning_trgSF[j], etabinning_trgSF[j + 1]
                msk_eta = (lep_eta >= low_eta) & (lep_eta < high_eta)

                msk = msk_pt & msk_eta
                nominal[msk] *= sf_nominal[i, j]

    return nominal


def make_events_dict(years, channels, samples_dir, samples, presel, THWW_path=None, fake_SF={"ele": 1, "mu": 1}):
    """
    Postprocess the parquets by applying preselection, saving a `nominal` weight column, and
    saving a THWW tagger score column, all in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [dict]: key=year, value=str pointing to the path of the parquets
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "df"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)

    Returns
        a dict() object events_dict[year][channel][samples] that contains a dataframe of procesed events.

    """

    if "Fake" in samples:  # Fake has a special treatment after the loop
        add_fake = True
        samples.remove("Fake")
    else:
        add_fake = False

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            logging.info(f"Processing {year}, {ch} channel")
            events_dict[year][ch] = {}

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):
                sample_to_use = utils.get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                is_data = True if sample_to_use == "Data" else False

                parquet_files = glob.glob(f"{samples_dir[year]}/{sample}/outfiles/*_{ch}.parquet")
                pkl_files = glob.glob(f"{samples_dir[year]}/{sample}/outfiles/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    df = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(df) == 0:
                    continue

                if "ggF" in sample_to_use:
                    if "GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8" in sample:
                        df = df[df["fj_genH_pt"] < 200]
                    else:
                        df = df[df["fj_genH_pt"] >= 200]

                df["xsecweight"] = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                df["nominal"] = get_nominal(df, year, ch, df["xsecweight"], is_data)

                if THWW_path is not None:
                    df["THWW"] = utils.get_finetuned_score(df, THWW_path)
                    df = df[df.columns.drop(list(df.filter(regex="hidNeuron")))]

                for selection in presel[ch]:
                    df = df.query(presel[ch][selection])

                if sample_to_use not in events_dict[year][ch]:
                    events_dict[year][ch][sample_to_use] = df
                else:
                    events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], df])

            if add_fake:
                df = pd.read_parquet(f"{samples_dir[year]}/Fake/fake_{year}_{ch}_FR_Nominal.parquet")
                for selection in presel[ch]:
                    df = df.query(presel[ch][selection])

                df["nominal"] *= fake_SF[ch]  # apply Fake SF

                events_dict[year][ch]["Fake"] = df

    return events_dict


def fix_neg_yields(h):
    """Will set the bin yields of a process to 0 if the nominal yield is negative."""

    for sample in h.axes["samples"]:
        neg_bins = np.where(h[{"samples": sample}].values() < 0)[0]

        if len(neg_bins) > 0:
            print(f"{sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

            sample_index = np.argmax(np.array(h.axes["samples"]) == sample)

            for neg_bin in neg_bins:
                h.view(flow=True)[sample_index, neg_bin + 1] = (0, 0)


def plot_hists_from_events_dict(events_dict, plot_config):
    """Takes an `events_dict` object that was processed by `make_events_dict` and starts filling histograms."""

    for region, sel in plot_config["regions_to_plot"].items():

        logging.info(f"Making stacked histograms for region {region}")

        if not os.path.exists(plot_config["outdir"] + f"/{region}/"):
            os.makedirs(plot_config["outdir"] + f"/{region}/")

        # instantiate histograms which contain the different up/down variations to extract the total syst. unc.
        if plot_config["plot_syst_unc"]:
            import sys

            sys.path.append("../combine/")
            from systematics import get_systematic_dict

            SYST_DICT = get_systematic_dict(plot_config["years_to_plot"])

            from syst_unc_utils import (
                fill_syst_unc_hists,
                get_total_syst_unc,
                initialize_syst_unc_hists,
            )

            SYST_UNC_up, SYST_UNC_down = {}, {}
            SYST_hists = initialize_syst_unc_hists(SYST_DICT, plot_config)

        # instantiate nominal histograms
        hists = {}
        for var_to_plot in plot_config["vars_to_plot"]:
            hists[var_to_plot] = hist2.Hist(
                hist2.axis.StrCategory([], name="samples", growth=True),
                utils.get_axis(var_to_plot, plot_config["massbin"]),
                storage=hist2.storage.Weight(),
            )

        # start filling the histograms
        for var_to_plot in plot_config["vars_to_plot"]:
            for year in plot_config["years_to_plot"]:
                for ch in plot_config["channels_to_plot"]:
                    for sample in plot_config["samples_to_plot"]:

                        if (ch == "mu") and (sample == "Fake"):
                            continue

                        # -------------- some samples may be split during plotting to matched/unmatched
                        if "TTbar" in sample:
                            df = events_dict[year][ch]["TTbar"]

                            if "TTbar_allmatched" in sample:
                                df = df[df["fj_isTop_W_lep_b"] == 1]
                            elif "TTbar_unmatched" in sample:
                                df = df[df["fj_isTop_W_lep_b"] != 1]

                        elif "WJetsLNu" in sample:
                            df = events_dict[year][ch]["WJetsLNu"]

                            if "unmatched" in sample:
                                df = df[df["fj_V_isMatched"] != 1]
                            elif "matched" in sample:
                                df = df[df["fj_V_isMatched"] == 1]

                        else:
                            df = events_dict[year][ch][sample]

                        df = df.query(sel)

                        # ----------- some variables need manual tweaking
                        if "lep_isolation_ele" in var_to_plot:
                            if ch != "ele":
                                continue
                            df = df[(df["lep_pt"] > 120)] if "highpt" in var_to_plot else df[(df["lep_pt"] < 120)]
                            df[var_to_plot] = df["lep_isolation"]

                        elif "lep_isolation_mu" in var_to_plot:
                            if ch != "mu":
                                continue
                            df = df[(df["lep_pt"] > 55)] if "highpt" in var_to_plot else df[(df["lep_pt"] < 55)]
                            df[var_to_plot] = df["lep_isolation"]

                        elif "lep_misolation_ele" in var_to_plot:
                            if ch != "ele":
                                continue
                            df = df[(df["lep_pt"] > 120)] if "highpt" in var_to_plot else df[(df["lep_pt"] < 120)]
                            df[var_to_plot] = df["lep_misolation"]

                        elif "lep_misolation_mu" in var_to_plot:
                            if ch != "mu":
                                continue
                            df = df[(df["lep_pt"] > 55)] if "highpt" in var_to_plot else df[(df["lep_pt"] < 55)]
                            df[var_to_plot] = df["lep_misolation"]

                        hists[var_to_plot].fill(
                            samples=sample,
                            var=df[var_to_plot],
                            weight=df["nominal"],
                        )

                        if plot_config["plot_syst_unc"]:
                            SYST_hists = fill_syst_unc_hists(SYST_DICT, SYST_hists, year, ch, sample, var_to_plot, df)

            fix_neg_yields(hists[var_to_plot])
            if plot_config["plot_syst_unc"]:
                SYST_UNC_up[var_to_plot], SYST_UNC_down[var_to_plot] = get_total_syst_unc(SYST_hists[var_to_plot])

        utils.plot_hists(
            hists,
            plot_config["years_to_plot"],
            plot_config["channels_to_plot"],
            plot_config["vars_to_plot"],
            add_data=plot_config["add_data"],
            add_soverb=plot_config["add_soverb"],
            blind_region=plot_config["blind_region"],
            logy=plot_config["logy"],
            mult=plot_config["mult"],
            legend_ncol=plot_config["legend_ncol"],
            outpath=plot_config["outdir"] + f"/{region}/",
            plot_Fake_unc=plot_config["plot_Fake_unc"] if plot_config["plot_Fake_unc"] != 0 else None,
            plot_syst_unc=(SYST_UNC_up, SYST_UNC_down) if plot_config["plot_syst_unc"] else None,
        )


def main(args):

    if args.make_events_dict:
        with open("config_make_events_dict.yaml", "r") as stream:
            events_dict_config = yaml.safe_load(stream)

        # create output directory if it doesn't exist
        if not os.path.exists(events_dict_config["outdir"]):
            os.makedirs(events_dict_config["outdir"])

        os.system(f"cp config_make_events_dict.yaml {events_dict_config['outdir']}/")

        events_dict = make_events_dict(
            events_dict_config["years"],
            events_dict_config["channels"],
            events_dict_config["samples_dir"],
            events_dict_config["samples"],
            events_dict_config["presel"],
            events_dict_config["THWW_path"],
        )
        with open(f"{events_dict_config['outdir']}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)

        logging.info(f"Done with building the events_dict and stored it at {events_dict_config['outdir']}/events_dict.pkl")

    if args.plot_stacked_hists:
        with open("config_plot_stacked_hists.yaml", "r") as stream:
            plot_config = yaml.safe_load(stream)

        try:
            with open(f"{plot_config['events_dict_path']}", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.error(f"Event dictionary not found in {plot_config['events_dict_path']}. Re-run with --make-events-dict")
            exit()

        if not os.path.exists(plot_config["outdir"]):
            os.makedirs(plot_config["outdir"])

        os.system(f"cp config_plot_stacked_hists.yaml {plot_config['outdir']}/")

        plot_hists_from_events_dict(
            events_dict,
            plot_config,
        )


if __name__ == "__main__":
    # e.g. python make_stacked_hists.py --plot-stacked-hists --make-events-dict

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--make-events-dict",
        dest="make_events_dict",
        help="Make events dictionary according to the config specified in config_make_events_dict.yaml",
        action="store_true",
    )
    parser.add_argument(
        "--plot-stacked-hists",
        dest="plot_stacked_hists",
        help="Plot stacked histograms according to the config specified in config_plot_stacked_hists.yaml",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
