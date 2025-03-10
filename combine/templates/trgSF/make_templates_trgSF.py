"""
Builds hist.Hist templates after adding systematics for all samples.

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
import pandas as pd
import pyarrow
import yaml
from utils_diffBins import get_common_sample_name, get_finetuned_score, get_xsecweight

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "TTbar", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (THWW>0.90)}`)
        model_path [str]: path to the ParT finetuned model.onnx
        add_fake [Bool]: if True will include Fake as an additional sample in the output hists

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (Sample, Systematic, Region, mass_observable)

    """

    with open("../binder/trg_eff_SF.pkl", "rb") as f:
        TRIGGER_SF = pkl.load(f)

    # add extra selections to preselection
    presel = {
        "mu": {
            "fj_mass": "fj_mass>40",
            "SR": "(n_bjets_T==0) & (THWW>0.905)",
        },
        "ele": {
            "fj_mass": "fj_mass>40",
            "SR": "(n_bjets_T==0) & (THWW>0.905)",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.Variable(
            list(range(55, 255, mass_binning)),
            name="mass_observable",
            label=r"Higgs reconstructed mass [GeV]",
            overflow=True,
        ),
        storage=hist2.storage.Weight(),
    )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                is_data = True if sample_to_use == "Data" else False

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["THWW"] = get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = get_xsecweight(
                    pkl_files, year, sample, sample_to_use, is_data, luminosity
                )

                df = data.copy()

                COMMON_systs_correlated = {
                    "sfelec_id": "weight_ele_id_electron",
                    "sfelec_reco": "weight_ele_reco_electron",
                }

                # ------------------- Nominal -------------------
                df["nominal"] = df[f"weight_{ch}"] * xsecweight * df["weight_btag"]

                # integrate the noninal trigger SF into the nominal weight and the variations
                ptbinning = [2000, 200, 120, 30]
                etabinning = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

                df["weight_trg_up"], df["weight_trg_down"] = df["nominal"].copy(), df["nominal"].copy()

                for i in range(len(ptbinning) - 1):
                    high_pt = ptbinning[i]
                    low_pt = ptbinning[i + 1]

                    msk_pt = (df["lep_pt"] >= low_pt) & (df["lep_pt"] < high_pt)

                    for j in range(len(etabinning) - 1):
                        low_eta = etabinning[j]
                        high_eta = etabinning[j + 1]

                        msk_eta = (abs(df["lep_eta"]) >= low_eta) & (abs(df["lep_eta"]) < high_eta)

                        df["nominal"][msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][i, j]

                        for syst, var in COMMON_systs_correlated.items():
                            df[var + "Up"][msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][
                                i, j
                            ]
                            df[var + "Down"][msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][
                                i, j
                            ]

                        df["weight_trg_up"][msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["up"][i, j]
                        df["weight_trg_down"][msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["down"][
                            i, j
                        ]

                hists.fill(
                    Systematic="nominal",
                    mass_observable=df["rec_higgs_m"],
                    weight=df["nominal"],
                )

                hists.fill(
                    Systematic="sfelec_trigger_up",
                    mass_observable=df["rec_higgs_m"],
                    weight=df["weight_trg_up"],
                )
                hists.fill(
                    Systematic="sfelec_trigger_down",
                    mass_observable=df["rec_higgs_m"],
                    weight=df["weight_trg_down"],
                )

                # ------------------- Common systematics  -------------------

                for syst, var in COMMON_systs_correlated.items():

                    shape_up = df[var + "Up"] * xsecweight * df["weight_btag"]
                    shape_down = df[var + "Down"] * xsecweight * df["weight_btag"]

                    hists.fill(
                        Systematic=f"{syst}_up",
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_up,
                    )
                    hists.fill(
                        Systematic=f"{syst}_down",
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_down,
                    )

    logging.info(hists)

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["model_path"],
    )

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/trgSF

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
