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
import numpy as np
import pandas as pd
import pyarrow
from systematics import sigs
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

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (Sample, Systematic, Region, mass_observable)

    """

    # add missing preselection
    presel = {
        "mu": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
        },
        "ele": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
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

                sample_label = get_common_sample_name(sample)

                if sample_label not in samples:
                    continue

                is_data = True if sample_label == "Data" else False

                logging.info(f"Finding {sample} samples and should combine them under {sample_label}")

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
                    pkl_files, year, sample, sample_label, is_data, luminosity
                )

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.

                    df = data.copy()
                    logging.info(f"Applying {region} selection on {len(df)} events")
                    df = df.query(region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(df)} events")

                    # ------------------- add Nominal weight -------------------
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * xsecweight

                        if "bjets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                            nominal *= df["weight_btag"]

                    hists.fill(
                        Sample=sample_label,
                        Systematic="nominal",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

                    # ------------------- add QCD scale acceptance -------------------

                    """
                    For the QCD acceptance uncertainty:
                    - we save the individual weights [0, 1, 3, 5, 7, 8]
                    - postprocessing: we obtain sum_sumlheweight
                    - postprocessing: we obtain LHEScaleSumw: sum_sumlheweight[i] / sum_sumgenweight
                    - postprocessing:
                    obtain histograms for 0, 1, 3, 5, 7, 8 and 4: h0, h1, ... respectively
                    weighted by scale_0, scale_1, etc
                    and normalize them by  (xsec * luminosity) / LHEScaleSumw[i]
                    - then, take max/min of h0, h1, h3, h5, h7, h8 w.r.t h4: h_up and h_dn
                    - the uncertainty is the nominal histogram * h_up / h4
                    """
                    if (sample_label in sigs + ["WJetsLNu", "TTbar", "SingleTop"]) and (
                        sample != "ST_s-channel_4f_hadronicDecays"
                    ):

                        R_4 = sumscaleweights[4] / sumgenweights
                        scaleweight_4 = df["weight_scale4"].values * nominal / R_4

                        scaleweights = []
                        for weight_i in sumscaleweights:
                            if weight_i == 4:
                                continue

                            # get the normalization factor per variation i (ratio of sumscaleweights_i/sumgenweights)
                            R_i = sumscaleweights[weight_i] / sumgenweights
                            scaleweight_i = df[f"weight_scale{weight_i}"].values * nominal / R_i

                            scaleweights.append(scaleweight_i)

                        scaleweights = np.array(scaleweights)

                        scaleweights = np.swapaxes(
                            np.array(scaleweights), 0, 1
                        )  # so that the shape is (# events, variation)

                        # TODO: debug
                        shape_up = nominal * np.max(scaleweights, axis=1) / scaleweight_4
                        shape_down = nominal * np.min(scaleweights, axis=1) / scaleweight_4

                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_label,
                        Systematic="weight_qcd_scale_up",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_label,
                        Systematic="weight_qcd_scale_down",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_down,
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
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 1e-3
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 1e-3


def main(args):

    years = args.years.split(",")
    channels = args.channels.split(",")

    os.system(f"mkdir -p {args.outdir}")

    samples = [
        "ggF",
        "VBF",
        "WH",
        "ZH",
        "ttH",
        "WJetsLNu",
        "TTbar",
        "SingleTop",
    ]

    regions_sel = {
        "VBFcat": "(n_bjets_T == 0) & (THWW > 0.905) & ((mjj > 1000) & (deta > 3.5))",
        "ggFcat": "(n_bjets_T == 0) & (THWW > 0.93) & ((mjj < 1000) | (deta < 3.5))",
        "TopCR": "(n_bjets_T > 0)",
        "WJetsCR": "(n_bjets_T == 0) & (THWW < 0.905)",
    }

    samples_dir = {
        "2018": "../eos/Oct10_hww_2018",
        "2017": "../eos/Oct10_hww_2017",
        "2016": "../eos/Oct10_hww_2016",
        "2016APV": "../eos/Oct10_hww_2016APV",
    }

    model_path = "../../weaver-core-dev/experiments_finetuning/v35_30/model.onnx"

    hists = get_templates(
        years,
        channels,
        samples,
        samples_dir,
        regions_sel,
        model_path,
    )

    fix_neg_yields(hists)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}"

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)

    if args.plot_unc:
        # plotting
        import matplotlib.pyplot as plt
        import mplhep as hep

        plt.style.use(hep.style.CMS)

        # get lumi
        with open("../fileset/luminosity.json") as f:
            luminosity = json.load(f)

        def get_lumi(years, channels):
            lum_ = 0
            for year in years:
                lum = 0
                for ch in channels:
                    lum += luminosity[ch][year] / 1000.0

                lum_ += lum / len(channels)
            return round(lum_)

        get_lumi(years, channels)

        color_dict = {
            "Nominal": "grey",
            "Up": "blue",
            "Down": "red",
        }

        label_dict = {
            "nominal": "Nominal",
            "up": "Up",
            "down": "Down",
        }

        for region in regions_sel:

            for sample in samples:

                fig, ax = plt.subplots(figsize=(7, 7))

                sum_envelope = {}
                for variation in ["nominal", "up", "down"]:

                    syst = variation if variation == "nominal" else f"weight_qcd_scale_{variation}"

                    hep.histplot(
                        hists[{"Sample": sample, "Region": region, "Systematic": syst}],
                        ax=ax,
                        linewidth=1,
                        histtype="step",
                        label=variation,
                        flow="none",
                        color=color_dict[label_dict[variation]],
                    )

                    sum_envelope[variation] = hists[{"Sample": sample, "Region": region, "Systematic": syst}].values().sum()

                ax.legend(title=region + " (QCDScaleacc unc)")

                ax.set_ylabel(f"{sample} events")

                hep.cms.lumitext(str(get_lumi(years, channels)) + r" fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
                hep.cms.text("Work in Progress", ax=ax, fontsize=15)

                plt.tight_layout()
                plt.savefig(f"{args.outdir}/qcd_scale_{region}_{sample}.pdf")


if __name__ == "__main__":
    # e.g.
    # python qcd_scale_templates.py --years 2017 --channels mu,ele --outdir templates/qcd_scale --plot-unc

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")
    parser.add_argument("--plot-unc", action="store_true")

    args = parser.parse_args()

    main(args)
