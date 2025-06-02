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
import yaml
from systematics import get_systematic_dict, sigs
from utils import get_common_sample_name, get_finetuned_score, get_xsecweight

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


with open("../binder/trg_eff_SF_ARC.pkl", "rb") as f:
    TRIGGER_SF = pkl.load(f)

THWW_SF = {
    "ggF": 0.948,
    "VBF": 0.984,
}
ptbinning_trgSF = [2000, 200, 170, 150, 130, 110, 90, 70, 50, 30]
etabinning_trgSF = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


def get_nominal_from_df(df, year, ch, sample_label, region, region_sel, xsecweight, is_data):

    if is_data:
        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
    else:
        nominal = df[f"weight_{ch}"] * xsecweight

        if "bjets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
            nominal *= df["weight_btag"]

        # apply trigger SF
        if ch == "ele":

            for i in range(len(ptbinning_trgSF) - 1):
                high_pt = ptbinning_trgSF[i]
                low_pt = ptbinning_trgSF[i + 1]

                msk_pt = (df["lep_pt"] >= low_pt) & (df["lep_pt"] < high_pt)

                for j in range(len(etabinning_trgSF) - 1):
                    low_eta = etabinning_trgSF[j]
                    high_eta = etabinning_trgSF[j + 1]

                    msk_eta = (abs(df["lep_eta"]) >= low_eta) & (abs(df["lep_eta"]) < high_eta)

                    nominal[msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][i, j]

        # apply THWW SF
        if ("ggF" in sample_label) or (sample_label in ["ggF", "VBF", "WH", "ZH", "ttH"]):
            if "ggF" in region:
                nominal *= THWW_SF["ggF"]
            else:
                nominal *= THWW_SF["VBF"]

    return nominal


def fill_systematics(
    data,
    hists,
    years,
    year,
    ch,
    regions_sel,
    is_data,
    sample,
    sample_label,
    xsecweight,
    sumpdfweights,
    sumgenweights,
    sumscaleweights,
):
    SYST_DICT = get_systematic_dict(years)

    for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.

        df = data.copy()
        df = df.query(region_sel)

        nominal = get_nominal_from_df(df, year, ch, sample_label, region, region_sel, xsecweight, is_data)

        hists.fill(
            Sample=sample_label,
            Systematic="nominal",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=nominal,
        )

        # ------------------- Trigger SF unc. -------------------
        # for the up/down must revert the nominal that i had applied before
        up, down = nominal.copy(), nominal.copy()
        if (ch == "ele") and (not is_data):
            # if ch == "ele":
            for i in range(len(ptbinning_trgSF) - 1):
                high_pt = ptbinning_trgSF[i]
                low_pt = ptbinning_trgSF[i + 1]

                msk_pt = (df["lep_pt"] >= low_pt) & (df["lep_pt"] < high_pt)

                for j in range(len(etabinning_trgSF) - 1):
                    low_eta = etabinning_trgSF[j]
                    high_eta = etabinning_trgSF[j + 1]

                    msk_eta = (abs(df["lep_eta"]) >= low_eta) & (abs(df["lep_eta"]) < high_eta)

                    up[msk_pt & msk_eta] /= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][i, j]
                    down[msk_pt & msk_eta] /= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["nominal"][i, j]

                    up[msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["up"][i, j]
                    down[msk_pt & msk_eta] *= TRIGGER_SF["UL" + year[2:].replace("APV", "")]["down"][i, j]

        hists.fill(
            Sample=sample_label,
            Systematic="trigger_ele_SF_up",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=up,
        )
        hists.fill(
            Sample=sample_label,
            Systematic="trigger_ele_SF_down",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=down,
        )

        # ------------------- signal EWK unc. -------------------
        up, down = nominal.copy(), nominal.copy()
        if sample_label in ["VBF", "WH", "ZH", "ttH"]:
            msk = df["fj_genH_pt"] > 400

            up = np.where(msk, nominal / df["EW_weight"], nominal)
            down = np.where(msk, nominal * df["EW_weight"], nominal)

        hists.fill(
            Sample=sample_label,
            Systematic="EW_up",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=up,
        )
        hists.fill(
            Sample=sample_label,
            Systematic="EW_down",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=down,
        )

        # ------------------- PDF acceptance -------------------

        """
        For the PDF acceptance uncertainty:
        - store 103 variations. 0-100 PDF values
        - The last two values: alpha_s variations.
        - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
        e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
        and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
        """
        if (sample_label in sigs + ["TTbar"]) and (sample != "ST_s-channel_4f_hadronicDecays"):

            pdfweights = []

            for weight_i in sumpdfweights:

                # noqa: get the normalization factor per variation i (ratio of sumpdfweights_i/sumgenweights)
                R_i = sumpdfweights[weight_i] / sumgenweights

                pdfweight = df[f"weight_pdf{weight_i}"].values * nominal / R_i
                pdfweights.append(pdfweight)

            pdfweights = np.swapaxes(np.array(pdfweights), 0, 1)  # so that the shape is (# events, variation)

            abs_unc = np.linalg.norm((pdfweights - nominal.values.reshape(-1, 1)), axis=1)
            # cap at 100% uncertainty
            rel_unc = np.clip(abs_unc / nominal, 0, 1)
            shape_up = nominal * (1 + rel_unc)
            shape_down = nominal * (1 - rel_unc)

        else:
            shape_up = nominal
            shape_down = nominal

        hists.fill(
            Sample=sample_label,
            Systematic="weight_pdf_acceptance_up",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=shape_up,
        )

        hists.fill(
            Sample=sample_label,
            Systematic="weight_pdf_acceptance_down",
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=shape_down,
        )

        # ------------------- QCD scale -------------------

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
        if (sample_label in sigs + ["TTbar", "SingleTop"]) and (sample != "ST_s-channel_4f_hadronicDecays"):

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

            scaleweights = np.swapaxes(np.array(scaleweights), 0, 1)  # so that the shape is (# events, variation)

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

        # ------------------- Common systematics  -------------------

        for syst, (yrs, smpls, var) in SYST_DICT["common"].items():

            if (sample_label in smpls) and (year in yrs) and (ch in var):
                shape_up = df[var[ch] + "Up"] * xsecweight
                shape_down = df[var[ch] + "Down"] * xsecweight

                if "bjets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                    shape_up *= df["weight_btag"]
                    shape_down *= df["weight_btag"]

            else:
                shape_up = nominal
                shape_down = nominal

            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_up",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=shape_up,
            )

            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_down",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=shape_down,
            )

        # ------------------- btag systematics  -------------------

        for syst, (yrs, smpls, var) in SYST_DICT["btag"].items():

            if (sample_label in smpls) and (year in yrs) and (ch in var):
                shape_up = df[var[ch] + "Up"] * nominal
                shape_down = df[var[ch] + "Down"] * nominal
            else:
                shape_up = nominal
                shape_down = nominal

            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_up",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=shape_up,
            )

            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_down",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=shape_down,
            )

    # ------------------- individual sources of JES -------------------

    """We apply the jet pt cut on the up/down variations."""
    for syst, (yrs, smpls, var) in SYST_DICT["JEC"].items():

        for variation in ["up", "down"]:

            for (
                region,
                region_sel,
            ) in regions_sel.items():  # e.g. pass, fail, top control region, etc.

                if (sample_label in smpls) and (year in yrs) and (ch in var):
                    region_sel = region_sel.replace("rec_higgs_pt", "rec_higgs_pt" + var[ch] + f"_{variation}")

                df = data.copy()
                df = df.query(region_sel)

                nominal = get_nominal_from_df(df, year, ch, sample_label, region, region_sel, xsecweight, is_data)

                if (sample_label in smpls) and (year in yrs) and (ch in var):
                    shape_variation = df["rec_higgs_m" + var[ch] + f"_{variation}"]
                else:
                    shape_variation = df["rec_higgs_m"]

                hists.fill(
                    Sample=sample_label,
                    Systematic=f"{syst}_{variation}",
                    Region=region,
                    mass_observable=shape_variation,
                    weight=nominal,
                )


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

    # add extra selections to preselection
    presel = {
        "mu": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
            "lepmiso": "(lep_pt<55) | ( (lep_pt>=55) & (lep_misolation<0.8))",  # needed for the fakes
        },
        "ele": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
        },
    }

    mass_binning = 20
    low_mass_bin, high_mass_bin = 75, 255

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            list(range(low_mass_bin, high_mass_bin, mass_binning)),
            name="mass_observable",
            label=r"Higgs reconstructed mass [GeV]",
            overflow=True,
            underflow=True,
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

                if ("ggF" in sample_to_use) or ("VBF" in sample_to_use):
                    if "Rivet" not in sample:
                        continue

                if sample_to_use not in samples:
                    continue

                is_data = True if sample_to_use == "Data" else False

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

                if sample_to_use == "ggF":
                    if "GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8" in sample:
                        data = data[data["fj_genH_pt"] < 200]
                    else:
                        data = data[data["fj_genH_pt"] >= 200]

                # use hidNeurons to get the finetuned scores
                data["THWW"] = get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    data = data.query(presel[ch][selection])

                # apply genlep recolep matching
                if not is_data:
                    data = data[data["dR_genlep_recolep"] < 0.005]

                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = get_xsecweight(
                    pkl_files, year, sample, sample_to_use, is_data, luminosity
                )

                if sample_to_use == "ggF":

                    stxs_list = [
                        "ggFpt200to300",
                        "ggFpt300to450",
                        "ggFpt450toInf",
                    ]

                    for stxs_bin in stxs_list:
                        df1 = data.copy()
                        if stxs_bin == "ggFpt200to300":
                            msk_gen = (df1["STXS_finecat"] % 100 == 1) | (df1["STXS_finecat"] % 100 == 5)
                        elif stxs_bin == "ggFpt300to450":
                            msk_gen = (df1["STXS_finecat"] % 100 == 2) | (df1["STXS_finecat"] % 100 == 6)
                        elif stxs_bin == "ggFpt450toInf":
                            msk_gen = (
                                (df1["STXS_finecat"] % 100 == 3)
                                | (df1["STXS_finecat"] % 100 == 7)
                                | (df1["STXS_finecat"] % 100 == 4)
                                | (df1["STXS_finecat"] % 100 == 8)
                            )

                        df1 = df1[msk_gen]

                        fill_systematics(
                            df1,
                            hists,
                            years,
                            year,
                            ch,
                            regions_sel,
                            is_data,
                            sample,
                            stxs_bin,  # use genprocess as label
                            xsecweight,
                            sumpdfweights,
                            sumgenweights,
                            sumscaleweights,
                        )
                if sample_to_use == "VBF":
                    stxs_list = [
                        "mjj1000toInf",
                    ]

                    for stxs_bin in stxs_list:
                        df1 = data.copy()
                        if stxs_bin == "mjj1000toInf":
                            msk_gen = (
                                (df1["STXS_finecat"] % 100 == 21)
                                | (df1["STXS_finecat"] % 100 == 22)
                                | (df1["STXS_finecat"] % 100 == 23)
                                | (df1["STXS_finecat"] % 100 == 24)
                            )

                        df1 = df1[msk_gen]

                        fill_systematics(
                            df1,
                            hists,
                            years,
                            year,
                            ch,
                            regions_sel,
                            is_data,
                            sample,
                            stxs_bin,  # use genprocess as label
                            xsecweight,
                            sumpdfweights,
                            sumgenweights,
                            sumscaleweights,
                        )

                fill_systematics(
                    data.copy(),
                    hists,
                    years,
                    year,
                    ch,
                    regions_sel,
                    is_data,
                    sample,
                    sample_to_use,  # use sample_to_use as label
                    xsecweight,
                    sumpdfweights,
                    sumgenweights,
                    sumscaleweights,
                )

    fake_SF = {
        "ele": 0.75,
        "mu": 1.0,
    }
    for variation in ["FR_Nominal", "FR_stat_Up", "FR_stat_Down", "EWK_SF_Up", "EWK_SF_Down"]:

        for year in years:
            for ch in channels:

                data = pd.read_parquet(f"{samples_dir[year]}/Fake/fake_{year}_{ch}_{variation}.parquet")

                # apply selection
                for selection in presel[ch]:
                    data = data.query(presel[ch][selection])

                data["nominal"] *= fake_SF[ch]  # the closure test SF

                for region in hists.axes["Region"]:
                    df = data.copy()

                    df = df.query(regions_sel[region])

                    if variation == "FR_Nominal":
                        hists.fill(
                            Sample="Fake",
                            Systematic="nominal",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=df["nominal"],
                        )
                    else:
                        hists.fill(
                            Sample="Fake",
                            Systematic=variation,
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=df["nominal"],
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

    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            for systematic in h.axes["Systematic"]:
                neg_bins = np.where(h[{"Sample": sample, "Systematic": systematic, "Region": region}].values() < 0)[0]

                if len(neg_bins) > 0:
                    print(
                        f"{region}, {sample}, has {len(neg_bins)} bins for {systematic} with negative yield.. will set them to 0"
                    )  # noqa

                    sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                    region_index = np.argmax(np.array(h.axes["Region"]) == region)

                    for neg_bin in neg_bins:
                        msk = h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value < 0
                        h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value[msk] = 0
                        h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance[msk] = 0


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

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)

    logging.info(f"hist object dumped at {args.outdir}/hists_templates_{save_as}.pkl")


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
