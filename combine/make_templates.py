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


with open("./trg_eff_SF_ARC.pkl", "rb") as f:
    TRIGGER_SF = pkl.load(f)

THWW_SF = {
    "ggF": 0.948,
    "VBF": 0.984,
}

ptbinning_trgSF = [2000, 200, 170, 150, 130, 110, 90, 70, 50, 30]
etabinning_trgSF = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


def get_nominal(df, year, ch, sample_label, region, region_sel, xsecweight, is_data):
    """Compute the nominal event weight."""

    if is_data:
        return np.ones_like(df["fj_pt"])

    nominal = df[f"weight_{ch}"] * xsecweight

    if "bjets" in region_sel:
        nominal *= df["weight_btag"]

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

    if sample_label in ["ggF", "VBF", "WH", "ZH", "ttH"] or "ggF" in sample_label:
        nominal *= THWW_SF["ggF"] if "ggF" in region else THWW_SF["VBF"]

    return nominal


def add_trigger_SF_unc(hists, df, year, ch, sample_label, region, nominal, is_data):
    """Add electron trigger SF up/down variations."""

    up = nominal.copy()
    down = nominal.copy()

    if ch == "ele" and not is_data:
        year_tag = f"UL{year[2:].replace('APV', '')}"
        sf_nominal = TRIGGER_SF[year_tag]["nominal"]
        sf_up = TRIGGER_SF[year_tag]["up"]
        sf_down = TRIGGER_SF[year_tag]["down"]

        lep_pt = df["lep_pt"].to_numpy()
        lep_eta = np.abs(df["lep_eta"].to_numpy())

        for i in range(len(ptbinning_trgSF) - 1):
            high_pt, low_pt = ptbinning_trgSF[i], ptbinning_trgSF[i + 1]
            msk_pt = (lep_pt >= low_pt) & (lep_pt < high_pt)

            for j in range(len(etabinning_trgSF) - 1):
                low_eta, high_eta = etabinning_trgSF[j], etabinning_trgSF[j + 1]
                msk_eta = (lep_eta >= low_eta) & (lep_eta < high_eta)

                msk = msk_pt & msk_eta

                up[msk] /= sf_nominal[i, j]
                down[msk] /= sf_nominal[i, j]

                up[msk] *= sf_up[i, j]
                down[msk] *= sf_down[i, j]

    for syst, weight in [("trigger_ele_SF_up", up), ("trigger_ele_SF_down", down)]:
        hists.fill(
            Sample=sample_label,
            Systematic=syst,
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=weight,
        )

    return hists


def add_hww_EKW_unc(hists, df, sample_label, region, nominal):
    """Add electroweak signal uncertainty for VBF, WH, ZH, ttH."""

    up = nominal.copy()
    down = nominal.copy()

    if sample_label in ["VBF", "WH", "ZH", "ttH"]:
        msk = df["fj_genH_pt"] > 400
        ew_weight = df["EW_weight"]

        up = np.where(msk, nominal / ew_weight, nominal)
        down = np.where(msk, nominal * ew_weight, nominal)

    for syst, weight in [("EW_up", up), ("EW_down", down)]:
        hists.fill(
            Sample=sample_label,
            Systematic=syst,
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=weight,
        )

    return hists


def add_pdfacc_unc(hists, df, sample, sample_label, region, nominal, sumpdfweights, sumgenweights):
    """
    For the PDF acceptance uncertainty:
    - store 103 variations. 0-100 PDF values
    - The last two values: alpha_s variations.
    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
    """

    if sample_label in sigs + ["TTbar"] and sample != "ST_s-channel_4f_hadronicDecays":
        pdfweights = []

        for weight_i in sumpdfweights:
            # get the normalization factor per variation i (ratio of sumpdfweights_i/sumgenweights)
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

    for syst, weight in [
        ("weight_pdf_acceptance_up", shape_up),
        ("weight_pdf_acceptance_down", shape_down),
    ]:
        hists.fill(
            Sample=sample_label,
            Systematic=syst,
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=weight,
        )

    return hists


def add_qcdscaleacc_unc(hists, df, sample, sample_label, region, nominal, sumscaleweights, sumgenweights):
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

    if sample_label in sigs + ["TTbar", "SingleTop"] and sample != "ST_s-channel_4f_hadronicDecays":
        R_4 = sumscaleweights[4] / sumgenweights
        central_scaleweight = df["weight_scale4"].values * nominal / R_4

        scaleweights = []
        for i in sumscaleweights:
            if i == 4:
                continue
            R_i = sumscaleweights[i] / sumgenweights
            w_i = df[f"weight_scale{i}"].values * nominal / R_i
            scaleweights.append(w_i)

        scaleweights = np.swapaxes(np.array(scaleweights), 0, 1)  # (n_events, n_variations)

        shape_up = nominal * np.max(scaleweights, axis=1) / central_scaleweight
        shape_down = nominal * np.min(scaleweights, axis=1) / central_scaleweight
    else:
        shape_up = nominal
        shape_down = nominal

    for syst, weight in [
        ("weight_qcd_scale_up", shape_up),
        ("weight_qcd_scale_down", shape_down),
    ]:
        hists.fill(
            Sample=sample_label,
            Systematic=syst,
            Region=region,
            mass_observable=df["rec_higgs_m"],
            weight=weight,
        )

    return hists


def add_btag_syst(hists, df, year, ch, sample_label, region, nominal, SYST_DICT):
    """b-tag uncertainties."""

    for syst, (yrs, smpls, var) in SYST_DICT["btag"].items():
        if (sample_label in smpls) and (year in yrs) and (ch in var):
            shape_up = df[var[ch] + "Up"] * nominal
            shape_down = df[var[ch] + "Down"] * nominal
        else:
            shape_up = nominal
            shape_down = nominal

        for variation, weight in [("up", shape_up), ("down", shape_down)]:
            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_{variation}",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=weight,
            )

    return hists


def add_JEC_unc(hists, data, year, ch, sample_label, SYST_DICT, region, regions_sel, nominal, xsecweight, is_data):
    """Individual sources of JES/JMS/JMR."""

    for syst, (yrs, smpls, var) in SYST_DICT["JEC"].items():
        for variation in ["up", "down"]:
            for region, region_sel in regions_sel.items():
                df = data.copy()

                if (sample_label in smpls) and (year in yrs) and (ch in var):
                    region_sel_mod = region_sel.replace("rec_higgs_pt", f"rec_higgs_pt{var[ch]}_{variation}")
                    df = df.query(region_sel_mod)
                    nominal_varied = get_nominal(df, year, ch, sample_label, region, region_sel_mod, xsecweight, is_data)
                    mass_obs = df[f"rec_higgs_m{var[ch]}_{variation}"]
                else:
                    df = df.query(region_sel)
                    nominal_varied = get_nominal(df, year, ch, sample_label, region, region_sel, xsecweight, is_data)
                    mass_obs = df["rec_higgs_m"]

                hists.fill(
                    Sample=sample_label,
                    Systematic=f"{syst}_{variation}",
                    Region=region,
                    mass_observable=mass_obs,
                    weight=nominal_varied,
                )

    return hists


def add_fake_unc(hists, years, channels, samples_dir, presel, regions_sel):
    """Fills Fake samples for nominal and fake-related systematics."""

    fake_SF = {
        "ele": 0.75,
        "mu": 1.0,
    }

    variations = ["FR_Nominal", "FR_stat_Up", "FR_stat_Down", "EWK_SF_Up", "EWK_SF_Down"]

    for year in years:
        for ch in channels:
            sf = fake_SF[ch]

            for variation in variations:
                filepath = f"{samples_dir[year]}/Fake/fake_{year}_{ch}_{variation}.parquet"
                if not os.path.exists(filepath):
                    logging.warning(f"Missing file: {filepath}")
                    continue

                df = pd.read_parquet(filepath)

                for cut in presel[ch].values():
                    df = df.query(cut)
                df["nominal"] *= sf  # Apply closure SF

                for region, region_sel in regions_sel.items():
                    df_region = df.query(region_sel)

                    syst_label = "nominal" if variation == "FR_Nominal" else variation
                    hists.fill(
                        Sample="Fake",
                        Systematic=syst_label,
                        Region=region,
                        mass_observable=df_region["rec_higgs_m"],
                        weight=df_region["nominal"],
                    )
    return hists


def add_common_syst(hists, df, year, ch, sample_label, SYST_DICT, region, region_sel, nominal, xsecweight):
    """Any other systematic uncertainty that is not listed above."""

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

        for variation, weight in [("up", shape_up), ("down", shape_down)]:
            hists.fill(
                Sample=sample_label,
                Systematic=f"{syst}_{variation}",
                Region=region,
                mass_observable=df["rec_higgs_m"],
                weight=weight,
            )

    return hists


def fill_systematics(
    df,
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
        df_region = df.query(region_sel)
        if df_region.empty:
            continue

        nominal = get_nominal(df_region, year, ch, sample_label, region, region_sel, xsecweight, is_data)

        # Nominal
        hists.fill(
            Sample=sample_label,
            Systematic="nominal",
            Region=region,
            mass_observable=df_region["rec_higgs_m"],
            weight=nominal,
        )

        # Systematics
        hists = add_trigger_SF_unc(hists, df_region, year, ch, sample_label, region, nominal, is_data)
        hists = add_hww_EKW_unc(hists, df_region, sample_label, region, nominal)
        hists = add_pdfacc_unc(hists, df_region, sample, sample_label, region, nominal, sumpdfweights, sumgenweights)
        hists = add_qcdscaleacc_unc(hists, df_region, sample, sample_label, region, nominal, sumscaleweights, sumgenweights)
        hists = add_btag_syst(hists, df_region, year, ch, sample_label, region, nominal, SYST_DICT)
        hists = add_common_syst(hists, df_region, year, ch, sample_label, SYST_DICT, region, region_sel, nominal, xsecweight)
        hists = add_JEC_unc(
            hists, df_region, year, ch, sample_label, SYST_DICT, region, regions_sel, nominal, xsecweight, is_data
        )


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path):
    """Postprocesses the parquets by applying preselections, and fills templates for different regions.
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
            "lepmiso": "(lep_pt<55) | ( (lep_pt>=55) & (lep_misolation<0.8))",  # needed for the fakes
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
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

    for year in years:
        for ch in channels:
            logging.info(f"Processing {year}, {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):
                sample_to_use = get_common_sample_name(sample)

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

    hists = add_fake_unc(hists, years, channels, samples_dir, presel, regions_sel)

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """

    sample_axis = np.array(h.axes["Sample"])
    syst_axis = np.array(h.axes["Systematic"])
    region_axis = np.array(h.axes["Region"])

    # Fix nominal bins with negative yield
    for region in region_axis:
        for sample in sample_axis:
            values = h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values()
            neg_bins = np.where(values < 0)[0]

            if len(neg_bins) == 0:
                continue

            print(f"[fix_neg_yields] {region}, {sample} has {len(neg_bins)} negative nominal bins — setting to 1e-3.")

            i_sample = np.argmax(sample_axis == sample)
            i_region = np.argmax(region_axis == region)

            for i_bin in neg_bins:
                bin_view = h.view(flow=True)[i_sample, :, i_region, i_bin + 1]
                bin_view.value = 1e-3
                bin_view.variance = 1e-3

    # Now zero out negative yields in variations
    for region in region_axis:
        for sample in sample_axis:
            i_sample = np.argmax(sample_axis == sample)
            i_region = np.argmax(region_axis == region)

            for syst in syst_axis:
                values = h[{"Sample": sample, "Systematic": syst, "Region": region}].values()
                neg_bins = np.where(values < 0)[0]

                if len(neg_bins) == 0:
                    continue

                print(f"[fix_neg_yields] {region}, {sample}, {syst} has {len(neg_bins)} negative bins — zeroing them.")

                for i_bin in neg_bins:
                    bin_view = h.view(flow=True)[i_sample, :, i_region, i_bin + 1]
                    msk = bin_view.value < 0
                    bin_view.value[msk] = 0
                    bin_view.variance[msk] = 0


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
    # e.g. python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
