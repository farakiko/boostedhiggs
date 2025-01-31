from __future__ import division, print_function

import json
import logging
import warnings

# from lib2to3.fixes.fix_operator import invocation
from typing import List

import pandas as pd
import rhalphalib as rl
from systematics import CONTROL_regions, SIG_regions, bkgs, samples, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def systs_not_from_parquets(years: List[str], lep_channels: List[str]):
    """
    Define systematics that are NOT stored in the parquets.

    Args
        years: e.g. ["2018", "2017", "2016APV", "2016"]
        lep_channels: e.g. ["mu", "ele"]

    Returns
        systs_dict [dict]:        keys are systematics; values are rl.NuisanceParameters
        systs_dict_values [dict]: keys are same as above; values are tuples (up, down) and if (up, None) then down=up
    """
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in lep_channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(lep_channels)

    # get the LUMI covered in the templates
    full_lumi = 0
    for year in years:
        full_lumi += LUMI[year]

    systs_dict, systs_dict_values = {}, {}

    # COMMON SYSTEMATICS
    systs_dict["all_samples"], systs_dict_values["all_samples"] = {}, {}

    # lumi systematics
    if "2016" in years:
        systs_dict["all_samples"]["lumi_13TeV_2016"] = rl.NuisanceParameter("lumi_13TeV_2016", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2016"] = (1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi), None)
    if "2017" in years:
        systs_dict["all_samples"]["lumi_13TeV_2017"] = rl.NuisanceParameter("lumi_13TeV_2017", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2017"] = (1.02 ** (LUMI["2017"] / full_lumi), None)

    if "2018" in years:
        systs_dict["all_samples"]["lumi_13TeV_2018"] = rl.NuisanceParameter("lumi_13TeV_2018", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2018"] = (1.015 ** (LUMI["2018"] / full_lumi), None)

    if len(years) == 4:
        systs_dict["all_samples"]["lumi_13TeV_correlated"] = rl.NuisanceParameter("lumi_13TeV_correlated", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_correlated"] = (
            (
                (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
                * (1.009 ** (LUMI["2017"] / full_lumi))
                * (1.02 ** (LUMI["2018"] / full_lumi))
            ),
            None,
        )

        systs_dict["all_samples"]["lumi_13TeV_1718"] = rl.NuisanceParameter("lumi_13TeV_1718", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_1718"] = (
            (1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi)),
            None,
        )

    # mini-isolation
    systs_dict["all_samples"]["miniisolation_SF_unc"] = rl.NuisanceParameter(
        f"{CMS_PARAMS_LABEL}_miniisolation_SF_unc", "lnN"
    )
    systs_dict_values["all_samples"]["miniisolation_SF_unc"] = (1.02, 0.98)

    systs_dict["all_samples"]["trigger_SF_unc"] = rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_trigger_syst_unc", "lnN")
    systs_dict_values["all_samples"]["trigger_SF_unc"] = (1.053, None)

    # PER SAMPLE SYSTEMATICS
    for sample in samples:
        systs_dict[sample], systs_dict_values[sample] = {}, {}

    n_br = rl.NuisanceParameter("BR_hww", "lnN")
    for sample in sigs:
        # branching ratio systematics
        systs_dict[sample]["BR_hww"] = n_br
        systs_dict_values[sample]["BR_hww"] = (1.0153, 0.9848)

    ############################################
    ############################################
    ############################################
    # Theory Systematics. (convention: https://gitlab.cern.ch/hh/naming-conventions#theory-uncertainties)

    # PDF
    systs_dict["ggF"]["pdf_Higgs_ggH"] = rl.NuisanceParameter("pdf_Higgs_ggH", "lnN")
    systs_dict_values["ggF"]["pdf_Higgs_ggH"] = (1.019, None)

    systs_dict["VBF"]["pdf_Higgs_qqH"] = rl.NuisanceParameter("pdf_Higgs_qqH", "lnN")
    systs_dict_values["VBF"]["pdf_Higgs_qqH"] = (1.021, None)

    n = rl.NuisanceParameter("pdf_Higgs_ttH", "lnN")
    systs_dict["ttH"]["pdf_Higgs_ttH"] = n
    systs_dict_values["ttH"]["pdf_Higgs_ttH"] = (1.03, None)

    n = rl.NuisanceParameter("pdf_Higgs_WH", "lnN")
    systs_dict["WH"]["pdf_Higgs_WH"] = n
    systs_dict_values["WH"]["pdf_Higgs_WH"] = (1.017, None)

    n = rl.NuisanceParameter("pdf_Higgs_ZH", "lnN")
    systs_dict["ZH"]["pdf_Higgs_ZH"] = n
    systs_dict_values["ZH"]["pdf_Higgs_ZH"] = (1.013, None)

    # QCD scale
    n = rl.NuisanceParameter("QCDscale_ggH", "lnN")
    systs_dict["ggF"]["QCDscale_ggH"] = n
    systs_dict_values["ggF"]["QCDscale_ggH"] = (1.039, None)

    n = rl.NuisanceParameter("QCDscale_qqH", "lnN")
    systs_dict["VBF"]["QCDscale_qqH"] = n
    systs_dict_values["VBF"]["QCDscale_qqH"] = (1.004, 0.997)

    n = rl.NuisanceParameter("QCDscale_ttH", "lnN")
    systs_dict["ttH"]["QCDscale_ttH"] = n
    systs_dict_values["ttH"]["QCDscale_ttH"] = (1.058, 0.908)

    n = rl.NuisanceParameter("QCDscale_WH", "lnN")
    systs_dict["WH"]["QCDscale_WH"] = n
    systs_dict_values["WH"]["QCDscale_WH"] = (1.005, 0.993)

    n = rl.NuisanceParameter("QCDscale_ZH", "lnN")
    systs_dict["ZH"]["QCDscale_ZH"] = n
    systs_dict_values["ZH"]["QCDscale_ZH"] = (1.038, 0.97)

    # alphas
    n = rl.NuisanceParameter("alpha_s", "lnN")
    systs_dict["ggF"]["alpha_s"] = n
    systs_dict_values["ggF"]["alpha_s"] = (1.026, None)
    systs_dict["VBF"]["alpha_s"] = n
    systs_dict_values["VBF"]["alpha_s"] = (1.005, None)
    systs_dict["ttH"]["alpha_s"] = n
    systs_dict_values["ttH"]["alpha_s"] = (1.020, None)
    systs_dict["WH"]["alpha_s"] = n
    systs_dict_values["WH"]["alpha_s"] = (1.009, None)
    systs_dict["ZH"]["alpha_s"] = n
    systs_dict_values["ZH"]["alpha_s"] = (1.009, None)

    return systs_dict, systs_dict_values


def systs_from_parquets(years):
    """
    Specify systematics that ARE stored in the parquets.

    The convention is:
        Dict[tuple[]]

        key: rl.NuisanceParameter with the name and type of nuissance in the datacard
        value: tuple (_, _, _)
            tuple[0]: name of nuissance in the templates
            tuple[1]: list of samples for which the nuissance is applied
            tuple[1]: list of regions for ewhich the nuissance is present

    """

    # ------------------- Common systematics -------------------

    SYSTEMATICS_correlated = {
        rl.NuisanceParameter("CMS_pileup_id", "shape"): (
            "weight_pileup_id",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_EW_unc", "lnN"): (
            "EW",
            ["VBF", "WH", "ZH", "ttH"],
            SIG_regions + CONTROL_regions,
        ),
        # ISR systematics
        rl.NuisanceParameter("ps_isr_ggH", "shape"): (
            "weight_PSISR",
            ["ggF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_qqH", "shape"): (
            "weight_PSISR",
            ["VBF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_WH", "shape"): (
            "weight_PSISR",
            ["WH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_ZH", "shape"): (
            "weight_PSISR",
            ["ZH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_ttH", "shape"): (
            "weight_PSISR",
            ["ttH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_wjets", "shape"): (
            "weight_PSISR",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_ttbar", "shape"): (
            "weight_PSISR",
            ["TTbar"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_isr_singletop", "shape"): (
            "weight_PSISR",
            ["SingleTop"],
            SIG_regions + CONTROL_regions,
        ),
        # FSR systematics
        rl.NuisanceParameter("ps_fsr_ggH", "shape"): (
            "weight_PSFSR",
            ["ggF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_fsr_qqH", "shape"): (
            "weight_PSFSR",
            ["VBF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_fsr_WH", "shape"): (
            "weight_PSFSR",
            ["WH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_fsr_ZH", "shape"): (
            "weight_PSFSR",
            ["ZH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_fsr_ttH", "shape"): (
            "weight_PSFSR",
            ["ttH"],
            SIG_regions + CONTROL_regions,
        ),
        # specefic handling of ps_fsr_wjets per region
        rl.NuisanceParameter("ps_fsr_wjets_ggFpt250to350", "lnN"): (
            "weight_PSFSR",
            ["WJetsLNu"],
            ["ggFpt250to350"],
        ),
        rl.NuisanceParameter("ps_fsr_wjets_ggFpt350to500", "lnN"): (
            "weight_PSFSR",
            ["WJetsLNu"],
            ["ggFpt350to500"],
        ),
        rl.NuisanceParameter("ps_fsr_wjets_ggFpt500toInf", "lnN"): (
            "weight_PSFSR",
            ["WJetsLNu"],
            ["ggFpt500toInf"],
        ),
        ######################
        rl.NuisanceParameter("ps_fsr_ttbar", "shape"): (
            "weight_PSFSR",
            ["TTbar"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("ps_fsr_singletop", "shape"): (
            "weight_PSFSR",
            ["SingleTop"],
            SIG_regions + CONTROL_regions,
        ),
        # systematics applied only on WJets & DYJets
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO", "lnN"): (
            "weight_d1K_NLO",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO", "lnN"): (
            "weight_d2K_NLO",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO", "lnN"): (
            "weight_d3K_NLO",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW", "lnN"): (
            "weight_d1kappa_EW",
            ["WJetsLNu", "DYJets"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW", "lnN"): (
            "weight_W_d2kappa_EW",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW", "lnN"): (
            "weight_W_d3kappa_EW",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW", "lnN"): (
            "weight_Z_d2kappa_EW",
            ["DYJets"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW", "lnN"): (
            "weight_Z_d3kappa_EW",
            ["DYJets"],
            SIG_regions + CONTROL_regions,
        ),
        # systematics for muon channel
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_isolation", "lnN"): (
            "weight_mu_isolation",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_iso", "lnN"): (
            "weight_mu_trigger_iso",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger", "lnN"): (
            "weight_mu_trigger_noniso",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_identification_syst", "lnN"): (
            "weight_mu_id_syst",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        # PDF acceptance
        rl.NuisanceParameter(f"PDF_ggH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["ggF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_qqH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["VBF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_ttH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["ttH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_WH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["WH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_ZH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["ZH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_wjets_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"PDF_ttbar_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_pdf_acceptance",
            ["TTbar"],
            SIG_regions + CONTROL_regions,
        ),
        # rl.NuisanceParameter(f"PDF_singletop_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
        #     "weight_pdf_acceptance",
        #     ["SingleTop"],
        # ),
        # QCD scale acceptance
        rl.NuisanceParameter(f"QCDscale_ggH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["ggF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"QCDscale_qqH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["VBF"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"QCDscale_ttH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["ttH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"QCDscale_WH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["WH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"QCDscale_ZH_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["ZH"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"QCDscale_wjets_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["WJetsLNu"],
            SIG_regions + CONTROL_regions,
        ),
        # rl.NuisanceParameter(f"QCDscale_ttbar_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
        #     "weight_qcd_scale",
        #     ["TTbar"],
        # ),
        rl.NuisanceParameter(f"QCDscale_singletop_ACCEPT_{CMS_PARAMS_LABEL}", "shape"): (
            "weight_qcd_scale",
            ["SingleTop"],
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_TopPtReweight", "shape"): (
            "weight_TopPtReweight",
            ["TTbar"],
            SIG_regions + CONTROL_regions,
        ),
    }

    SYSTEMATICS_uncorrelated = {}
    for year in years:
        SYSTEMATICS_uncorrelated = {
            **SYSTEMATICS_uncorrelated,
            **{
                rl.NuisanceParameter(f"CMS_pileup_{year}", "shape"): (
                    f"weight_pileup_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_identification_stat_{year}", "lnN"): (
                    "weight_mu_id_stat",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                # systematics for electron channel
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_identification_{year}", "lnN"): (
                    "weight_ele_id",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_reconstruction_{year}", "lnN"): (
                    "weight_ele_reco",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                # trigger SF
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_trigger_stat_unc_{year}", "shape"): (
                    "trigger_ele_SF",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"ps_fsr_wjets_{year}", "shape"): (
                    f"weight_PSFSR_{year}",
                    ["WJetsLNu"],
                    SIG_regions + CONTROL_regions,
                ),
            },
        }
        if year != "2018":
            SYSTEMATICS_uncorrelated = {
                **SYSTEMATICS_uncorrelated,
                **{
                    rl.NuisanceParameter(f"CMS_l1_ecal_prefiring_{year}", "shape"): (
                        f"weight_L1Prefiring_{year}",
                        sigs + bkgs,
                        SIG_regions + CONTROL_regions,
                    ),
                },
            }

    # ------------------- btag systematics -------------------

    # systematics correlated across all years
    BTAG_systs_correlated = {
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFlightCorrelated", "lnN"): (
            "weight_btagSFlightCorrelated",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbcCorrelated", "lnN"): (
            "weight_btagSFbcCorrelated",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
    }

    # systematics uncorrelated across all years
    BTAG_systs_uncorrelated = {}
    for year in years:
        BTAG_systs_uncorrelated = {
            **BTAG_systs_uncorrelated,
            **{
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFlight_{year}", "lnN"): (
                    f"weight_btagSFlight_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbc_{year}", "lnN"): (
                    f"weight_btagSFbc_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
            },
        }

    # ------------------- JECs -------------------

    # systematics correlated across all years
    JEC_systs_correlated = {
        rl.NuisanceParameter("unclustered_Energy", "lnN"): (
            "UES",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        # individual sources
        rl.NuisanceParameter("CMS_scale_j_FlavQCD", "lnN"): (
            "JES_FlavorQCD",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("CMS_scale_j_RelBal", "lnN"): (
            "JES_RelativeBal",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("CMS_scale_j_HF", "lnN"): (
            "JES_HF",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("CMS_scale_j_BBEC1", "lnN"): (
            "JES_BBEC1",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("CMS_scale_j_EC2", "lnN"): (
            "JES_EC2",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
        rl.NuisanceParameter("CMS_scale_j_Abs", "lnN"): (
            "JES_Absolute",
            sigs + bkgs,
            SIG_regions + CONTROL_regions,
        ),
    }

    # systematics uncorrelated across all years
    JEC_systs_uncorrelated = {}
    for year in years:
        JEC_systs_uncorrelated = {
            **JEC_systs_uncorrelated,
            **{
                rl.NuisanceParameter(f"CMS_res_j_{year}", "lnN"): (
                    f"JER_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_jmr_{year}", "lnN"): (
                    f"JMR_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_jms_{year}", "lnN"): (
                    f"JMS_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"CMS_scale_j_BBEC1_{year}", "lnN"): (
                    f"JES_BBEC1_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"CMS_scale_j_RelSample_{year}", "lnN"): (
                    f"JES_RelativeSample_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"CMS_scale_j_EC2_{year}", "lnN"): (
                    f"JES_EC2_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"CMS_scale_j_HF_{year}", "lnN"): (
                    f"JES_HF_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
                rl.NuisanceParameter(f"CMS_scale_j_Abs_{year}", "lnN"): (
                    f"JES_Absolute_{year}",
                    sigs + bkgs,
                    SIG_regions + CONTROL_regions,
                ),
            },
        }

    SYSTEMATICS = {
        **SYSTEMATICS_correlated,
        **SYSTEMATICS_uncorrelated,
        **BTAG_systs_correlated,
        **BTAG_systs_uncorrelated,
        **JEC_systs_correlated,
        **JEC_systs_uncorrelated,
    }

    return SYSTEMATICS
