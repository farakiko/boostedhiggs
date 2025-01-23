bkgs = ["TTbar", "WJetsLNu", "SingleTop", "DYJets", "Diboson", "EWKvjets"]
ttbar_list = [
    "TTbar_is_top_lq",
    "TTbar_is_top_lqq",
    "TTbar_others",
    "TTbar_LP",
    "TTbar_LP_is_top_lq",
    "TTbar_LP_is_top_lqq",
    "TTbar_LP_others",
]
# bkgs += ttbar_list
sigs = ["mjj1000toInf", "ggFpt200to300", "ggFpt300to450", "ggFpt450toInf", "ggF", "VBF", "WH", "ZH", "ttH"]

samples = sigs + bkgs + ["Fake"]


def get_systematic_dict(years):
    """
    The following dictionaries have the following convention,
        key [str] --> name of systematic to store in the histogram / template
        value [tuple] --> (t1, t2, t3):
            t1 [list]: years to process the up/down variations for (store nominal value for the other years)
            t2 [list]: samples to apply systematic for (store nominal value for the other samples)
            t3 [dict]:
                key(s): the channels to apply the systematic for (store nominal value for the other channels)
                value(s): the name of the variable in the parquet for that channel
    """

    COMMON_systs_correlated = {
        "weight_pileup_id": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_pileupIDSF", "mu": "weight_mu_pileupIDSF"},
        ),
        # ISR/FSR
        "weight_PSFSR": (
            years,
            sigs + ["TTbar", "WJetsLNu", "SingleTop"] + ttbar_list,
            {"ele": "weight_ele_PSFSR", "mu": "weight_mu_PSFSR"},
        ),
        "weight_PSISR": (
            years,
            sigs + ["TTbar", "WJetsLNu", "SingleTop"] + ttbar_list,
            {"ele": "weight_ele_PSISR", "mu": "weight_mu_PSISR"},
        ),
        # systematics applied only on WJets & DYJets
        "weight_d1K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d1K_NLO", "mu": "weight_mu_d1K_NLO"},
        ),
        "weight_d2K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d2K_NLO", "mu": "weight_mu_d2K_NLO"},
        ),
        "weight_d3K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d3K_NLO", "mu": "weight_mu_d3K_NLO"},
        ),
        "weight_d1kappa_EW": (
            years,
            ["WJetsLNu", "DYJets"],
            {"ele": "weight_ele_d1kappa_EW", "mu": "weight_mu_d1kappa_EW"},
        ),
        "weight_W_d2kappa_EW": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_W_d2kappa_EW", "mu": "weight_mu_W_d2kappa_EW"},
        ),
        "weight_W_d3kappa_EW": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_W_d3kappa_EW", "mu": "weight_mu_W_d3kappa_EW"},
        ),
        "weight_Z_d2kappa_EW": (
            years,
            ["DYJets"],
            {"ele": "weight_ele_Z_d2kappa_EW", "mu": "weight_mu_Z_d2kappa_EW"},
        ),
        "weight_Z_d3kappa_EW": (
            years,
            ["DYJets"],
            {"ele": "weight_ele_Z_d3kappa_EW", "mu": "weight_mu_Z_d3kappa_EW"},
        ),
        # systematics for electron channel
        "weight_ele_id": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_id_electron"},
        ),
        "weight_ele_reco": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_reco_electron"},
        ),
        # systematics for muon channel
        "weight_mu_isolation": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_isolation_muon"},
        ),
        "weight_mu_id_stat": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_id_muon_stat"},
        ),
        "weight_mu_id_syst": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_id_muon_syst"},
        ),
        "weight_mu_trigger_iso": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_trigger_iso_muon"},
        ),
        "weight_mu_trigger_noniso": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_trigger_noniso_muon"},
        ),
        "weight_TopPtReweight": (
            years,
            ["TTbar"] + ttbar_list,
            {"mu": "weight_mu_TopPtReweight", "ele": "weight_ele_TopPtReweight"},
        ),
    }

    COMMON_systs_uncorrelated = {}
    for year in years:
        COMMON_systs_uncorrelated = {
            **COMMON_systs_uncorrelated,
            **{
                f"weight_pileup_{year}": (
                    [year],
                    sigs + bkgs,
                    {"ele": "weight_ele_pileup", "mu": "weight_mu_pileup"},
                ),
            },
        }
        if year != "2018":
            COMMON_systs_uncorrelated = {
                **COMMON_systs_uncorrelated,
                **{
                    f"weight_L1Prefiring_{year}": (
                        [year],
                        sigs + bkgs,
                        {"ele": "weight_ele_L1Prefiring", "mu": "weight_mu_L1Prefiring"},
                    ),
                },
            }

    # btag syst. have a different treatment because they are not stored in the nominal
    BTAG_systs_correlated = {
        "weight_btagSFlightCorrelated": (
            years,
            sigs + bkgs,
            {"ele": "weight_btagSFlightCorrelated", "mu": "weight_btagSFlightCorrelated"},
        ),
        "weight_btagSFbcCorrelated": (
            years,
            sigs + bkgs,
            {"ele": "weight_btagSFbcCorrelated", "mu": "weight_btagSFbcCorrelated"},
        ),
    }

    # LP syst. have a different treatment because they are not stored in the nominal
    LP_systs_correlated = {}
    sys_keys = ["sys", "prongs", "unclust", "distortion"]
    for sys_key in sys_keys:
        LP_systs_correlated[f"LP_weight_{sys_key}"] = (
            years,
            ["TTbar_LP", "TTbar_LP_is_top_lq", "TTbar_LP_is_top_lqq"],
            {"ele": f"LP_weight_{sys_key}", "mu": f"LP_weight_{sys_key}"},
        )

    BTAG_systs_uncorrelated = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        BTAG_systs_uncorrelated = {
            **BTAG_systs_uncorrelated,
            **{
                f"weight_btagSFlight_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"weight_btagSFlight{yearlabel}", "mu": f"weight_btagSFlight{yearlabel}"},
                ),
                f"weight_btagSFbc_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"weight_btagSFbc{yearlabel}", "mu": f"weight_btagSFbc{yearlabel}"},
                ),
            },
        }

    # JEC / JMS
    JEC_systs_correlated = {
        "JES_FlavorQCD": (
            years,
            sigs + bkgs,
            {"ele": "JES_FlavorQCD", "mu": "JES_FlavorQCD"},
        ),
        "JES_RelativeBal": (
            years,
            sigs + bkgs,
            {"ele": "JES_RelativeBal", "mu": "JES_RelativeBal"},
        ),
        "JES_HF": (
            years,
            sigs + bkgs,
            {"ele": "JES_HF", "mu": "JES_HF"},
        ),
        "JES_BBEC1": (
            years,
            sigs + bkgs,
            {"ele": "JES_BBEC1", "mu": "JES_BBEC1"},
        ),
        "JES_EC2": (
            years,
            sigs + bkgs,
            {"ele": "JES_EC2", "mu": "JES_EC2"},
        ),
        "JES_Absolute": (
            years,
            sigs + bkgs,
            {"ele": "JES_Absolute", "mu": "JES_Absolute"},
        ),
        "UES": (
            years,
            sigs + bkgs,
            {"ele": "UES", "mu": "UES"},
        ),
    }

    JEC_systs_uncorrelated = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        JEC_systs_uncorrelated = {
            **JEC_systs_uncorrelated,
            **{
                f"JES_BBEC1_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_BBEC1_{yearlabel}", "mu": f"JES_BBEC1_{yearlabel}"},
                ),
                f"JES_RelativeSample_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_RelativeSample_{yearlabel}", "mu": f"JES_RelativeSample_{yearlabel}"},
                ),
                f"JES_EC2_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_EC2_{yearlabel}", "mu": f"JES_EC2_{yearlabel}"},
                ),
                f"JES_HF_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_HF_{yearlabel}", "mu": f"JES_HF_{yearlabel}"},
                ),
                f"JES_Absolute_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_Absolute_{yearlabel}", "mu": f"JES_Absolute_{yearlabel}"},
                ),
                f"JER_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": "JER", "mu": "JER"},
                ),
                f"JMR_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": "JMR", "mu": "JMR"},
                ),
                f"JMS_{year}": (
                    year,
                    sigs + bkgs,
                    {"ele": "JMS", "mu": "JMS"},
                ),
            },
        }

    SYST_DICT = {
        "common": {**COMMON_systs_correlated, **COMMON_systs_uncorrelated},
        "btag": {**BTAG_systs_correlated, **BTAG_systs_uncorrelated},
        "JEC": {**JEC_systs_correlated, **JEC_systs_uncorrelated},
        "LP": {**LP_systs_correlated},
    }

    return SYST_DICT
