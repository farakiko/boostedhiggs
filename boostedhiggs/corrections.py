import importlib.resources
import pickle
import warnings
from typing import Dict

import awkward as ak
import correctionlib
import numpy as np
from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import candidate, vector
from coffea.nanoevents.methods.nanoaod import GenParticleArray, JetArray

ak.behavior.update(vector.behavior)

btagWPs = {
    "deepJet": {
        "2016APV": {
            "L": 0.0508,
            "M": 0.2598,
            "T": 0.6502,
        },
        "2016": {
            "L": 0.0480,
            "M": 0.2489,
            "T": 0.6377,
        },
        "2017": {
            "L": 0.0532,
            "M": 0.3040,
            "T": 0.7476,
        },
        "2018": {
            "L": 0.0490,
            "M": 0.2783,
            "T": 0.7100,
        },
    },
    "deepCSV": {
        "2016APV": {
            "L": 0.2027,
            "M": 0.6001,
            "T": 0.8819,
        },
        "2016": {
            "L": 0.1918,
            "M": 0.5847,
            "T": 0.8767,
        },
        "2017": {
            "L": 0.1355,
            "M": 0.4506,
            "T": 0.7738,
        },
        "2018": {
            "L": 0.1208,
            "M": 0.4168,
            "T": 0.7665,
        },
    },
}

with importlib.resources.path("boostedhiggs.data", "msdcorr.json") as filename:
    msdcorr = correctionlib.CorrectionSet.from_file(str(filename))


def corrected_msoftdrop(fatjets):
    msdraw = np.sqrt(
        np.maximum(
            0.0,
            (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum().mass2,
        )
    )
    # msoftdrop = fatjets.msoftdrop
    msdfjcorr = msdraw / (1 - fatjets.rawFactor)

    corr = msdcorr["msdfjcorr"].evaluate(
        np.array(ak.flatten(msdfjcorr / fatjets.pt)),
        np.array(ak.flatten(np.log(fatjets.pt))),
        np.array(ak.flatten(fatjets.eta)),
    )
    corr = ak.unflatten(corr, ak.num(fatjets))
    corrected_mass = msdfjcorr * corr

    return corrected_mass


with importlib.resources.path("boostedhiggs.data", "ULvjets_corrections.json") as filename:
    vjets_kfactors = correctionlib.CorrectionSet.from_file(str(filename))


def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24)) & genpart.hasFlags(["fromHardProcess", "isLastCopy"])]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))


def add_VJets_kFactors(weights, genpart, dataset, events):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    znlosysts = [
        "d1kappa_EW",
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]
    wnlosysts = [
        "d1kappa_EW",
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )

    vpt = get_vpt(genpart)
    qcdcorr = np.ones_like(vpt)
    ewcorr = np.ones_like(vpt)

    # alternative QCD NLO correction (for WJets)
    # derived from https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2019/229
    alt_qcdcorr = {}
    alt_qcdcorr["nominal"] = np.ones_like(vpt)
    alt_qcdcorr["up"] = np.ones_like(vpt)
    alt_qcdcorr["down"] = np.ones_like(vpt)

    if "ZJetsToQQ_HT" in dataset or "DYJetsToLL_M-" in dataset:
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)

    elif "DYJetsToLL_Pt" in dataset or "DYJetsToLL_LHEFilterPtZ" in dataset:
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(znlosysts, qcdcorr, ewkcorr, vpt)

    elif "WJetsToLNu_1J" in dataset or "WJetsToLNu_0J" in dataset or "WJetsToLNu_2J" in dataset:
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(wnlosysts, qcdcorr, ewkcorr, vpt)

    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu_HT" in dataset or "WJetsToLNu_TuneCP5" in dataset:
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)

        # added by farouk
        """
        from: https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2019/229
        Bhadrons       Systematic
        0             1.628±0.005 - (1.339±0.020)·10−3 pT(V)
        1             1.586±0.027 - (1.531±0.112)·10−3 pT(V)
        2             1.440±0.048 - (0.925±0.203)·10−3 pT(V)
        """
        genjets = events.GenJet
        goodgenjets = genjets[(genjets.pt > 20.0) & (np.abs(genjets.eta) < 2.4)]

        nB0 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 0).to_numpy()
        nB1 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 1).to_numpy()
        nB2 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 2).to_numpy()

        alt_qcdcorr["nominal"][nB0] = 1.628 - (1.339 * 1e-3 * vpt[nB0])
        alt_qcdcorr["nominal"][nB1] = 1.586 - (1.531 * 1e-3 * vpt[nB1])
        alt_qcdcorr["nominal"][nB2] = 1.440 - (0.925 * 1e-3 * vpt[nB2])

        # added by farouk on Feb 13, 2025
        alt_qcdcorr["up"][nB0] = 1.628 + 0.005 - ((1.339 + 0.020) * 1e-3 * vpt[nB0])
        alt_qcdcorr["up"][nB1] = 1.586 + 0.027 - ((1.531 + 0.112) * 1e-3 * vpt[nB1])
        alt_qcdcorr["up"][nB2] = 1.440 + 0.048 - ((0.925 + 0.203) * 1e-3 * vpt[nB2])

        alt_qcdcorr["down"][nB0] = 1.628 - 0.005 - ((1.339 - 0.02) * 1e-3 * vpt[nB0])
        alt_qcdcorr["down"][nB1] = 1.586 - 0.027 - ((1.531 - 0.112) * 1e-3 * vpt[nB1])
        alt_qcdcorr["down"][nB2] = 1.440 - 0.048 - ((0.925 - 0.203) * 1e-3 * vpt[nB2])

    return ewcorr, qcdcorr, alt_qcdcorr


def add_ps_weight(weights, ps_weights):
    nweights = len(weights.weight())
    nom = np.ones(nweights)
    up_isr = np.ones(nweights)
    down_isr = np.ones(nweights)
    up_fsr = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if ps_weights is not None:
        if len(ps_weights[0]) == 4:
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
        else:
            warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")
    weights.add("PSISR", nom, up_isr, down_isr)
    weights.add("PSFSR", nom, up_fsr, down_fsr)


with importlib.resources.path("boostedhiggs.data", "EWHiggsCorrections.json") as filename:
    hew_kfactors = correctionlib.CorrectionSet.from_file(str(filename))


def add_HiggsEW_kFactors(genpart, dataset):
    """EW Higgs corrections"""

    def get_hpt():
        boson = ak.firsts(genpart[(genpart.pdgId == 25) & genpart.hasFlags(["fromHardProcess", "isLastCopy"])])
        return np.array(ak.fill_none(boson.pt, 0.0))

    hpt = get_hpt()
    ewknominal = np.ones_like(hpt)

    if "VBF" in dataset:
        ewkcorr = hew_kfactors["VBF_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        ewknominal[hpt >= 400] = ewknom[hpt >= 400]

    if (
        "WplusH" in dataset
        or "WminusH" in dataset
        or "ZH" in dataset
        or "HWminus" in dataset
        or "HWplus" in dataset
        or "HZ" in dataset
    ):
        ewkcorr = hew_kfactors["VH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        ewknominal[hpt >= 400] = ewknom[hpt >= 400]

    if "ttH" in dataset:
        ewkcorr = hew_kfactors["ttH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        ewknominal[hpt >= 400] = ewknom[hpt >= 400]

    return ewknominal


def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask

    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)


lumi_masks = {
    "2016": build_lumimask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
    "2017": build_lumimask("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
    "2018": build_lumimask("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
}


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "jec": ["JME", "fatJet_jerc.json.gz"],
    "jmar": ["JME", "jmar.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
}


def get_UL_year(year):
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"
    return f"{year}_UL"


def get_pog_json(obj, year):
    try:
        pog_json = pog_jsons[obj]
    except ValueError:
        print(f"No json for {obj}")
    year = get_UL_year(year)
    return f"{pog_correction_path}POG/{pog_json[0]}/{year}/{pog_json[1]}"


def get_btag_weights(
    year: str,
    jets: JetArray,
    jet_selector: ak.Array,
    wp: str = "M",
    algo: str = "deepJet",
    systematics: bool = False,
):
    """
    Following https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
    and 1a
    """

    try:
        cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))
    except FileNotFoundError:
        cset = correctionlib.CorrectionSet.from_file("btagging.json.gz")

    ul_year = get_UL_year(year)
    with importlib.resources.path("boostedhiggs.data", f"btageff_{algo}_{wp}_{ul_year}.coffea") as filename:
        efflookup = cutil.load(filename)

    def _btagSF(jets, flavour, syst="central"):
        j, nj = ak.flatten(jets), ak.num(jets)
        corrs = cset[f"{algo}_comb"] if flavour == "bc" else cset[f"{algo}_incl"]
        sf = corrs.evaluate(
            syst,
            wp,
            np.array(j.hadronFlavour),
            np.array(abs(j.eta)),
            np.array(j.pt),
        )
        return ak.unflatten(sf, nj)

    lightJets = jets[jet_selector & (jets.hadronFlavour == 0)]
    bcJets = jets[jet_selector & (jets.hadronFlavour > 0)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(lightJets, "light")
    bcSF = _btagSF(bcJets, "bc")

    lightPass = lightJets.btagDeepB > btagWPs[algo][year][wp]
    bcPass = bcJets.btagDeepB > btagWPs[algo][year][wp]

    # 1a method
    # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
    def _combine(eff, sf, passbtag):
        # tagged SF = SF*eff / eff = SF
        tagged_sf = ak.prod(sf[passbtag], axis=-1)
        # untagged SF = (1 - SF*eff) / (1 - eff)
        untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff))[~passbtag], axis=-1)
        return ak.fill_none(tagged_sf * untagged_sf, 1.0)

    ret_weights = {}

    # one common multiplicative SF is to be applied to the nominal prediction
    """
    bc_0btag = _get_weight(True, bcEff, bcSF)
    light_0btag = _get_weight(True, lightEff, lightSF)
    ret_weights["0btag_1b"] = bc_0btag * light_0btag

    bc_1pbtag = _get_weight(False, bcEff, bcSF)
    light_1pbtag = _get_weight(False, lightEff, lightSF)
    ret_weights["1pbtag_1b"] = bc_1pbtag * light_1pbtag
    """
    bc = _combine(bcEff, bcSF, bcPass)
    light = _combine(lightEff, lightSF, lightPass)
    ret_weights["weight_btag"] = bc * light

    # Separate uncertainties are applied for b/c jets and light jets
    if systematics:
        ret_weights[f"weight_btagSFlight{year}Up"] = _combine(lightEff, _btagSF(lightJets, "light", syst="up"), lightPass)
        ret_weights[f"weight_btagSFlight{year}Down"] = _combine(
            lightEff, _btagSF(lightJets, "light", syst="down"), lightPass
        )

        ret_weights[f"weight_btagSFbc{year}Up"] = _combine(bcEff, _btagSF(bcJets, "bc", syst="up"), bcPass)
        ret_weights[f"weight_btagSFbc{year}Down"] = _combine(bcEff, _btagSF(bcJets, "bc", syst="down"), bcPass)

        ret_weights["weight_btagSFlightCorrelatedUp"] = _combine(
            lightEff, _btagSF(lightJets, "light", syst="up_correlated"), lightPass
        )
        ret_weights["weight_btagSFlightCorrelatedDown"] = _combine(
            lightEff, _btagSF(lightJets, "light", syst="down_correlated"), lightPass
        )

        ret_weights["weight_btagSFbcCorrelatedUp"] = _combine(bcEff, _btagSF(bcJets, "bc", syst="up_correlated"), bcPass)
        ret_weights["weight_btagSFbcCorrelatedDown"] = _combine(bcEff, _btagSF(bcJets, "bc", syst="down_correlated"), bcPass)

    return ret_weights


"""
Lepton Scale Factors
----

Muons:
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018

- UL CorrectionLib html files:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/MUO_2017_UL_muon_Z.html
  e.g. one example of the correction json files can be found here:
  https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/raw/master/Run2/UL/2017/2017_trigger/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers_schemaV2.json
  - Trigger iso and non-iso
  - Isolation: We use RelIso<0.25 (LooseRelIso) with medium prompt ID
  - Reconstruction ID: We use medium prompt ID

Electrons:
- UL CorrectionLib htmlfiles:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
  - ID and Isolation:
    - wp90noiso for high pT electrons
    - wp90iso for low pT electrons
  - Reconstruction: RecoAbove20
  - Trigger: Derived using EGamma recommendation: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
"""

lepton_corrections = {
    "trigger_iso": {
        "muon": {  # For IsoMu24 (| IsoTkMu24 )
            "2016APV": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # preVBP
            "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # postVBF
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        },
    },
    "trigger_noniso": {
        "muon": {  # For Mu50 (| TkMu50 )
            "2016APV": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2016": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2017": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2018": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        },
    },
    "isolation": {
        "muon": {
            "2016APV": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2016": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2017": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2018": "NUM_LooseRelIso_DEN_MediumPromptID",
        },
        # "electron": {  # TODO: remove because these are ID SFs
        #     "2016APV": "wp90iso",
        #     "2016": "wp90iso",
        #     "2017": "wp90iso",
        #     "2018": "wp90iso",
        # },
    },
    "id": {
        "muon": {
            "2016APV": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2016": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2017": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2018": "NUM_MediumPromptID_DEN_TrackerMuons",
        },
        "electron": {
            "2016APV": "wp90noiso",
            "2016": "wp90noiso",
            "2017": "wp90noiso",
            "2018": "wp90noiso",
        },
    },
    "reco": {
        "electron": {
            "2016APV": "RecoAbove20",
            "2016": "RecoAbove20",
            "2017": "RecoAbove20",
            "2018": "RecoAbove20",
        },
    },
}


def add_lepton_weight(weights, lepton, year, lepton_type="muon"):
    ul_year = get_UL_year(year)
    if lepton_type == "electron":
        ul_year = ul_year.replace("_UL", "")

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    def set_isothreshold(corr, value, lepton_pt, lepton_type):
        """
        restrict values to 1 for some SFs if we are above/below the ISO threshold
        """
        iso_threshold = {"muon": 55.0, "electron": 120.0}[lepton_type]
        if corr == "trigger_iso":
            value[lepton_pt > iso_threshold] = 1.0
        elif corr == "trigger_noniso":
            value[lepton_pt < iso_threshold] = 1.0
        elif corr == "isolation":
            value[lepton_pt > iso_threshold] = 1.0
        elif corr == "id" and lepton_type == "electron":
            value[lepton_pt < iso_threshold] = 1.0
        return value

    def get_clip(lep_pt, lep_eta, lepton_type, corr=None):
        clip_pt = [0.0, 2000]
        clip_eta = [-2.4999, 2.4999]
        if lepton_type == "electron":
            clip_pt = [10.0, 499.999]
            if corr == "reco":
                clip_pt = [20.1, 499.999]
        elif lepton_type == "muon":
            clip_pt = [30.0, 1000.0]
            clip_eta = [0.0, 2.3999]
            if corr == "trigger_noniso":
                clip_pt = [52.0, 1000.0]
        lepton_pt = np.clip(lep_pt, clip_pt[0], clip_pt[1])
        lepton_eta = np.clip(lep_eta, clip_eta[0], clip_eta[1])
        return lepton_pt, lepton_eta

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.array(ak.fill_none(lepton.eta, 0.0))
    if lepton_type == "muon":
        lep_eta = np.abs(lep_eta)

    for corr, corrDict in lepton_corrections.items():
        if lepton_type not in corrDict.keys():
            continue
        if year not in corrDict[lepton_type].keys():
            continue

        json_map_name = corrDict[lepton_type][year]

        lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)

        values = {}
        if lepton_type == "muon":
            values["nominal"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "nominal")
        else:
            values["nominal"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sf", json_map_name, lepton_eta, lepton_pt)

        if (lepton_type == "muon") and (corr == "id"):
            # split the stat. and syst. unc. for the id SF for muons
            for unc_type in ["stat", "syst"]:
                values[unc_type] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, unc_type)
                for key, val in values.items():
                    values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

                up = values["nominal"] * (1 + values[unc_type])
                down = values["nominal"] * (1 - values[unc_type])
                weights.add(f"{corr}_{lepton_type}_{unc_type}", values["nominal"], up, down)

        else:
            if lepton_type == "muon":
                values["up"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systup")
                values["down"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systdown")
            else:
                values["up"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfup", json_map_name, lepton_eta, lepton_pt)
                values["down"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfdown", json_map_name, lepton_eta, lepton_pt)

            for key, val in values.items():
                values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

            # add weights (for now only the nominal weight)
            weights.add(f"{corr}_{lepton_type}", values["nominal"], values["up"], values["down"])


def get_pileup_weight(year: str, mod: str, nPU: np.ndarray):
    """
    Should be able to do something similar to lepton weight but w pileup
    e.g. see here: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
    """
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year + mod))

    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}

    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

    return values


def add_pileup_weight(weights: Weights, year: str, mod: str, nPU: np.ndarray):
    """Separate wrapper function in case we just want the values separately."""
    values = get_pileup_weight(year, mod, nPU)
    weights.add("pileup", values["nominal"], values["up"], values["down"])


def add_pileupid_weights(weights: Weights, year: str, mod: str, jets: JetArray, genjets, wp: str = "L"):
    """Pileup ID scale factors
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL#Data_MC_Efficiency_Scale_Factors

    Takes ak4 jets which already passed the pileup ID WP.
    Only applies to jets with pT < 50 GeV and those geometrically matched to a gen jet.
    """

    # pileup ID should only be used for jets with pT < 50
    jets = jets[(jets.pt < 50) & (jets.pt > 12.5)]
    # check that there's a geometrically matched genjet (99.9% are, so not really necessary...)
    jets = jets[ak.any(jets.metric_table(genjets) < 0.4, axis=-1)]

    sf_cset = correctionlib.CorrectionSet.from_file(get_pog_json("jmar", year + mod))["PUJetID_eff"]

    # save offsets to reconstruct jagged shape
    offsets = jets.pt.layout.offsets

    sfs_var = []
    for var in ["nom", "up", "down"]:
        # correctionlib < 2.3 doesn't accept jagged arrays (but >= 2.3 needs awkard v2)
        sfs = sf_cset.evaluate(ak.flatten(jets.eta), ak.flatten(jets.pt), var, "L")
        # reshape flat effs
        sfs = ak.Array(ak.layout.ListOffsetArray64(offsets, ak.layout.NumpyArray(sfs)))
        # product of SFs across arrays, automatically defaults empty lists to 1
        sfs_var.append(ak.prod(sfs, axis=1))

    weights.add("pileupIDSF", *sfs_var)


"""
JECs
----
"""

# find corrections path using this file's path
try:
    with importlib.resources.path("boostedhiggs.data", "jec_compiled.pkl") as filename:
        with open(filename, "rb") as filehandler:
            jmestuff = pickle.load(filehandler)
        ak4jet_factory = jmestuff["jet_factory"]
        fatjet_factory = jmestuff["fatjet_factory"]
        met_factory = jmestuff["met_factory"]
except KeyError:
    print("Failed loading compiled JECs")


def _add_jec_variables(jets: JetArray, event_rho: ak.Array) -> JetArray:
    """add variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    # gen pT needed for smearing
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def get_jec_jets(events, jets, year: str, isData: bool = False, jecs: Dict[str, str] = None, fatjets: bool = True):
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    Eventually update to V5 JECs once I figure out what's going on with the 2017 UL V5 JER scale factors

    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/

    If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
    """

    jec_vars = ["pt"]  # variables we are saving that are affected by JECs
    if fatjets:
        jet_factory = fatjet_factory
    else:
        jet_factory = ak4jet_factory

    apply_jecs = not (not ak.any(jets.pt) or isData)

    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_UL_year(year)}mc".replace("_UL", "")

    # fatjet_factory.build gives an error if there are no fatjets in event
    if apply_jecs:
        jets = jet_factory[corr_key].build(_add_jec_variables(jets, events.fixedGridRhoFastjetAll), jec_cache)

    # return only fatjets if no jecs given
    if jecs is None:
        return jets

    jec_shifted_vars = {}

    for jec_var in jec_vars:
        tdict = {"": jets[jec_var]}
        if apply_jecs:
            for key, shift in jecs.items():
                for var in ["up", "down"]:
                    tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

        jec_shifted_vars[jec_var] = tdict

    return jets, jec_shifted_vars


"""
The following are added on Feb9_2024 by Farouk.
"""
PAD_VAL = -99999


def pad_val(
    arr: ak.Array,
    target: int,
    value: float = PAD_VAL,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret


jmsr_vars = ["msoftdrop"]

jmsValues = {}
jmrValues = {}

# jet mass resolution: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# nominal, down, up (these are switched in the github!!!)
jmrValues["msoftdrop_twiki"] = {
    "2016": [1.0, 0.8, 1.2],
    "2017": [1.09, 1.04, 1.14],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [1.09, 1.04, 1.14],
}
# convert from SF on sigma to value to smear mass by
# note: clamped to 1.0 (so 2016 JMR down needs to be interpolated by combine)
jmrValues["msoftdrop"] = {}
for year in jmrValues["msoftdrop_twiki"]:
    rel_res = 0.12 if year == "2016" else 0.10
    jmrValues["msoftdrop"][year] = [
        1 + np.sqrt(max(sf_res**2 - 1, 0)) * rel_res for sf_res in jmrValues["msoftdrop_twiki"][year]
    ]

# jet mass scale
# W-tagging PUPPI softdrop JMS values: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# 2016 values
jmsValues["msoftdrop"] = {
    "2016": [1.00, 0.9906, 1.0094],  # nominal, down, up
    "2017": [0.982, 0.978, 0.986],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [0.982, 0.978, 0.986],
}


def get_jmsr(fatjets, num_jets: int, year: str, isData: bool = False, seed: int = 42) -> Dict:
    """Calculates post JMS/R masses and shifts"""
    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = pad_val(fatjets[mkey], num_jets, axis=1)

        if isData:
            tdict[""] = mass
        else:
            np.random.seed(seed)
            smearing = np.random.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = [((smearing * max(jmrValues[mkey][year][i] - 1, 0)) + 1) for i in range(3)]
            jms_nom, jms_down, jms_up = jmsValues[mkey][year]

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass_jms * jmr_nom
            tdict["JMS_down"] = mass_jmr * jms_down
            tdict["JMS_up"] = mass_jmr * jms_up
            tdict["JMR_down"] = mass_jms * jmr_down
            tdict["JMR_up"] = mass_jms * jmr_up

        jmsr_shifted_vars[mkey] = tdict

    return jmsr_shifted_vars


def getJECVariables(fatjetvars, candidatelep_p4, met, pt_shift=None, met_shift=None):
    """
    get variables affected by JES_up, JES_down, JER_up, JER_down, UES_up, UES_down
    """
    variables = {}

    ptlabel = pt_shift if pt_shift is not None else ""

    if met_shift is not None:
        metlabel = met_shift
        if met_shift == "UES_up":
            metvar = met.MET_UnclusteredEnergy.up
        elif met_shift == "UES_down":
            metvar = met.MET_UnclusteredEnergy.down
    else:
        metlabel = ""
        if ptlabel != "":
            if ptlabel == "JES_up":
                metvar = met.JES_jes.up
            elif ptlabel == "JES_down":
                metvar = met.JES_jes.down
            # elif ptlabel == "JER_up":
            #     metvar = met.JER.up
            # elif ptlabel == "JER_down":
            #     metvar = met.JER.down
            else:
                if "up" in ptlabel:
                    metvar = met[ptlabel.replace("_up", "")].up
                elif "down" in ptlabel:
                    metvar = met[ptlabel.replace("_down", "")].down
        else:
            metvar = met

    shift = ptlabel + metlabel

    candidatefj = ak.zip(
        {
            "pt": fatjetvars[f"fj_pt{ptlabel}"],
            "eta": fatjetvars["fj_eta"],
            "phi": fatjetvars["fj_phi"],
            "mass": fatjetvars["fj_mass"],
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    candidateNeutrinoJet = ak.zip(
        {
            "pt": metvar.pt,
            "eta": candidatefj.eta,
            "phi": met.phi,
            "mass": 0,
            "charge": 0,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    rec_W_lnu = candidatelep_p4 + candidateNeutrinoJet
    rec_W_qq = candidatefj - candidatelep_p4
    rec_higgs = rec_W_qq + rec_W_lnu

    variables[f"rec_higgs_m{shift}"] = rec_higgs.mass
    variables[f"rec_higgs_pt{shift}"] = rec_higgs.pt

    if shift == "":
        variables[f"rec_W_qq_m{shift}"] = rec_W_qq.mass
        variables[f"rec_W_qq_pt{shift}"] = rec_W_qq.pt

        variables[f"rec_W_lnu_m{shift}"] = rec_W_lnu.mass
        variables[f"rec_W_lnu_pt{shift}"] = rec_W_lnu.pt

    return variables


def getJMSRVariables(fatjetvars, candidatelep_p4, met, mass_shift=None):
    """
    get variables affected by JMS_up, JMS_down, JMR_up, JMR_down
    """
    variables = {}

    candidatefj = ak.zip(
        {
            "pt": fatjetvars["fj_pt"],
            "eta": fatjetvars["fj_eta"],
            "phi": fatjetvars["fj_phi"],
            "mass": fatjetvars[f"fj_mass{mass_shift}"],
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    candidateNeutrinoJet = ak.zip(
        {
            "pt": met.pt,
            "eta": candidatefj.eta,
            "phi": met.phi,
            "mass": 0,
            "charge": 0,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    rec_W_lnu = candidatelep_p4 + candidateNeutrinoJet
    rec_W_qq = candidatefj - candidatelep_p4
    rec_higgs = rec_W_qq + rec_W_lnu

    variables[f"rec_higgs_m{mass_shift}"] = rec_higgs.mass
    variables[f"rec_higgs_pt{mass_shift}"] = rec_higgs.pt

    if mass_shift == "":
        variables[f"rec_W_qq_m{mass_shift}"] = rec_W_qq.mass
        variables[f"rec_W_qq_pt{mass_shift}"] = rec_W_qq.pt

        variables[f"rec_W_lnu_m{mass_shift}"] = rec_W_lnu.mass
        variables[f"rec_W_lnu_pt{mass_shift}"] = rec_W_lnu.pt

    return variables


def add_TopPtReweighting(weights, topPt):

    toppt_weight1 = np.exp(0.0615 - 0.0005 * np.clip(topPt[:, 0], 0.0, 500.0))
    toppt_weight2 = np.exp(0.0615 - 0.0005 * np.clip(topPt[:, 1], 0.0, 500.0))

    nominal = np.sqrt(toppt_weight1 * toppt_weight2)

    weights.add("TopPtReweight", nominal, np.ones_like(nominal), nominal**2)

    # weights.add(
    #     "TopPtReweight",
    #     np.sqrt(toppt_weight1 * toppt_weight2),
    #     np.ones_like(toppt_weight1),
    #     np.sqrt(toppt_weight1 * toppt_weight2),
    # )

    return nominal


def get_JetVetoMap(jets, year: str):
    """
    Jet Veto Maps recommendation from JERC.

    All JERC analysers and anybody doing precision measurements with jets should use the strictest veto maps.
    Recommended for analyses strongly relying on events with large MET.

    Recommendation link: https://cms-jerc.web.cern.ch/Recommendations/#run-3_2

    Get event selection that rejects events with jets in the veto map.
    """

    # Jet veto maps are synced daily here
    era_tags = {"2016APV": "2016preVFP_UL", "2016": "2016postVFP_UL", "2017": "2017_UL", "2018": "2018_UL"}
    era_tag = era_tags[year]

    fname = f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{era_tag}/jetvetomaps.json.gz"

    # correctionlib doesn't support awkward arrays, so we have to flatten them out
    j, nj = ak.flatten(jets), ak.num(jets)
    j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
    j_eta = np.clip(np.array(j.eta), -4.7, 4.7)

    # load the correction set
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    # apply the correction and recreate the awkward array shape
    hname = {"2016APV": "Summer19UL16_V1", "2016": "Summer19UL16_V1", "2017": "Summer19UL17_V1", "2018": "Summer19UL18_V1"}

    weight = evaluator[hname[year]].evaluate("jetvetomap", j_eta, j_phi)
    weight_ak = ak.unflatten(np.array(weight), counts=nj)

    # any non-zero weight means the jet is vetoed
    jetmask = weight_ak == 0

    # events are selected only if they have no jets in the vetoed region
    eventmask = ak.sum(weight_ak, axis=-1) == 0

    return jetmask, eventmask


"""
------------------- Lund plane reweighting ------------------- #
"""


from .utils import (
    ELE_PDGID,
    FILL_NONE_VALUE,
    GAMMA_PDGID,
    GEN_FLAGS,
    HIGGS_PDGID,
    MU_PDGID,
    TAU_PDGID,
    TOP_PDGID,
    W_PDGID,
    Z_PDGID,
    b_PDGID,
    get_pid_mask,
)


def getGenLepGenQuarks(dataset, genparts: GenParticleArray):
    """Gen matching for Higgs samples or Top samples.

    An edited version of the function that returns the genquark info that is useful for LP method.
    """
    if "HToWW" in dataset:
        higgs = ak.firsts(genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)])

        children_mask = get_pid_mask(higgs.children, [W_PDGID], byall=False)

        # order by mass, select lower mass child as V* and higher as V
        higgs_children = higgs.children[children_mask]

        # VV daughters
        # requires coffea-0.7.21
        all_daus = higgs_children.distinctChildrenDeep  # TODO: double check it's always 4
        all_daus_flat = ak.flatten(all_daus, axis=2)  # flattents the daughters of the two Ws
        all_daus_flat_pdgId = abs(all_daus_flat.pdgId)

        leptons = (all_daus_flat_pdgId == ELE_PDGID) | (all_daus_flat_pdgId == MU_PDGID) | (all_daus_flat_pdgId == TAU_PDGID)
        quarks = all_daus_flat_pdgId < b_PDGID

        lepVars = {
            "lepton_pt": all_daus_flat[leptons].pt,
            "lepton_eta": all_daus_flat[leptons].eta,
            "lepton_phi": all_daus_flat[leptons].phi,
            "lepton_mass": all_daus_flat[leptons].mass,
        }

        quarkVars = {
            "quark_pt": all_daus_flat[quarks].pt,
            "quark_eta": all_daus_flat[quarks].eta,
            "quark_phi": all_daus_flat[quarks].phi,
            "quark_mass": all_daus_flat[quarks].mass,
        }

        return lepVars, quarkVars, None

    elif "TT" in dataset:
        tops = genparts[get_pid_mask(genparts, TOP_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

        # take all possible daughters!
        daughters = ak.flatten(tops.distinctChildren, axis=2)
        daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
        daughters_pdgId = abs(daughters.pdgId)

        wboson_daughters = ak.flatten(daughters[(daughters_pdgId == W_PDGID)].distinctChildren, axis=2)
        wboson_daughters = wboson_daughters[wboson_daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
        wboson_daughters_pdgId = abs(wboson_daughters.pdgId)

        leptons = (
            (wboson_daughters_pdgId == ELE_PDGID)
            | (wboson_daughters_pdgId == MU_PDGID)
            | (wboson_daughters_pdgId == TAU_PDGID)
        )

        quarks = wboson_daughters_pdgId <= b_PDGID

        bquarks = daughters[(daughters_pdgId == b_PDGID)]

        # bquarksdaughters = ak.flatten(bquarks.distinctChildren, axis=2)
        # bquarksdaughters_pdgId = abs(bquarksdaughters.pdgId)

        # bquarkslep = (
        #     (bquarksdaughters_pdgId == ELE_PDGID)
        #     | (bquarksdaughters_pdgId == MU_PDGID)
        #     | (bquarksdaughters_pdgId == TAU_PDGID)
        # )

        # print("bquarkslep", bquarkslep)

        lepVars = {
            "lepton_pt": wboson_daughters[leptons].pt,
            "lepton_eta": wboson_daughters[leptons].eta,
            "lepton_phi": wboson_daughters[leptons].phi,
            "lepton_mass": wboson_daughters[leptons].mass,
        }

        quarkVars = {
            "quark_pt": wboson_daughters[quarks].pt,
            "quark_eta": wboson_daughters[quarks].eta,
            "quark_phi": wboson_daughters[quarks].phi,
            "quark_mass": wboson_daughters[quarks].mass,
        }

        bquarksVars = {
            "bquarks_pt": bquarks.pt,
            "bquarks_eta": bquarks.eta,
            "bquarks_phi": bquarks.phi,
            "bquarks_mass": bquarks.mass,
        }

        # def ang_dist(phi1, phi2):
        #     phi1 = phi1 % (2.0 * np.pi)
        #     phi2 = phi2 % (2.0 * np.pi)
        #     dphi = phi1 - phi2
        #     if len(dphi.shape) > 0:
        #         dphi[dphi < -np.pi] += 2.0 * np.pi
        #         dphi[dphi > np.pi] -= 2.0 * np.pi
        #     else:
        #         if dphi < -np.pi:
        #             dphi += 2.0 * np.pi
        #         if dphi > np.pi:
        #             dphi -= 2.0 * np.pi

        #     return dphi

        # topsdphi = ang_dist(ak.to_numpy(tops[:, 0].phi), ak.to_numpy(tops[:, 1].phi))

        return lepVars, quarkVars, bquarksVars

    else:

        vs = genparts[get_pid_mask(genparts, [W_PDGID, Z_PDGID], byall=False) * genparts.hasFlags(GEN_FLAGS)]
        daughters = ak.flatten(vs.distinctChildren, axis=2)
        daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
        daughters_pdgId = abs(daughters.pdgId)

        leptons = (daughters_pdgId == ELE_PDGID) | (daughters_pdgId == MU_PDGID) | (daughters_pdgId == TAU_PDGID)
        quarks = daughters_pdgId < b_PDGID

        lepVars = {
            "lepton_pt": daughters[leptons].pt,
            "lepton_eta": daughters[leptons].eta,
            "lepton_phi": daughters[leptons].phi,
            "lepton_mass": daughters[leptons].mass,
        }

        quarkVars = {
            "quark_pt": daughters[quarks].pt,
            "quark_eta": daughters[quarks].eta,
            "quark_phi": daughters[quarks].phi,
            "quark_mass": daughters[quarks].mass,
        }

        return lepVars, quarkVars, None


def getLPweights(dataset, events, candidatefj, fj_idx_lep, candidatelep_p4):
    """
    Relies on
        (1) ak8_jets
        (2) gen_parts_eta_phi
        (3) pf_cands
    """

    try:
        candidatefj = candidatefj - candidatelep_p4
    except ValueError:
        candidatefj = candidatefj

    lepVars, quarkVars, bquarksVars = getGenLepGenQuarks(dataset, events.GenPart)

    ak8_jets = np.array(
        np.stack(
            (np.array(candidatefj.pt), np.array(candidatefj.eta), np.array(candidatefj.phi), np.array(candidatefj.mass)),
            axis=1,
        )  # four vector for HWW jet
    )

    skim_vars = {
        "eta": "Eta",
        "phi": "Phi",
        "mass": "Mass",
        "pt": "Pt",
    }

    # prepare eta, phi array only for 2q, used for Lund Plane reweighting
    # since it only takes quarks gen-level 4-vector as input
    Gen2qVars = {
        f"Gen2q{var}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(quarkVars[f"quark_{key}"], 2, axis=1, clip=True),
                FILL_NONE_VALUE,
            )
        )
        for key, var in skim_vars.items()
    }
    gen_parts_eta_phi = np.array(np.dstack((Gen2qVars["Gen2qEta"], Gen2qVars["Gen2qPhi"])))
    gen_parts_pt_mass = np.array(np.dstack((Gen2qVars["Gen2qPt"], Gen2qVars["Gen2qMass"])))

    if "TT" in dataset:

        GenbquarksVars = {
            f"Genbquarks{var}": ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(bquarksVars[f"bquarks_{key}"], 2, axis=1, clip=True),
                    FILL_NONE_VALUE,
                )
            )
            for key, var in skim_vars.items()
        }

        bgen_parts_eta_phi = np.array(np.dstack((GenbquarksVars["GenbquarksEta"], GenbquarksVars["GenbquarksPhi"])))

    else:
        bgen_parts_eta_phi = None

    # prepare the Gen lepton in case we mask objects around it
    GenlepVars = {
        f"Genlep{var}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lepVars[f"lepton_{key}"], 1, axis=1, clip=True),
                FILL_NONE_VALUE,
            )
        )
        for key, var in skim_vars.items()
    }

    GenLep = ak.zip(
        {
            "pt": GenlepVars["GenlepPt"],
            "eta": GenlepVars["GenlepEta"],
            "phi": GenlepVars["GenlepPhi"],
            "mass": GenlepVars["GenlepMass"],
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # PF candidates in the AK8 jet
    HWW_FatJetPFCands = events.FatJetPFCands.jetIdx == ak.firsts(fj_idx_lep)
    HWW_FatJetPFCands_pFCandsIdx = events.FatJetPFCands.pFCandsIdx[HWW_FatJetPFCands]

    jet_pfcands = events.PFCands[HWW_FatJetPFCands_pFCandsIdx]
    pt_array = ak.Array(jet_pfcands.pt)
    eta_array = ak.Array(jet_pfcands.eta)
    phi_array = ak.Array(jet_pfcands.phi)
    mass_array = ak.Array(jet_pfcands.mass)
    pid_array = ak.Array(abs(jet_pfcands.pdgId))

    # build any masking you want
    msk_lep = (pid_array == ELE_PDGID) | (pid_array == MU_PDGID) | (pid_array == TAU_PDGID)
    msk_gamma = pid_array == GAMMA_PDGID
    msk_delta = GenLep.delta_r(jet_pfcands) < 0.2
    msk_pt = pt_array < 1

    msk = ((msk_lep | msk_gamma) & msk_delta) | msk_pt

    # apply the masking by selecting particles that don't have "msk"
    selected_pt = pt_array[~msk]
    selected_eta = eta_array[~msk]
    selected_phi = phi_array[~msk]
    selected_mass = mass_array[~msk]

    # pad the selected 4-vec array up to length of 150 to match the Lund Plane input
    selected_pt_padded = pad_val(selected_pt, 150, 0, 1, True)
    selected_eta_padded = pad_val(selected_eta, 150, 0, 1, True)
    selected_phi_padded = pad_val(selected_phi, 150, 0, 1, True)
    selected_mass_padded = pad_val(selected_mass, 150, 0, 1, True)

    pf_cands_px = selected_pt_padded * np.cos(selected_phi_padded)
    pf_cands_py = selected_pt_padded * np.sin(selected_phi_padded)
    pf_cands_pz = selected_pt_padded * np.sinh(selected_eta_padded)
    pf_cands_E = np.sqrt(pf_cands_px**2 + pf_cands_py**2 + pf_cands_pz**2 + selected_mass_padded**2)

    pf_cands = np.dstack((pf_cands_px, pf_cands_py, pf_cands_pz, pf_cands_E))

    genlep = np.array(
        np.stack(
            (GenlepVars["GenlepPt"], GenlepVars["GenlepEta"], GenlepVars["GenlepPhi"], GenlepVars["GenlepMass"]),
            axis=1,
        )
    )

    return pf_cands, gen_parts_eta_phi, gen_parts_pt_mass, ak8_jets, bgen_parts_eta_phi, genlep
