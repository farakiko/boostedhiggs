import importlib.resources
import json
import warnings

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, column_accumulator

from boostedhiggs.corrections import (
    add_lepton_weight,
    add_pileup_weight,
    add_VJets_kFactors,
    corrected_msoftdrop,
)
from boostedhiggs.utils import match_H

# we suppress ROOT warnings where our input ROOT tree has duplicate branches - these are handled correctly.
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


class TriggerEfficienciesProcessor(ProcessorABC):
    """Accumulates yields from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self, year="2017"):
        super(TriggerEfficienciesProcessor, self).__init__()
        self._year = year
        self._trigger_dict = {
            "2017": {
                "ele35": ["Ele35_WPTight_Gsf"],
                "ele115": ["Ele115_CaloIdVT_GsfTrkIdT"],
                "Photon200": ["Photon200"],
                "Mu50": ["Mu50"],
                "IsoMu27": ["IsoMu27"],
                "OldMu100": ["OldMu100"],
                "TkMu100": ["TkMu100"],
            }
        }[self._year]
        self._triggers = {
            "ele": ["ele35", "ele115", "Photon200"],
            "mu": ["Mu50", "IsoMu27", "OldMu100", "TkMu100"],
        }

        self._channels = ["ele", "mu"]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

    def pad_val(
        self,
        arr: ak.Array,
        target: int,
        value: float,
        axis: int = 0,
        to_numpy: bool = True,
    ):
        """pads awkward array up to `target` index along axis `axis` with value `value`,
        optionally converts to numpy array"""
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=True), value)
        return ret.to_numpy() if to_numpy else ret

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger histograms from input NanoAOD events"""
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.weights_per_ch = {}

        def pad_val_nevents(arr: ak.Array):
            """pad values with the length equal to the number of events"""
            return self.pad_val(arr, nevents, -1)

        # skimmed events for different channels
        out = {}
        for channel in self._channels:
            out[channel] = {}
            out[channel]["triggers"] = {}

        """ Save OR of triggers as booleans """
        for channel in self._channels:
            HLT_triggers = {}
            for t in self._triggers[channel]:
                HLT_triggers["HLT_" + t] = np.any(
                    np.array([events.HLT[trigger] for trigger in self._trigger_dict[t] if trigger in events.HLT.fields]),
                    axis=0,
                )
            out[channel]["triggers"] = {**out[channel]["triggers"], **HLT_triggers}

        ######################
        # METFLITERS
        ######################

        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        ######################
        # OBJECT DEFINITION
        ######################

        # OBJECT: taus
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        # OBJECT: muons
        muons = ak.with_field(events.Muon, 0, "flavor")

        loose_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
            & (((muons.pfRelIso04_all < 0.25) & (muons.pt < 55)) | (muons.pt >= 55))
        )

        tight_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )

        n_loose_muons = ak.sum(loose_muons, axis=1)
        good_muons = tight_muons

        n_good_muons = ak.sum(good_muons, axis=1)

        # OBJECT: electrons
        electrons = ak.with_field(events.Electron, 1, "flavor")

        loose_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WPL)
            & (((electrons.pfRelIso03_all < 0.25) & (electrons.pt < 120)) | (electrons.pt >= 120))
        )

        tight_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WP90)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
            # additional cuts
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
        )

        n_loose_electrons = ak.sum(loose_electrons, axis=1)
        good_electrons = tight_electrons

        n_good_electrons = ak.sum(good_electrons, axis=1)

        # OBJECT: candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        # fatjets["msdcorr"] = fatjets.msoftdrop
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        # OBJECT: candidate fatjet
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        met = events.MET

        NumFatjets = ak.num(good_fatjets)

        # delta R between AK8 jet and lepton
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        # delta phi MET and higgs candidate
        met_fj_dphi = candidatefj.delta_phi(met)

        ######################
        # Baseline weight
        ######################

        self.weights.add("genweight", events.genWeight)
        self.weights.add(
            "L1Prefiring",
            events.L1PreFiringWeight.Nom,
            events.L1PreFiringWeight.Up,
            events.L1PreFiringWeight.Dn,
        )
        add_pileup_weight(self.weights, self._year, "", nPU=ak.to_numpy(events.Pileup.nPU))
        add_VJets_kFactors(self.weights, events.GenPart, dataset, events)

        ######################
        # Baseline selection
        ######################

        for channel in self._channels:
            selection = PackedSelection()
            if channel == "mu":
                add_lepton_weight(self.weights, candidatelep, self._year, "muon")
                selection.add(
                    "OneLep",
                    ((n_good_muons == 1) & (n_loose_electrons == 0)),
                )
                selection.add("NoTaus", (n_loose_taus_mu == 0))

            elif channel == "ele":
                add_lepton_weight(self.weights, candidatelep, self._year, "electron")
                selection.add(
                    "OneLep",
                    ((n_loose_muons == 0) & (n_good_electrons == 1)),
                )
                selection.add("NoTaus", (n_loose_taus_ele == 0))

            selection.add("METFilters", (metfilters))
            selection.add("AtLeastOneFatJet", (NumFatjets >= 1))
            selection.add("CandidateJetpT", (candidatefj.pt > 250))
            selection.add("LepInJet", (lep_fj_dr < 0.8))
            selection.add("JetLepOverlap", (lep_fj_dr > 0.03))
            selection.add("dPhiJetMET", (np.abs(met_fj_dphi) < 1.57))
            selection.add("MET", (met.pt > 20))
            selection.add("CandidateJetSoftdropMass", (candidatefj.msdcorr > 40))

            ######################
            # variables to store
            ######################
            out[channel]["vars"] = {}
            out[channel]["vars"]["fj_pt"] = pad_val_nevents(candidatefj.pt)
            out[channel]["vars"]["fj_eta"] = pad_val_nevents(candidatefj.eta)
            out[channel]["vars"]["fj_msoftdrop"] = pad_val_nevents(candidatefj.msoftdrop)
            out[channel]["vars"]["met_pt"] = pad_val_nevents(met.pt)

            out[channel]["vars"]["lep_pt"] = pad_val_nevents(candidatelep.pt)
            out[channel]["vars"]["lep_eta"] = pad_val_nevents(candidatelep.eta)

            if "HToWW" in dataset:
                genVars, _ = match_H(events.GenPart, candidatefj)
                matchedH_pt = genVars["fj_genH_pt"]
            else:
                matchedH_pt = ak.zeros_like(candidatefj.pt)
            out[channel]["vars"]["fj_genH_pt"] = pad_val_nevents(matchedH_pt).data

            out[channel]["weights"] = {}
            for key in self.weights._weights.keys():
                # store the individual weights (ONLY for now until we debug)
                out[channel]["weights"][f"weight_{key}"] = self.weights.partial_weight([key])
                if channel in self.weights_per_ch.keys():
                    self.weights_per_ch[channel].append(key)

            # use column accumulators
            for key_ in out[channel].keys():
                out[channel][key_] = {
                    key: column_accumulator(value[selection.all(*selection.names)])
                    for (key, value) in out[channel][key_].items()
                }

        return {self._year: {dataset: {"nevents": nevents, "skimmed_events": out}}}

    def postprocess(self, accumulator):
        for year, datasets in accumulator.items():
            for dataset, output in datasets.items():
                for channel in output["skimmed_events"].keys():
                    for key_ in output["skimmed_events"][channel].keys():
                        output["skimmed_events"][channel][key_] = {
                            key: value.value for (key, value) in output["skimmed_events"][channel][key_].items()
                        }

        return accumulator