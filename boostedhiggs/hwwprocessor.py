import importlib.resources
import json
import logging
import os
import pathlib
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate

logger = logging.getLogger(__name__)

from boostedhiggs.corrections import (
    add_HiggsEW_kFactors,
    add_lepton_weight,
    add_pileup_weight,
    add_pileupid_weights,
    add_ps_weight,
    add_TopPtReweighting,
    add_VJets_kFactors,
    btagWPs,
    get_btag_weights,
    get_jec_jets,
    get_JetVetoMap,
    get_jmsr,
    get_pileup_weight,
    getJECVariables,
    getJMSRVariables,
    met_factory,
)
from boostedhiggs.utils import (
    ELE_PDGID,
    MU_PDGID,
    VScore,
    get_pid_mask,
    match_H,
    match_Top,
    match_V,
    sigs,
)

from .run_tagger_inference import runInferenceTriton

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")


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


class HwwProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        systematics=False,
        getLPweights=False,
        uselooselep=False,
        fakevalidation=False,
        no_trigger=False,
        no_selection=False,
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels or ["ele", "mu"]

        self._systematics = systematics
        self._getLPweights = getLPweights
        self._uselooselep = uselooselep
        self._fakevalidation = fakevalidation
        self._apply_trigger = not no_trigger
        self._apply_selection = not no_selection
        self._output_location = output_location
        self._inference = inference

        def load_json_config(file_name):
            with importlib.resources.path("boostedhiggs.data", file_name) as path:
                with open(path, "r") as f:
                    return json.load(f)[self._year]

        self._HLTs = load_json_config("triggers.json")
        self._metfilters = load_json_config("metfilters.json")

        self.dataset_per_ch = {
            "ele": "EGamma" if self._year == "2018" else "SingleElectron",
            "mu": "SingleMuon",
        }

        self.jecs = {
            "JES": "JES_jes",
            "JER": "JER",
            # individual sources
            "JES_FlavorQCD": "JES_FlavorQCD",
            "JES_RelativeBal": "JES_RelativeBal",
            "JES_HF": "JES_HF",
            "JES_BBEC1": "JES_BBEC1",
            "JES_EC2": "JES_EC2",
            "JES_Absolute": "JES_Absolute",
            f"JES_BBEC1_{self._year}": f"JES_BBEC1_{self._year}",
            f"JES_RelativeSample_{self._year}": f"JES_RelativeSample_{self._year}",
            f"JES_EC2_{self._year}": f"JES_EC2_{self._year}",
            f"JES_HF_{self._year}": f"JES_HF_{self._year}",
            f"JES_Absolute_{self._year}": f"JES_Absolute_{self._year}",
            "JES_Total": "JES_Total",
        }

        self.tagger_resources_path = os.path.join(str(pathlib.Path(__file__).parent.resolve()), "tagger_resources")

    @property
    def accumulator(self):
        """Required Coffea interface property. Not used in this processor, returns a dummy accumulator."""
        return self._accumulator

    def _save_dfs_parquet(self, fname, dfs_dict, ch):
        """Save the given pandas DataFrame dictionary as a Parquet file to the output directory."""

        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def _ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        """Converts an Awkward Array collection to a flat pandas DataFrame for to write the output."""

        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """
        Adds a boolean mask to the PackedSelection object for the given channel(s) and updates the
        cutflow dictionary with the sum of selected events or weights.
        """

        channels = self._channels if channel == "all" else [channel]

        for ch in channels:
            if ch not in self._channels:
                logger.warning(f"Attempted to add selection to unexpected channel: {ch} not in %s" % (self._channels))
                continue

            # add selection
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            if self.isMC:
                weight = self.weights[ch].partial_weight(["genweight"])
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def _compute_lhe_pdf_weights(self, events):
        """Computes and returns summed LHE scale and PDF weights (used for QCD and PDF uncertainty calculations)."""

        sumlheweight = {}
        if "LHEScaleWeight" in events.fields:
            if len(events.LHEScaleWeight[0]) == 9:
                for i in range(9):
                    sumlheweight[i] = ak.sum(events.LHEScaleWeight[:, i] * events.genWeight)

        sumpdfweight = {}
        if "LHEPdfWeight" in events.fields:
            for i in range(len(events.LHEPdfWeight[0])):
                sumpdfweight[i] = ak.sum(events.LHEPdfWeight[:, i] * events.genWeight)

        return sumlheweight, sumpdfweight

    def _compute_trigger_msk(self, events):
        """Computes boolean mask for trigger."""
        nevents = len(events)
        trigger = {}
        for ch in ["ele", "mu_lowpt", "mu_highpt"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]

        trigger["ele"] = trigger["ele"] & (~trigger["mu_lowpt"]) & (~trigger["mu_highpt"])
        trigger["mu_highpt"] = trigger["mu_highpt"] & (~trigger["ele"])
        trigger["mu_lowpt"] = trigger["mu_lowpt"] & (~trigger["ele"])

        return trigger

    def _compute_metfilters_msk(self, events):
        """Computes boolean mask for METfilters."""
        nevents = len(events)
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        return metfilters

    def _build_objects(self, events):
        """Constructs physics objects (e.g. leptons, jets) and applies basic kinematic and quality cuts."""

        # MUONS
        muons = ak.with_field(events.Muon, 0, "flavor")

        # for now use 2 definitions of loose lepton and cut on the looser definition (i.e. without miso cut)
        msk_loose_muons1 = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
            & (((muons.pfRelIso04_all < 0.25) & (muons.pt < 55)) | (muons.pt >= 55))
        )
        msk_loose_muons2 = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
            & (((muons.pfRelIso04_all < 0.25) & (muons.pt < 55)) | ((muons.pt >= 55) & (muons.miniPFRelIso_all < 0.8)))
        )

        msk_tight_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | ((muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2)))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )

        # ELECTRONS
        electrons = ak.with_field(events.Electron, 1, "flavor")

        msk_loose_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WPL)
            & (((electrons.pfRelIso03_all < 0.25) & (electrons.pt < 120)) | (electrons.pt >= 120))
        )

        msk_tight_electrons = (
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

        if self._uselooselep:
            msk_good_muons = msk_loose_muons1
            msk_good_electrons = msk_loose_electrons
        else:
            msk_good_muons = msk_tight_muons
            msk_good_electrons = msk_tight_electrons

        # CANDIDATE LEP
        goodleptons = ak.concatenate([muons[msk_good_muons], electrons[msk_good_electrons]], axis=1)
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        # AK8 JETS
        fatjets = events.FatJet
        fatjets["msdcorr"] = fatjets.msoftdrop
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        good_fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True
        )

        FirstFatjet = ak.firsts(good_fatjets[:, 0:1])
        SecondFatjet = ak.firsts(good_fatjets[:, 1:2])

        # HIGGS CANDIDATE
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        jmsr_shifted_fatjetvars = get_jmsr(good_fatjets[fj_idx_lep], num_jets=1, year=self._year, isData=not self.isMC)

        candidatefj["msdcorr"] = ak.firsts(jmsr_shifted_fatjetvars["msoftdrop"][""])  # apply nominal JMS/JMR

        # AK4 JETS
        jets, jec_shifted_jetvars = get_jec_jets(events, events.Jet, self._year, not self.isMC, self.jecs, fatjets=False)
        met = met_factory.build(events.MET, jets, {}) if self.isMC else events.MET

        jet_selector = (
            (jets.pt > 15)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
            & (jets.chEmEF + jets.neEmEF < 0.9)  # neutral and charged energy fraction
        )
        jets = jets[jet_selector]
        jet_veto_map, cut_jetveto = get_JetVetoMap(jets, self._year)
        jets = jets[(jets.pt > 30) & jet_veto_map]

        # AK4 JETS (OUTSIDE AK8)
        msk_ak4_outside_ak8 = jets.delta_r(candidatefj) > 0.8
        ak4_outside_ak8 = jets[msk_ak4_outside_ak8]

        # VBF VARIABLES
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]

        # TAUS
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        loose_taus = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3)
        loose_taus = events.Tau[loose_taus]

        # b-jets (only for jets with abs(eta)<2.5)
        bjet_selector = (jets.delta_r(candidatefj) > 0.8) & (abs(jets.eta) < 2.5)

        objects = {
            "candidatefj": candidatefj,
            "candidatelep": candidatelep,
            "candidatelep_p4": candidatelep_p4,
            "met": met,
            # muons
            "muons": muons,
            "good_muons": muons[msk_good_muons],
            "tight_muons": muons[msk_tight_muons],
            "loose_muons1": muons[msk_loose_muons1],
            "loose_muons2": muons[msk_loose_muons2],
            # electrons
            "electrons": electrons,
            "good_electrons": electrons[msk_good_electrons],
            "tight_electrons": electrons[msk_tight_electrons],
            "loose_electrons": electrons[msk_loose_electrons],
            # taus
            "loose_taus": loose_taus,
            "loose_taus_mu": loose_taus_mu,
            "loose_taus_ele": loose_taus_ele,
            # jets
            "good_fatjets": good_fatjets,
            "jets": jets,
            "jet1": jet1,
            "jet2": jet2,
            "FirstFatjet": FirstFatjet,
            "SecondFatjet": SecondFatjet,
            # jecs
            "jmsr_shifted_fatjetvars": jmsr_shifted_fatjetvars,
            "jec_shifted_fatjetvars": jec_shifted_fatjetvars,
            "jec_shifted_jetvars": jec_shifted_jetvars,
            # others (easier to just store them here)
            "fj_idx_lep": fj_idx_lep,
            "msk_ak4_outside_ak8": msk_ak4_outside_ak8,
            "bjet_selector": bjet_selector,
        }

        return objects

    def _derive_variables(self, events, objects):
        """Derives physics variables from `objects` (e.g. deltaR(jet,lep))."""

        ht = ak.sum(objects["jets"].pt, axis=1)

        # VH jet
        minDeltaR = ak.argmin(objects["candidatelep_p4"].delta_r(objects["good_fatjets"]), axis=1)
        fatJetIndices = ak.local_index(objects["good_fatjets"], axis=1)
        mask_candidatefj = fatJetIndices != minDeltaR

        allScores = VScore(objects["good_fatjets"])
        VH_fj = ak.firsts(objects["good_fatjets"][allScores == ak.max(allScores[mask_candidatefj], axis=1)])

        # nleptons
        n_loose_taus_mu = ak.sum(objects["loose_taus_mu"], axis=1)
        n_loose_taus_ele = ak.sum(objects["loose_taus_ele"], axis=1)

        n_loose_muons1 = ak.num(objects["loose_muons1"], axis=1)
        n_loose_muons2 = ak.num(objects["loose_muons2"], axis=1)
        n_tight_muons = ak.num(objects["tight_muons"], axis=1)
        n_good_muons = ak.num(objects["good_muons"], axis=1)

        n_loose_electrons = ak.num(objects["loose_electrons"], axis=1)
        n_tight_electrons = ak.num(objects["tight_electrons"], axis=1)
        n_good_electrons = ak.num(objects["good_electrons"], axis=1)

        # lep isolation
        lep_reliso = (
            objects["candidatelep"].pfRelIso04_all
            if hasattr(objects["candidatelep"], "pfRelIso04_all")
            else objects["candidatelep"].pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = objects["candidatelep"].miniPFRelIso_all  # miniso for candidate lepton

        # b-jets (only for jets with abs(eta)<2.5)
        ak4_bjet_candidate = objects["jets"][objects["bjet_selector"]]

        # VBF variables
        deta = abs(ak.firsts(objects["jet1"]).eta - ak.firsts(objects["jet2"]).eta)
        mjj = (ak.firsts(objects["jet1"]) + ak.firsts(objects["jet2"])).mass

        # njets
        NumFatjets = ak.num(objects["good_fatjets"])
        NumOtherJets = ak.sum(objects["msk_ak4_outside_ak8"], axis=1)

        # n-bjets
        n_bjets_L = ak.sum(ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["L"], axis=1)
        n_bjets_M = ak.sum(ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"], axis=1)
        n_bjets_T = ak.sum(ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"], axis=1)

        # delta R between AK8 jet and lepton
        lep_fj_dr = objects["candidatefj"].delta_r(objects["candidatelep_p4"])

        # transverse mass
        mt_lep_met = np.sqrt(
            2.0
            * objects["candidatelep_p4"].pt
            * objects["met"].pt
            * (ak.ones_like(objects["met"].pt) - np.cos(objects["candidatelep_p4"].delta_phi(objects["met"])))
        )

        # delta phi MET and higgs candidate
        met_fj_dphi = objects["candidatefj"].delta_phi(objects["met"])

        # leptonic tau veto
        leptonic_taus = (objects["loose_taus"]["decayMode"] == ELE_PDGID) | (objects["loose_taus"]["decayMode"] == MU_PDGID)
        msk_leptonic_taus = ~ak.any(leptonic_taus, axis=1)

        # get the dR(genlep, recolep) to check the matching
        if self.isMC:
            genlep = events.GenPart[
                get_pid_mask(events.GenPart, [ELE_PDGID, MU_PDGID], byall=False)
                * events.GenPart.hasFlags(["fromHardProcess", "isLastCopy", "isPrompt"])
            ]

            GenLep = ak.zip(
                {
                    "pt": genlep.pt,
                    "eta": genlep.eta,
                    "phi": genlep.phi,
                    "mass": genlep.mass,
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )

            # get the dR between the recolep and the genlep that is closest to the reco lep
            dR_genlep_recolep = GenLep.delta_r(objects["candidatelep_p4"])
            genlep_idx = ak.argmin(dR_genlep_recolep, axis=1, keepdims=True)
            dR_genlep_recolep = ak.firsts(dR_genlep_recolep[genlep_idx])

            # store gen jet flavors that may be useful for studying NLO WJets
            genjets = events.GenJet
            goodgenjets = genjets[(genjets.pt > 20.0) & (np.abs(genjets.eta) < 2.4)]

            nBjets = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1)).to_numpy()
            nCjets = (ak.sum(goodgenjets.hadronFlavour == 4, axis=1)).to_numpy()

        derived_vars = {
            # candidatefj
            "fj_pt": objects["candidatefj"].pt,
            "fj_eta": objects["candidatefj"].eta,
            "fj_phi": objects["candidatefj"].phi,
            "fj_mass": objects["candidatefj"].msdcorr,
            "fj_lsf3": objects["candidatefj"].lsf3,
            "fj_VScore": VScore(objects["candidatefj"]),
            # candidatelep
            "lep_pt": objects["candidatelep"].pt,
            "lep_eta": objects["candidatelep"].eta,
            "lep_phi": objects["candidatelep"].phi,
            "lep_mass": objects["candidatelep"].mass,
            "lep_isolation": lep_reliso,
            "lep_misolation": lep_miso,
            "lep_fj_dr": lep_fj_dr,
            "lep_met_mt": mt_lep_met,
            # met
            "met_fj_dphi": met_fj_dphi,
            "met_pt": objects["met"].pt,
            "met_phi": objects["met"].phi,
            # nleptons
            "n_loose_taus_mu": n_loose_taus_mu,
            "n_loose_taus_ele": n_loose_taus_ele,
            "n_loose_muons1": n_loose_muons1,
            "n_loose_muons2": n_loose_muons2,
            "n_tight_muons": n_tight_muons,
            "n_good_muons": n_good_muons,
            "n_loose_electrons": n_loose_electrons,
            "n_tight_electrons": n_tight_electrons,
            "n_good_electrons": n_good_electrons,
            "n_loose_taus_mu": n_loose_taus_mu,
            # njets
            "NumFatjets": NumFatjets,
            "NumOtherJets": NumOtherJets,
            "n_bjets_L": n_bjets_L,
            "n_bjets_M": n_bjets_M,
            "n_bjets_T": n_bjets_T,
            # vbf variables
            "deta": deta,
            "mjj": mjj,
            # leading fatjet
            "FirstFatjet_pt": objects["FirstFatjet"].pt,
            "FirstFatjet_eta": objects["FirstFatjet"].eta,
            "FirstFatjet_phi": objects["FirstFatjet"].phi,
            "FirstFatjet_msd": objects["FirstFatjet"].msdcorr,
            "FirstFatjet_Vscore": VScore(objects["SecondFatjet"]),
            # second leading fatjet
            "SecondFatjet_pt": objects["SecondFatjet"].pt,
            "SecondFatjet_eta": objects["SecondFatjet"].eta,
            "SecondFatjet_phi": objects["SecondFatjet"].phi,
            "SecondFatjet_msd": objects["SecondFatjet"].msdcorr,
            "SecondFatjet_Vscore": VScore(objects["FirstFatjet"]),
            # second fatjet after candidate jet
            "VH_fj_pt": VH_fj.pt,
            "VH_fj_eta": VH_fj.eta,
            "VH_fj_VScore": VScore(VH_fj),
            # others
            "ht": ht,
            "loose_lep1_miso": ak.firsts(
                objects["loose_muons1"][ak.argsort(objects["loose_muons1"].pt, ascending=False)]
            ).miniPFRelIso_all,
            "loose_lep1_pt": ak.firsts(objects["loose_muons1"][ak.argsort(objects["loose_muons1"].pt, ascending=False)]).pt,
            "msk_leptonic_taus": msk_leptonic_taus,
        }

        if self.isMC:
            derived_vars.update(
                {
                    "dR_genlep_recolep": dR_genlep_recolep,
                    "ngenjets_b": nBjets,
                    "ngenjets_c": nCjets,
                }
            )

        return derived_vars

    def _apply_JEC(self, objects, variables):
        """Computes JES/JER and JMSR shift variations and appends them to the set of output variables."""

        fatjetvars = {
            "fj_pt": objects["candidatefj"].pt,
            "fj_eta": objects["candidatefj"].eta,
            "fj_phi": objects["candidatefj"].phi,
            "fj_mass": objects["candidatefj"].msdcorr,
        }

        if self._systematics and self.isMC:
            fatjetvars_sys = {}
            # JEC vars
            for shift, vals in objects["jec_shifted_fatjetvars"]["pt"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_pt{shift}"] = ak.firsts(vals[objects["fj_idx_lep"]])

            # JMSR vars
            for shift, vals in objects["jmsr_shifted_fatjetvars"]["msoftdrop"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_mass{shift}"] = ak.firsts(vals)

            variables = {**variables, **fatjetvars_sys}
            fatjetvars = {**fatjetvars, **fatjetvars_sys}

            # add variables affected by JECs/MET
            mjj_shift = {}
            for shift, vals in objects["jec_shifted_jetvars"]["pt"].items():
                if shift != "":
                    pt_1 = objects["jet1"].pt
                    pt_2 = objects["jet2"].pt
                    try:
                        pt_1 = vals[objects["msk_ak4_outside_ak8"]][:, 0]
                    except Exception:
                        pt_1 = objects["jet1"].pt
                    try:
                        pt_2 = vals[objects["msk_ak4_outside_ak8"]][:, 1]
                    except Exception:
                        pt_2 = objects["jet2"].pt

                    jet1_shift = ak.zip(
                        {
                            "pt": pt_1,
                            "eta": objects["jet1"].eta,
                            "phi": objects["jet1"].phi,
                            "mass": objects["jet1"].mass,
                            "charge": 0,
                        },
                        with_name="PtEtaPhiMCandidate",
                        behavior=candidate.behavior,
                    )
                    jet2_shift = ak.zip(
                        {
                            "pt": pt_2,
                            "eta": objects["jet2"].eta,
                            "phi": objects["jet2"].phi,
                            "mass": objects["jet2"].mass,
                            "charge": 0,
                        },
                        with_name="PtEtaPhiMCandidate",
                        behavior=candidate.behavior,
                    )
                    mjj_shift[f"mjj{shift}"] = (ak.firsts(jet1_shift) + ak.firsts(jet2_shift)).mass
            variables = {**variables, **mjj_shift}

            for met_shift in ["UES_up", "UES_down"]:
                jecvariables = getJECVariables(
                    fatjetvars, objects["candidatelep_p4"], objects["met"], pt_shift=None, met_shift=met_shift
                )
                variables = {**variables, **jecvariables}

        for shift in objects["jec_shifted_fatjetvars"]["pt"]:
            if shift != "" and not self._systematics:
                continue
            jecvariables = getJECVariables(
                fatjetvars, objects["candidatelep_p4"], objects["met"], pt_shift=shift, met_shift=None
            )
            variables = {**variables, **jecvariables}

        for shift in objects["jmsr_shifted_fatjetvars"]["msoftdrop"]:
            if shift != "" and not self._systematics:
                continue
            jmsrvariables = getJMSRVariables(fatjetvars, objects["candidatelep_p4"], objects["met"], mass_shift=shift)
            variables = {**variables, **jmsrvariables}

        return variables

    def _apply_pileup_cutoff(self, events, year, yearmod, cutoff: float = 4):
        """Apply a pileup weight cutoff on MC events.

        Events are selected only if all pileup weight variations (nominal, up, down)
        are below the specified cutoff.
        """

        pweights = get_pileup_weight(year, yearmod, events.Pileup.nPU.to_numpy())
        pw_pass = (pweights["nominal"] <= cutoff) * (pweights["up"] <= cutoff) * (pweights["down"] <= cutoff)
        logging.info(f"Passing pileup weight cut: {np.sum(pw_pass)} out of {len(events)} events")

        self.add_selection(name="PU_cutoff", sel=pw_pass)

    def _add_selections(self, events, trigger, metfilters, objects, variables):
        """Adds event pre-selection to PackedSelection for cutflow and filtering."""

        if self.isMC:
            self._apply_pileup_cutoff(events, self._year, self._yearmod, cutoff=4)

        if self._apply_trigger:
            for ch in self._channels:
                if ch == "mu":
                    self.add_selection(
                        name="Trigger",
                        sel=((objects["candidatelep"].pt < 55) & trigger["mu_lowpt"])
                        | ((objects["candidatelep"].pt >= 55) & trigger["mu_highpt"]),
                        channel=ch,
                    )
                else:
                    self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)

        if self._apply_selection:
            self.add_selection(name="METFilters", sel=metfilters)
            self.add_selection(
                name="OneLep", sel=(variables["n_good_muons"] == 1) & (variables["n_loose_electrons"] == 0), channel="mu"
            )
            self.add_selection(
                name="OneLep", sel=(variables["n_loose_muons1"] == 0) & (variables["n_good_electrons"] == 1), channel="ele"
            )
            self.add_selection(name="NoTaus", sel=(variables["n_loose_taus_mu"] == 0), channel="mu")
            self.add_selection(name="NoTaus", sel=(variables["n_loose_taus_ele"] == 0), channel="ele")
            self.add_selection(name="AtLeastOneFatJet", sel=(variables["NumFatjets"] >= 1))

            fj_pt_sel = objects["candidatefj"].pt > 250
            if self.isMC:  # make an OR of all the JECs
                for k, v in self.jecs.items():
                    for var in ["up", "down"]:
                        fj_pt_sel = fj_pt_sel | (objects["candidatefj"][v][var].pt > 250)
            self.add_selection(name="CandidateJetpT", sel=(fj_pt_sel == 1))

            self.add_selection(name="LepInJet", sel=(variables["lep_fj_dr"] < 0.8))
            self.add_selection(name="JetLepOverlap", sel=(variables["lep_fj_dr"] > 0.03))
            self.add_selection(name="dPhiJetMET", sel=(np.abs(variables["met_fj_dphi"]) < 1.57))

            if self._fakevalidation:
                self.add_selection(name="MET", sel=(objects["met"].pt < 20))
            else:
                self.add_selection(name="MET", sel=(objects["met"].pt > 20))

            # hem-cleaning selection
            if self._year == "2018":
                hem_veto = ak.any(
                    (
                        (objects["jets"].eta > -3.2)
                        & (objects["jets"].eta < -1.3)
                        & (objects["jets"].phi > -1.57)
                        & (objects["jets"].phi < -0.87)
                    ),
                    -1,
                ) | ak.any(
                    (
                        (objects["electrons"].pt > 30)
                        & (objects["electrons"].eta > -3.2)
                        & (objects["electrons"].eta < -1.3)
                        & (objects["electrons"].phi > -1.57)
                        & (objects["electrons"].phi < -0.87)
                    ),
                    -1,
                )

                hem_cleaning = (
                    ((events.run >= 319077) & (not self.isMC))  # if data check if in Runs C or D
                    # else for MC randomly cut based on lumi fraction of C&D
                    | ((np.random.rand(len(events)) < 0.632) & self.isMC)
                ) & (hem_veto)

                self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

        if not self._apply_trigger and not self._apply_selection:  # apply dummy selection
            self.add_selection(name="dummy", sel=(variables["NumFatjets"] > -1))

    def _store_genVars(self, dataset, events, objects, variables):
        """Matches generator-level particles to reconstructed jets or leptons."""

        signal_mask = None
        if self.isMC:
            if self.isSignal:
                genVars, signal_mask = match_H(events.GenPart, objects["candidatefj"], fatjet_pt=objects["FirstFatjet"])
                # genVars = {**genVars, **match_H_alljets(events.GenPart, fatjets)}
                # add signal mask and modify sum of genweights to only consider those events that pass the mask
                self.add_selection(name="Signal", sel=signal_mask)
                lhehpt = events.LHEPart[events.LHEPart.pdgId == 25].pt
                if ak.any(lhehpt):
                    genVars["LHE_Hpt"] = lhehpt
                if "HTXS" in events.fields:
                    genVars["STXS_Higgs_pt"] = events.HTXS.Higgs_pt
                    genVars["STXS_cat"] = events.HTXS.stage1_2_cat_pTjet30GeV
                    genVars["STXS_finecat"] = events.HTXS.stage1_2_fine_cat_pTjet30GeV
            elif "HToTauTau" in dataset:
                genVars, signal_mask = match_H(events.GenPart, objects["candidatefj"], dau_pdgid=15)
                self.add_selection(name="Signal", sel=signal_mask)
            elif ("WJets" in dataset) or ("ZJets" in dataset) or ("DYJets" in dataset):
                genVars, _ = match_V(events.GenPart, objects["candidatefj"])
                if "LHE_HT" in events.fields:
                    genVars["LHE_HT"] = events.LHE.HT
                if "LHE_Vpt" in events.fields:
                    genVars["LHE_Vpt"] = events.LHE.Vpt
            elif "TT" in dataset:
                genVars, _ = match_Top(events.GenPart, objects["candidatefj"])
            else:
                genVars = {}
            # save gen jet mass (not msd)
            genVars["fj_genjetmass"] = objects["candidatefj"].matched_gen.mass
            genVars["fj_genjetpt"] = objects["candidatefj"].matched_gen.pt
            variables = {**variables, **genVars}

        return variables

    def _add_MC_weights(self, dataset, events, objects, variables):
        """Applies event-level weights and corrections for MC."""

        for ch in self._channels:
            if self._year in ("2016", "2017"):
                self.weights[ch].add(
                    "L1Prefiring",
                    events.L1PreFiringWeight.Nom,
                    events.L1PreFiringWeight.Up,
                    events.L1PreFiringWeight.Dn,
                )
            add_pileup_weight(
                self.weights[ch],
                self._year,
                self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )

            add_pileupid_weights(self.weights[ch], self._year, self._yearmod, objects["jets"], events.GenJet, wp="L")

            if ch == "mu":
                add_lepton_weight(self.weights[ch], objects["candidatelep"], self._year + self._yearmod, "muon")
            elif ch == "ele":
                add_lepton_weight(self.weights[ch], objects["candidatelep"], self._year + self._yearmod, "electron")

            ewk_corr, qcd_corr, alt_qcd_corr = add_VJets_kFactors(self.weights[ch], events.GenPart, dataset, events)
            # add corrections for plotting
            variables["weight_ewkcorr"] = ewk_corr
            variables["weight_qcdcorr"] = qcd_corr
            variables["weight_altqcdcorr"] = alt_qcd_corr["nominal"]
            variables["weight_altqcdcorr_up"] = alt_qcd_corr["up"]
            variables["weight_altqcdcorr_down"] = alt_qcd_corr["down"]

            # add top_reweighting
            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
            if "TT" in dataset:
                tops = events.GenPart[get_pid_mask(events.GenPart, 6, byall=False) * events.GenPart.hasFlags(["isLastCopy"])]

                # will also save it as a variable just in case
                variables["top_reweighting"] = add_TopPtReweighting(self.weights[ch], tops.pt)

            if self.isSignal:
                ew_weight = add_HiggsEW_kFactors(events.GenPart, dataset)
                # save EW weights but do not apply by default
                variables["EW_weight"] = ew_weight

            if self._systematics:
                if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
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
                    scale_weights = {}
                    if "LHEScaleWeight" in events.fields:
                        # save individual weights
                        if len(events.LHEScaleWeight[0]) == 9:
                            for i in [0, 1, 3, 5, 7, 8, 4]:
                                scale_weights[f"weight_scale{i}"] = events.LHEScaleWeight[:, i]
                    variables = {**variables, **scale_weights}

                if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values
                    - The last two values: alpha_s variations.
                    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
                    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
                    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
                    """
                    pdf_weights = {}
                    if "LHEPdfWeight" in events.fields:
                        # save individual weights
                        for i in range(len(events.LHEPdfWeight[0])):
                            pdf_weights[f"weight_pdf{i}"] = events.LHEPdfWeight[:, i]
                    variables = {**variables, **pdf_weights}

            if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
                add_ps_weight(
                    self.weights[ch],
                    events.PSWeight if "PSWeight" in events.fields else [],
                )

            # store the final weight per ch
            variables[f"weight_{ch}"] = self.weights[ch].weight()

            if self._systematics:
                for systematic in self.weights[ch].variations:
                    variables[f"weight_{ch}_{systematic}"] = self.weights[ch].weight(modifier=systematic)

            # store b-tag weight
            for wp_ in ["T"]:
                variables = {
                    **variables,
                    **get_btag_weights(
                        self._year,
                        objects["jets"],
                        objects["bjet_selector"],
                        wp=wp_,
                        algo="deepJet",
                        systematics=self._systematics,
                    ),
                }
        return variables

    def _add_lundplane_weights(self, events, dataset, selection_ch, objects):
        """Computes variables used for Lund plane reweighting technique."""

        from boostedhiggs.corrections import getLPweights

        pf_cands, gen_parts_eta_phi, gen_parts_pt_mass, ak8_jets, bgen_parts_eta_phi, genlep = getLPweights(
            dataset,
            events[selection_ch],
            objects["candidatefj"][selection_ch],
            objects["fj_idx_lep"][selection_ch],
            objects["candidatelep_p4"][selection_ch],
        )
        lpvars = {}
        for pfcandidx in range(pf_cands.shape[1]):
            lpvars[f"LP_pfcand{pfcandidx}_px"] = pf_cands[:, pfcandidx, 0]
            lpvars[f"LP_pfcand{pfcandidx}_py"] = pf_cands[:, pfcandidx, 1]
            lpvars[f"LP_pfcand{pfcandidx}_pz"] = pf_cands[:, pfcandidx, 2]
            lpvars[f"LP_pfcand{pfcandidx}_energy"] = pf_cands[:, pfcandidx, 3]
        for quarkidx in range(gen_parts_eta_phi.shape[1]):
            lpvars[f"LP_quark{quarkidx}_eta"] = gen_parts_eta_phi[:, quarkidx, 0]
            lpvars[f"LP_quark{quarkidx}_phi"] = gen_parts_eta_phi[:, quarkidx, 1]
            lpvars[f"LP_quark{quarkidx}_pt"] = gen_parts_pt_mass[:, quarkidx, 0]
            lpvars[f"LP_quark{quarkidx}_mass"] = gen_parts_pt_mass[:, quarkidx, 1]
        if bgen_parts_eta_phi is not None:
            for quarkidx in range(bgen_parts_eta_phi.shape[1]):
                lpvars[f"LP_bquark{quarkidx}_eta"] = bgen_parts_eta_phi[:, quarkidx, 0]
                lpvars[f"LP_bquark{quarkidx}_phi"] = bgen_parts_eta_phi[:, quarkidx, 1]
        lpvars["LP_genlep_pt"] = genlep[:, 0]
        lpvars["LP_genlep_eta"] = genlep[:, 1]
        lpvars["LP_genlep_phi"] = genlep[:, 2]
        lpvars["LP_genlep_mass"] = genlep[:, 3]
        lpvars["LP_fj_pt"] = ak8_jets[:, 0]
        lpvars["LP_fj_eta"] = ak8_jets[:, 1]
        lpvars["LP_fj_phi"] = ak8_jets[:, 2]
        lpvars["LP_fj_mass"] = ak8_jets[:, 3]
        return lpvars

    def _run_inference(self, events, selection_ch, objects):
        """Runs ParT tagger inference using Triton."""

        model_name = "ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"
        pnet_vars = runInferenceTriton(
            self.tagger_resources_path, events[selection_ch], objects["fj_idx_lep"][selection_ch], model_name=model_name
        )
        pnet_df = self._ak_to_pandas(pnet_vars)
        scores = {"fj_ParT_score": pnet_df[sigs].sum(axis=1).values}
        hidNeurons = {k: v for k, v in pnet_vars.items() if "hidNeuron" in k}
        reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}

        return {**scores, **reg_mass, **hidNeurons}

    def _build_output(self, events, dataset, variables, objects):
        """Stores the events passing the selections in a pandas dataframes."""

        output = {}
        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)
            fill_output = True
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {var: item for var, item in variables.items()}

                output[ch] = {key: value[selection_ch] for key, value in out.items()}

                if self._getLPweights:
                    output[ch] = {**output[ch], **self._add_lundplane_weights(events, dataset, selection_ch, objects)}

                if self._inference:
                    output[ch] = {**output[ch], **self._run_inference(events, selection_ch, objects)}
            else:
                output[ch] = {}

            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self._ak_to_pandas(output[ch])

            for var_ in [
                "rec_higgs_m",
                "rec_higgs_pt",
                "rec_W_qq_m",
                "rec_W_qq_pt",
                "rec_W_lnu_m",
                "rec_W_lnu_pt",
            ]:
                if var_ in output[ch].keys():
                    output[ch][var_] = np.nan_to_num(output[ch][var_], nan=-1)

        return output

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts as pandas dataframe."""

        dataset = events.metadata["dataset"]
        nevents = len(events)

        self.isMC = hasattr(events, "genWeight")
        self.isSignal = any(s in dataset for s in ["HToWW", "ttHToNonbb"])

        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents
        sumlheweight, sumpdfweight = self._compute_lhe_pdf_weights(events) if self.isMC else ({}, {})

        if self.isMC:  # add genweight before filling cutflow
            for ch in self._channels:
                self.weights[ch].add("genweight", events.genWeight)

        trigger = self._compute_trigger_msk(events)
        metfilters = self._compute_metfilters_msk(events)

        objects = self._build_objects(events)
        variables = self._derive_variables(events, objects)

        # store the genweight as a column
        for ch in self._channels:
            variables[f"weight_{ch}_genweight"] = self.weights[ch].partial_weight(["genweight"])

        variables = self._apply_JEC(objects, variables)

        self._add_selections(events, trigger, metfilters, objects, variables)

        variables = self._store_genVars(dataset, events, objects, variables)

        if self.isMC:
            variables = self._add_MC_weights(dataset, events, objects, variables)

        output = self._build_output(events, dataset, variables, objects)

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self._save_dfs_parquet(fname, output[ch], ch)

        return {
            dataset: {
                "mc": self.isMC,
                self._year
                + self._yearmod: {
                    "sumgenweight": sumgenweight,
                    "sumlheweight": sumlheweight,
                    "sumpdfweight": sumpdfweight,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        """Placeholder for coffea postprocessing (currently does nothing)."""
        return accumulator
