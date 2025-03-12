import logging
import os
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
    add_VJets_kFactors,
)
from boostedhiggs.utils import match_V

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


class VjetsProcessor(processor.ProcessorABC):
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
        self._channels = channels
        self._systematics = systematics
        self._getLPweights = getLPweights
        self._uselooselep = uselooselep
        self._fakevalidation = fakevalidation

        self._apply_trigger = not no_trigger
        self._apply_selection = not no_selection

        self._output_location = output_location

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
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

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]

        self.isMC = hasattr(events, "genWeight")
        self.isSignal = True if ("HToWW" in dataset) or ("ttHToNonbb" in dataset) else False

        nevents = len(events)
        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents

        # sum LHE weight
        sumlheweight = {}
        if "LHEScaleWeight" in events.fields and self.isMC:
            if len(events.LHEScaleWeight[0]) == 9:
                for i in range(len(events.LHEScaleWeight[0])):
                    sumlheweight[i] = ak.sum(events.LHEScaleWeight[:, i] * events.genWeight)

        # sum PDF weight
        sumpdfweight = {}
        if "LHEPdfWeight" in events.fields and self.isMC:
            for i in range(len(events.LHEPdfWeight[0])):
                sumpdfweight[i] = ak.sum(events.LHEPdfWeight[:, i] * events.genWeight)

        # add genweight before filling cutflow
        if self.isMC:
            for ch in self._channels:
                self.weights[ch].add("genweight", events.genWeight)

        genht = ak.sum(events.GenJet.pt, axis=1)

        # store gen jet flavors that may be useful for studying NLO WJets
        genjets = events.GenJet
        goodgenjets = genjets[(genjets.pt > 20.0) & (np.abs(genjets.eta) < 2.4)]

        nBjets = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1)).to_numpy()
        nCjets = (ak.sum(goodgenjets.hadronFlavour == 4, axis=1)).to_numpy()

        variables = {
            "nBjets": nBjets,
            "nCjets": nCjets,
        }

        # store the genweight as a column
        for ch in self._channels:
            variables[f"weight_{ch}_genweight"] = self.weights[ch].partial_weight(["genweight"])

        # store additional relevant jet variables
        muons = ak.with_field(events.Muon, 0, "flavor")
        tight_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | ((muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2)))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )
        good_muons = tight_muons
        electrons = ak.with_field(events.Electron, 1, "flavor")
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
        good_electrons = tight_electrons
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt
        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton
        fatjets = events.FatJet
        fatjets["msdcorr"] = fatjets.msoftdrop
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        fatjetvars = {
            "fj_pt": candidatefj.pt,
            "fj_eta": candidatefj.eta,
            "fj_phi": candidatefj.phi,
            "fj_mass": candidatefj.msdcorr,
        }

        variables = {**variables, **fatjetvars}

        # store gen-level matching variables
        if self.isMC:
            if ("WJets" in dataset) or ("ZJets" in dataset) or ("DYJets" in dataset):
                genVars, _ = match_V(events.GenPart, candidatefj)
                genVars["LHE_HT"] = events.LHE.HT
                genVars["LHE_Vpt"] = events.LHE.Vpt
            else:
                genVars = {}
            # save gen jet mass (not msd)
            # genVars["fj_genjetmass"] = candidatefj.matched_gen.mass
            # genVars["fj_genjetpt"] = candidatefj.matched_gen.pt
            variables = {**variables, **genVars}

        # apply dummy selection
        self.add_selection(name="dummy", sel=(genht > -99999))

        if self.isMC:
            for ch in self._channels:
                ewk_corr, qcd_corr, alt_qcd_corr = add_VJets_kFactors(self.weights[ch], events.GenPart, dataset, events)
                # add corrections for plotting
                variables["weight_ewkcorr"] = ewk_corr
                variables["weight_qcdcorr"] = qcd_corr
                # variables["weight_altqcdcorr"] = alt_qcd_corr["nominal"]
                # variables["weight_altqcdcorr_up"] = alt_qcd_corr["up"]
                # variables["weight_altqcdcorr_down"] = alt_qcd_corr["down"]

                # store the final weight per ch
                variables[f"weight_{ch}"] = self.weights[ch].weight()

                if self._systematics:
                    for systematic in self.weights[ch].variations:
                        variables[f"weight_{ch}_{systematic}"] = self.weights[ch].weight(modifier=systematic)

        ###############################
        # Initialize pandas dataframe
        ###############################

        output = {}
        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {}
                for var, item in variables.items():
                    # pad all the variables that are not a cut with -1
                    # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    # fill out dictionary
                    out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
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
        return accumulator
