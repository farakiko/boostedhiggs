"""
Creates "combine datacards" using hist.Hist templates, and
sets up data-driven WJets+QCD background estimate ('rhalphabet' method)

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
"""

from __future__ import division, print_function

import argparse
import logging
import math
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rhalphalib as rl
from combine.datacard_systematics import systs_from_parquets, systs_not_from_parquets
from utils_diffBins import get_template, labels, load_templates, samples, shape_to_num, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


@dataclass
class ShapeVar:

    """For storing and calculating info about variables used in fit"""

    name: str = None
    bins: np.ndarray = None  # bin edges
    order: int = None  # TF order

    def __post_init__(self):
        # use bin centers for polynomial fit
        self.pts = self.bins[:-1] + 0.5 * np.diff(self.bins)
        # scale to be between [0, 1]
        self.scaled = (self.pts - self.bins[0]) / (self.bins[-1] - self.bins[0])


def create_datacard(hists_templates, years, lep_channels, do_rhalphabet, order):
    # define the systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, lep_channels)
    sys_from_parquets = systs_from_parquets(years)

    # define the model
    model = rl.Model("testModel")

    # define the signal and control regions
    sig_regions = ["SRVBF97", "SRggFpt250to300", "SRggFpt300to450", "SRggFpt450toInf"]
    regions = sig_regions + ["WJetsCR"] + ["TopCR"]

    # fill datacard with systematics and rates
    for ChName in regions:
        if do_rhalphabet:  # use MC wjets+qcd if do_rhalphabet=False
            Samples = samples.copy()
            Samples.remove("WJetsLNu")
            Samples.remove("QCD")
        else:
            Samples = samples.copy()

        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in Samples:
            templ = get_template(hists_templates, sName, ChName)
            stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

            sample.autoMCStats(lnN=True)

            # SYSTEMATICS NOT FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in systs_dict[syst_on_sample].items():
                    if systs_dict_values[syst_on_sample][sys_name][1] is None:  # if up and down are the same
                        sample.setParamEffect(sys_value, systs_dict_values[syst_on_sample][sys_name][0])
                    else:
                        sample.setParamEffect(
                            sys_value,
                            systs_dict_values[syst_on_sample][sys_name][0],
                            systs_dict_values[syst_on_sample][sys_name][1],
                        )

            # SYSTEMATICS FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in sys_from_parquets[syst_on_sample].items():
                    syst_up = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": sys_name + "Up"}].values()
                    syst_do = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": sys_name + "Down"}].values()
                    nominal = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": "nominal"}].values()

                    if sys_value.combinePrior == "lnN":
                        eff_up = shape_to_num(syst_up, nominal)
                        eff_do = shape_to_num(syst_do, nominal)

                        if math.isclose(eff_up, eff_do, rel_tol=1e-2):  # if up and down are the same
                            sample.setParamEffect(sys_value, max(eff_up, eff_do))
                        else:
                            sample.setParamEffect(sys_value, max(eff_up, eff_do), min(eff_up, eff_do))

                    else:
                        sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))

            ch.addSample(sample)

        # add data
        data_obs = get_template(hists_templates, "Data", ChName)
        ch.setObservation(data_obs)

    if do_rhalphabet:  # data-driven estimation
        rhalphabet(model, hists_templates, order, passChNames=sig_regions, failChName="WJetsCR")

    return model


def rhalphabet(model, hists_templates, order, passChNames, failChName):
    shape_var = ShapeVar(
        name=hists_templates.axes["mass_observable"].name, bins=hists_templates.axes["mass_observable"].edges, order=order
    )
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    # wjets params
    wjets_params = np.array(
        [rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0) for i in range(m_obs.nbins)]
    )

    failCh = model[failChName]

    initial_wjets = failCh.getObservation().astype(float)
    if np.any(initial_wjets < 0.0):
        logging.warning(f"initial_wjets negative for some bins... {initial_wjets}")
        initial_wjets[initial_wjets < 0] = 0

    for sample in failCh:
        if sample.sampletype == rl.Sample.SIGNAL:
            continue
        # logging.info(f"subtracting {sample._name} from wjets+qcd")
        initial_wjets -= sample.getExpectation(nominal=True)

    # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^wjetsparams
    # will result in wjetsparams errors ~±1
    # but because wjets is poorly modelled we're scaling sigma scale

    sigmascale = 10  # to scale the deviation from initial
    scaled_params = initial_wjets * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_wjets))) ** wjets_params

    fail_wjets = rl.ParametericSample(
        f"{failChName}_{CMS_PARAMS_LABEL}_wjets_datadriven", rl.Sample.BACKGROUND, m_obs, scaled_params
    )
    failCh.addSample(fail_wjets)

    for passChName in passChNames:
        logging.info(f"setting transfer factor for region {passChName}, from region {failChName}")

        # define a seperate transfer factor per passChName
        tf_dataResidual = rl.BasisPoly(
            f"{CMS_PARAMS_LABEL}_tf_dataResidual_{passChName}",
            (shape_var.order,),
            [shape_var.name],
            basis="Bernstein",
            limits=(-20, 20),
            # square_params=True,   # TODO: figure why this can't be uncommented
        )

        den = hists_templates[{"Region": failChName, "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}].sum().value
        num = hists_templates[{"Region": passChName, "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}].sum().value

        # get the transfer factor
        wjets_eff = num / den

        tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
        tf_params_pass = wjets_eff * tf_dataResidual_params  # scale params initially by wjets eff

        passCh = model[passChName]
        pass_wjets = rl.TransferFactorSample(
            f"{passChName}_{CMS_PARAMS_LABEL}_wjets_datadriven", rl.Sample.BACKGROUND, tf_params_pass, fail_wjets
        )
        passCh.addSample(pass_wjets)


def main(args):
    years = args.years.split(",")
    lep_channels = args.channels.split(",")

    hists_templates = load_templates(years, lep_channels, args.outdir)

    model = create_datacard(hists_templates, years, lep_channels, do_rhalphabet=args.rhalphabet, order=args.order)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1 --rhalphabet --order 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--rhalphabet", action="store_true", help="if provided will run rhalphabet with default order=2")
    parser.add_argument("--order", default=2, type=int, help="bernstein polynomial order when running with --rhalphabet")
    parser.add_argument("--outdir", default="templates/test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
