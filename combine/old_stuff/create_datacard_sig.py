"""
Creates combine datacards using hist.Hist templates specifically to run studies on the asmimov significance.

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
"""

from __future__ import division, print_function

import argparse
import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rhalphalib as rl
from combine.datacard_systematics import systs_not_from_parquets
from utils_diffBins import get_template, labels, load_templates, samples, sigs

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


def create_datacard(hists_templates, years, lep_channels):
    # fill the cards with only some rudimentary rate systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, lep_channels)

    regions = list(hists_templates.axes["Region"])
    model = rl.Model("testModel")

    # fill datacard with systematics and rates
    Samples = samples.copy()
    for region in regions:
        h = hists_templates.copy()

        ChName = f"{region}"

        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in Samples:
            templ = get_template(h, sName, region)
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
            ch.addSample(sample)

        # add data
        data_obs = get_template(h, "Data", region)
        ch.setObservation(data_obs)

    return model


def main(args):
    years = args.years.split(",")
    lep_channels = args.channels.split(",")

    hists_templates = load_templates(years, lep_channels, args.outdir)

    model = create_datacard(hists_templates, years, lep_channels)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard_sig.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="name of template directory")
    args = parser.parse_args()

    main(args)
