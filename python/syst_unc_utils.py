import hist as hist2
import numpy as np
import utils


def initialize_syst_unc_hists(SYST_DICT, plot_config):
    """Uses the script systematics.py in the `combine` directory to retrieve the dictionnary of systematics `SYST_DICT`.

    Insantiates histograms,
        nominal
        up -> for each variation
        down -> for each variation
    """

    SYST_hists = {}

    for var_to_plot in plot_config["vars_to_plot"]:

        SYST_hists[var_to_plot] = {}

        # instantiate the nominal hist
        SYST_hists[var_to_plot]["nominal"] = hist2.Hist(
            utils.get_axis(var_to_plot, plot_config["massbin"]),
            storage=hist2.storage.Weight(),
        )

        # instantiate the up/down hist
        SYST_hists[var_to_plot]["up"], SYST_hists[var_to_plot]["down"] = {}, {}

        for syst in {**SYST_DICT["common"], **SYST_DICT["btag"], **SYST_DICT["LP"]}:

            # if "pileup" in syst:
            #     continue

            SYST_hists[var_to_plot]["up"][syst] = hist2.Hist(
                utils.get_axis(var_to_plot, plot_config["massbin"]),
                storage=hist2.storage.Weight(),
            )
            SYST_hists[var_to_plot]["down"][syst] = hist2.Hist(
                utils.get_axis(var_to_plot, plot_config["massbin"]),
                storage=hist2.storage.Weight(),
            )

        if (var_to_plot == "fj_pt") or (var_to_plot == "rec_higgs_m") or (var_to_plot == "rec_higgs_pt"):
            for syst in SYST_DICT["JEC"]:
                if (var_to_plot != "rec_higgs_m") and (("JMR" in syst) or ("JMS" in syst) or ("UES" in syst)):
                    continue

                SYST_hists[var_to_plot]["up"][syst] = hist2.Hist(
                    utils.get_axis(var_to_plot, plot_config["massbin"]),
                    storage=hist2.storage.Weight(),
                )
                SYST_hists[var_to_plot]["down"][syst] = hist2.Hist(
                    utils.get_axis(var_to_plot, plot_config["massbin"]),
                    storage=hist2.storage.Weight(),
                )

    return SYST_hists


def fill_syst_unc_hists(SYST_DICT, SYST_hists, year, ch, sample, var_to_plot, df):

    # only get systematic uncertainty on MC samples
    if ("Fake" in sample) or ("Data" in sample) or (sample in utils.signals):
        return SYST_hists

    # get the nominal
    SYST_hists[var_to_plot]["nominal"].fill(
        var=df[var_to_plot],
        weight=df["nominal"],
    )

    # get the up/down
    for syst, (yrs, smpls, var) in SYST_DICT["common"].items():

        if (sample in smpls) and (year in yrs) and (ch in var):
            up = df["xsecweight"] * df[var[ch] + "Up"]
            down = df["xsecweight"] * df[var[ch] + "Down"]

            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=up,
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=down,
            )
        else:
            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )

    for syst, (yrs, smpls, var) in SYST_DICT["btag"].items():

        if (sample in smpls) and (year in yrs) and (ch in var):
            up = df["nominal"] * df[var[ch] + "Up"]
            down = df["nominal"] * df[var[ch] + "Down"]

            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=up,
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=down,
            )
        else:
            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )

    for syst, (yrs, smpls, var) in SYST_DICT["LP"].items():
        if (sample in smpls) and (year in yrs) and (ch in var):
            up = df[var[ch] + "_up"]
            down = df[var[ch] + "_down"]

            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=up,
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=down,
            )
        else:
            SYST_hists[var_to_plot]["up"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )
            SYST_hists[var_to_plot]["down"][syst].fill(
                var=df[var_to_plot],
                weight=df["nominal"],
            )

    if (var_to_plot == "fj_pt") or (var_to_plot == "rec_higgs_m") or (var_to_plot == "rec_higgs_pt"):
        for syst, (yrs, smpls, var) in SYST_DICT["JEC"].items():

            if (var_to_plot != "rec_higgs_m") and (("JMR" in syst) or ("JMS" in syst) or ("UES" in syst)):
                continue

            if (sample in smpls) and (year in yrs) and (ch in var):
                shape_up = df[var_to_plot + var[ch] + "_up"]
                shape_down = df[var_to_plot + var[ch] + "_down"]

                SYST_hists[var_to_plot]["up"][syst].fill(
                    var=shape_up,
                    weight=df["nominal"],
                )
                SYST_hists[var_to_plot]["down"][syst].fill(
                    var=shape_down,
                    weight=df["nominal"],
                )
            else:
                SYST_hists[var_to_plot]["up"][syst].fill(
                    var=df[var_to_plot],
                    weight=df["nominal"],
                )
                SYST_hists[var_to_plot]["down"][syst].fill(
                    var=df[var_to_plot],
                    weight=df["nominal"],
                )

    return SYST_hists


def get_total_syst_unc(H):

    # initialize as 0%
    total_unc_up = H["nominal"].values() * 0

    for sys in H["up"]:

        up = H["up"][sys].values()
        nom = H["nominal"].values()

        total_unc_up += (((up / nom) - 1)) ** 2

    # initialize as 0%
    total_unc_down = H["nominal"].values() * 0

    for sys in H["down"]:

        down = H["down"][sys].values()
        nom = H["nominal"].values()

        total_unc_down += (((down / nom) - 1)) ** 2

    return np.sqrt(total_unc_up), np.sqrt(total_unc_down)
