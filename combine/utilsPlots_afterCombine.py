#!/usr/bin/python

import json
import os
import warnings

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")


# PLOTTING UTILS
# color_by_sample = {
#     "ggF": "pink",
#     "VBF": "aqua",
#     "WH": "green",
#     "ZH": "blue",
#     "ttH": "yellow",
#     # background
#     "QCD": "#9c9ca1",
#     "WJetsLNu": "#f89c20",
#     "TTbar": "#e42536",
#     "Diboson": "orchid",
#     "SingleTop": "#964a8b",
#     "EWKvjets": "tab:grey",
#     "DYJets": "tab:purple",
#     "WZQQ": "khaki",
#     "Fake": "#9c9ca1",
#     "Rest": "#5790fc",
#     ###################################
#     # stxs
#     "ggH_hww_200_300": "lightsteelblue",
#     "ggH_hww_300_450": "tab:olive",
#     "ggH_hww_450_Inf": "tab:brown",
#     "qqH_hww_mjj_1000_Inf": "peru",
# }
color_by_sample = {
    "ggF": "lightsteelblue",
    "VBF": "peru",
    # signal that is background
    "WH": "tab:brown",
    "ZH": "yellowgreen",
    "ttH": "tab:olive",
    # background
    "QCD": "tab:orange",
    "Fake": "tab:orange",
    "WJetsLNu": "tab:green",
    "TTbar": "tab:blue",
    "Diboson": "orchid",
    "SingleTop": "tab:cyan",
    # "WJetsLNu_unmatched": "tab:grey",
    # "WJetsLNu_matched": "tab:green",
    "EWKvjets": "tab:grey",
    # TODO: make sure it's WZQQ is NLO in next iteration
    "DYJets": "tab:purple",
    "WZQQ": "khaki",
    "WZQQorDYJets": "khaki",
    "Rest": "#5790fc",
}

plot_labels = {
    "ggF": "ggF",
    "VBF": "VBF",
    "WH": "WH",
    "ZH": "ZH",
    "ttH": "ttH",
    # bkg
    "WJetsLNu": "WJets",
    "TTbar": "TT",
    "SingleTop": "ST",
    "Rest": "Rest",
    # stxs
    "ggH_hww_200_300": r"ggF $pT^{H}: 200-300$",
    "ggH_hww_300_450": r"ggF $pT^{H}: 300-450$",
    "ggH_hww_450_Inf": r"ggF $pT^{H}: 450-Inf$",
    "qqH_hww_mjj_1000_Inf": r"VBF $m_{jj}: 1000-Inf$",
}

label_by_ch = {"mu": "Muon", "ele": "Electron"}


def get_lumi(years, channels):
    luminosity = 0
    for year in years:
        lum = 0
        for ch in channels:
            with open("../fileset/luminosity.json") as f:
                lum += json.load(f)[ch][year] / 1000.0

        luminosity += lum / len(channels)
    return luminosity


signals = ["VBF", "ggF", "ttH", "WH", "ZH"]
signals += ["ggH_hww_200_300", "ggH_hww_300_450", "ggH_hww_450_Inf", "qqH_hww_mjj_1000_Inf"]


def plot_hists(
    h,
    years,
    channels,
    mult,
    out_dir,
    out_fname=None,
    ax_plot_sig=True,
    ax_plot_bkg=True,
    ax_plot_tot_sig=False,
    ax_plot_tot_bkg_sig=False,
    rax_plot_sig=True,
    rax_plot_bkg=True,
    rax_plot_tot_sig=True,
    legend_title="",
    blind_region=None,
    label_on_plot="",
    remove_samples=[],
    ratio_plot="Data-MC",
    use_postfit_errors=False,
    postfit_errors_mc=None,
    postfit_errors_data=None,
    logy=False,
):

    # get samples existing in histogram
    samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

    for s in remove_samples:
        if s in samples:
            samples.remove(s)

    # get data
    data = h[{"Sample": "Data"}]

    # get signal
    signal_labels = [label for label in samples if label in signals]
    signal = [h[{"Sample": label}] for label in signal_labels]
    signal_mult = [s * mult for s in signal]

    tot_signal = None
    for i, sig in enumerate(signal_mult):
        if tot_signal is None:
            tot_signal = signal[i].copy()
        else:
            tot_signal = tot_signal + signal[i]
    tot_signal_mult = tot_signal * mult

    # get bkg
    # bkg_labels = [label for label in samples if (label and label not in signal_labels and (label not in ["Data"]))]
    bkg = []
    bkg_labels = []
    for label in samples:
        if label and label not in signal_labels and (label not in ["Data"]):

            if label in ["WJetsLNu", "TTbar"]:
                bkg_labels += [label]
                bkg += [h[{"Sample": label}]]

    bkg_rest_sum = 0
    for label in samples:
        if label and label not in signal_labels and (label not in ["Data"]):
            if label not in ["WJetsLNu", "TTbar"]:

                # Let's say your histogram is called `hf`
                sample_axis = h.axes[0]  # Assuming Sample is the first axis
                bkg_idx = sample_axis.index(label)

                bkg_rest_sum += h[bkg_idx, :]
    bkg_labels += ["Rest"]
    bkg += [bkg_rest_sum]

    # sum all of the background
    if len(bkg) > 0:

        tot_val_MC = np.array(bkg).sum(axis=0)
        tot_val_MC_zero_mask = tot_val_MC == 0  # check if this is for the ratio or not
        tot_val_MC[tot_val_MC_zero_mask] = 1

        if use_postfit_errors is True:
            tot_err_MC = postfit_errors_mc
        else:
            tot_err_MC = np.sqrt(tot_val_MC)

    # setup the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 9),
        gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
        sharex=True,
    )

    errps = {
        "hatch": "////",
        "facecolor": "none",
        "lw": 0,
        "color": "k",
        "edgecolor": (0, 0, 0, 0.5),
        "linewidth": 0,
        "alpha": 0.4,
    }

    ##########################
    # ax start

    # plot the data
    data_err_opts = {
        "linestyle": "none",
        "marker": ".",
        "markersize": 10.0,
        "elinewidth": 1,
    }

    if blind_region:
        massbins = data.axes[-1].edges
        lv = int(np.searchsorted(massbins, blind_region[0], "right"))
        rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)

        data.view(flow=True)[lv:rv].value = 0
        data.view(flow=True)[lv:rv].variance = 0

    hep.histplot(
        data,
        ax=ax,
        histtype="errorbar",
        color="k",
        capsize=4,
        yerr=True,
        label="Data",
        **data_err_opts,
        flow="none",
    )

    # plot the background
    if len(bkg) > 0 and ax_plot_bkg:
        hep.histplot(
            bkg,
            ax=ax,
            stack=True,
            sort="yield",
            edgecolor="black",
            linewidth=1,
            histtype="fill",
            label=[plot_labels[bkg_label] for bkg_label in bkg_labels],
            color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
            flow="none",
        )

        ax.stairs(
            values=tot_val_MC + tot_err_MC,
            baseline=tot_val_MC - tot_err_MC,
            edges=bkg[0].copy().axes[0].edges,
            **errps,
            label="Syst. Unc.",
        )

    # plot the signal
    if len(signal) > 0 and ax_plot_sig:
        for i, sig in enumerate(signal_mult):
            if mult == 1:
                lab_sig_mult = f"{plot_labels[signal_labels[i]]}"
            else:
                lab_sig_mult = f"{mult} * {plot_labels[signal_labels[i]]}"

            hep.histplot(
                sig,
                ax=ax,
                label=lab_sig_mult,
                linewidth=3,
                color=color_by_sample[signal_labels[i]],
                flow="none",
            )

    # plot total sig
    if len(signal) > 0 and ax_plot_tot_sig:

        if mult == 1:
            siglabel = r"Total Signal"
        else:
            siglabel = r"Total Signal $\times$" + f"{mult}"

        hep.histplot(
            tot_signal_mult,
            ax=ax,
            label=siglabel,
            linewidth=2,
            color="black",
            flow="none",
        )
        # # add MC stat errors
        # ax.stairs(
        #     values=tot_signal_mult.values() + np.sqrt(tot_signal_mult.values()),
        #     baseline=tot_signal_mult.values() - np.sqrt(tot_signal_mult.values()),
        #     edges=sig.axes[0].edges,
        #     **errps,
        # )

    # plot total sig+bkg
    if len(signal) > 0 and ax_plot_tot_bkg_sig:

        sig_plus_bkg = tot_signal + np.array(bkg).sum(axis=0)
        hep.histplot(
            sig_plus_bkg,
            ax=ax,
            label=r"Background + Signal",
            linewidth=2,
            color="tab:red",
            flow="none",
        )
        # add MC stat errors
        ax.stairs(
            values=sig_plus_bkg.values() + np.sqrt(sig_plus_bkg.values()),
            baseline=sig_plus_bkg.values() - np.sqrt(sig_plus_bkg.values()),
            edges=sig.axes[0].edges,
            **errps,
        )

    ax.set_ylabel("Events")
    ax.set_xlabel("")

    # ax end
    ##########################

    ##########################
    # rax start

    if ratio_plot == "Pulls":
        rax.axhline(0, ls="--", color="k")
        rax.set_ylim(-6, 6)
        # rax.set_ylim(-2.6, 2.6)
        rax.set_ylabel(r"Pull: $\frac{Data-MC}{\sigma_{stat}}$", fontsize=18, labelpad=10)
    elif ratio_plot == "Data/MC":
        rax.axhline(1, ls="--", color="k")
        rax.set_ylim(0.2, 1.8)
        rax.set_ylabel("Data/MC", fontsize=20, labelpad=10)
    elif ratio_plot == "Data-MC":
        rax.axhline(0, ls="--", color="k")
        rax.set_ylim(-15, 15)
        # rax.set_ylim(-120, 120)
        rax.set_ylabel("Data-MC", fontsize=20, labelpad=10)

    if len(bkg) > 0 and rax_plot_bkg:

        if ratio_plot == "Pulls":

            data_val = data.values()
            data_val[tot_val_MC_zero_mask] = 1

            if use_postfit_errors is True:
                sigma_data = postfit_errors_data
            else:
                from scipy.stats import chi2

                def garwood_interval(n, cl=0.68):
                    """Calculate Garwood's 68% CL asymmetric confidence interval for binomial proportion."""
                    alpha = 1 - cl
                    lower = chi2.ppf(alpha / 2, 2 * n) / 2 if n > 0 else 0
                    upper = chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2
                    return (lower, upper)

                # Calculate uncertainties using Garwood interval
                data_uncertainties = np.array([garwood_interval(n) for n in data_val])
                data_errors = np.vstack(data_uncertainties).T
                data_errors[0] = data_val - data_errors[0]
                data_errors[1] = data_errors[1] - data_val

                sigma_data = np.sqrt(data_errors.mean(axis=0))

            pulls = (data_val - tot_val_MC) / sigma_data

            hep.histplot(
                pulls,
                bkg[0].copy().axes[0].edges,
                yerr=1,
                ax=rax,
                histtype="errorbar",
                color="k",
                capsize=4,
                flow="none",
                label="Pull",
            )

            rax.stairs(
                values=0 + tot_err_MC / sigma_data,
                baseline=0 - tot_err_MC / sigma_data,
                edges=bkg[0].copy().axes[0].edges,
                **errps,
                label=r"$\sigma_{syst}/\sigma_{stat}$",
            )

        elif ratio_plot == "Data/MC":

            data_val = data.values()
            data_val[tot_val_MC_zero_mask] = 1

            yerr = np.sqrt(data_val) / tot_val_MC

            hep.histplot(
                data_val / tot_val_MC,
                bkg[0].copy().axes[0].edges,
                yerr=yerr,
                ax=rax,
                histtype="errorbar",
                color="k",
                capsize=4,
                flow="none",
            )
            rax.stairs(
                values=1 + tot_err_MC / tot_val_MC,
                baseline=1 - tot_err_MC / tot_val_MC,
                edges=bkg[0].copy().axes[0].edges,
                **errps,
                label=r"$\sigma_{syst}/MC$",
            )

        elif ratio_plot == "Data-MC":

            data_val = data.values()
            data_val[tot_val_MC_zero_mask] = 1

            yerr = np.sqrt(data_val) / tot_val_MC

            hep.histplot(
                data_val - tot_val_MC,
                bkg[0].copy().axes[0].edges,
                yerr=yerr,
                ax=rax,
                histtype="errorbar",
                color="k",
                capsize=4,
                flow="none",
            )
            rax.stairs(
                values=0 + tot_err_MC,
                baseline=0 - tot_err_MC,
                edges=bkg[0].copy().axes[0].edges,
                **errps,
                label=r"$\sigma_{syst}$",
            )

    # plot the signal
    if len(signal) > 0 and rax_plot_sig:

        if ratio_plot == "Pulls":

            for i, sig in enumerate(signal_mult):
                hep.histplot(
                    sig / sigma_data,
                    ax=rax,
                    linewidth=3,
                    color=color_by_sample[signal_labels[i]],
                    flow="none",
                )

        elif ratio_plot == "Data-MC":
            for i, sig in enumerate(signal_mult):

                hep.histplot(
                    sig,
                    ax=rax,
                    linewidth=3,
                    color=color_by_sample[signal_labels[i]],
                    flow="none",
                )

    # plot total signal
    if len(signal) > 0 and rax_plot_tot_sig:
        if ratio_plot == "Pulls":

            hep.histplot(
                tot_signal / (sigma_data),
                ax=rax,
                label=r"Signal/$\sigma_{stat}$",
                linewidth=2,
                color="tab:red",
                flow="none",
                histtype="fill",
            )

        elif ratio_plot == "Data-MC":

            hep.histplot(
                tot_signal,
                ax=rax,
                label=r"Total Signal",
                linewidth=2,
                color="tab:red",
                flow="none",
                histtype="fill",
            )

    rax.legend(fontsize=16, loc="upper right", ncol=3)
    rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

    # rax end
    ##########################

    # get handles and labels of legend
    handles, labels = ax.get_legend_handles_labels()

    # append legend labels in order to a list

    # get total yield of backgrounds per label
    # (sort by yield in fixed fj_pt histogram after pre-sel)
    order_dic = {}
    for bkg_label in bkg_labels:
        if bkg_label != "Rest":
            order_dic[plot_labels[bkg_label]] = h[{"Sample": bkg_label}].sum()
        else:
            order_dic[plot_labels[bkg_label]] = bkg_rest_sum.sum()

    summ = []
    for label in labels[: len(bkg_labels)]:
        summ.append(order_dic[label])
    # get indices of labels arranged by yield
    order = []
    for i in range(len(summ)):
        order.append(np.argmax(np.array(summ)))
        summ[np.argmax(np.array(summ))] = -100

    # plot bkg, then signal, then data
    hand = [handles[i] for i in order] + handles[len(bkg) : -1] + [handles[-1]]
    lab = [labels[i] for i in order] + labels[len(bkg) : -1] + [labels[-1]]

    lab_new, hand_new = [], []
    for i in range(len(lab)):
        # if "Stat" in lab[i]:
        #     continue

        lab_new.append(lab[i])
        hand_new.append(hand[i])

    ax.legend(
        [hand_new[idx] for idx in range(len(hand_new))],
        [lab_new[idx] for idx in range(len(lab_new))],
        # title=legend_title,
        loc="upper right",
        ncol=2,
        fontsize=14,
        frameon=False,
    )

    _, a = ax.get_ylim()
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-1, a * 15.7)
    else:
        ax.set_ylim(0, a * 1.7)

    # ax.set_xlim(45, 210)
    ax.set_xlim(75, 235)

    # hep.cms.lumitext("%.0f " % get_lumi(years, channels) + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
    # hep.cms.text("Work in Progress", ax=ax, fontsize=15)

    # if "prefit" in label_on_plot:
    #     label_on_plot = "Prefit"
    # elif "shapes_fit_b" in label_on_plot:
    #     label_on_plot = "B-only fit"
    # elif "shapes_fit_s" in label_on_plot:
    #     label_on_plot = "S+B fit"

    # ax.text(0.05, 0.95, label_on_plot, transform=ax.transAxes, verticalalignment="top", fontweight="bold")

    ax.text(0.05, 0.83, legend_title, fontsize=17, color="black", ha="left", transform=ax.transAxes)
    ax.text(0.05, 0.73, label_on_plot, fontsize=17, color="black", ha="left", transform=ax.transAxes)

    hep.cms.label(
        loc=1,
        data=True,
        ax=ax,
        lumi=f"{get_lumi(years, channels):.0f}",
        fontsize=18,
        llabel="Preliminary",
    )

    # save plot
    os.makedirs(out_dir, exist_ok=True)

    if out_fname:
        plt.savefig(f"{out_dir}/stacked_hists_{out_fname}.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"{out_dir}/stacked_hists.pdf", bbox_inches="tight")
