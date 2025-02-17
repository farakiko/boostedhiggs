#!/usr/bin/python

import json
import os
import pickle as pkl
import warnings

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import onnx
import onnxruntime as ort
import scipy

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")


combine_samples_by_name = {
    "GluGluHToWW_Pt-200ToInf_M-125": "ggF",
    "VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil": "VBF",
    "ttHToNonbb_M125": "ttH",
    "HWminusJ_HToWW_M-125": "WH",
    "HWplusJ_HToWW_M-125": "WH",
    "HZJ_HToWW_M-125": "ZH",
    "GluGluZH_HToWW_M-125_TuneCP5_13TeV-powheg-pythia8": "ZH",
    "GluGluHToTauTau": "HTauTau",
}

combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # bkg
    "QCD_Pt": "QCD",
    "TT": "TTbar",
    "WJetsToLNu_": "WJetsLNu",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "EWK": "EWKvjets",
    # TODO: make sure it's WZQQ is NLO in next iteration
    "DYJets": "WZQQorDYJets",
    "JetsToQQ": "WZQQorDYJets",
}

signals = ["VBF", "ggF"]


def get_sum_sumgenweight(pkl_files, year, sample):
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


# ---------------------------------------------------------
# TAGGER STUFF
def get_finetuned_score(data, modelv="v2_nor2"):
    # add finetuned tagger score
    PATH = f"../../weaver-core-dev/experiments_finetuning/{modelv}/model.onnx"

    input_dict = {
        "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
    }

    onnx_model = onnx.load(PATH)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(
        PATH,
        providers=["AzureExecutionProvider"],
    )
    outputs = ort_sess.run(None, input_dict)

    return scipy.special.softmax(outputs[0], axis=1)[:, 0]


# ---------------------------------------------------------

# PLOTTING UTILS
color_by_sample = {
    "ggF": "lightsteelblue",
    "VBF": "peru",
    # signal that is background
    "WH": "tab:brown",
    "ZH": "yellowgreen",
    "ttH": "tab:olive",
    # background
    "QCD": "tab:orange",
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
    "Fake": "tab:orange",
}

plot_labels = {
    "ggF": "ggF",
    "WH": "WH",
    "ZH": "ZH",
    "VH": "VH",
    # "VH": "VH(WW)",
    # "VBF": r"VBFH(WW) $(qq\ell\nu)$",
    "VBF": r"VBF",
    # "ttH": "ttH(WW)",
    "ttH": r"$t\bar{t}$H",
    "QCD": "Multijet",
    "Diboson": "VV",
    "WJetsLNu": r"W$(\ell\nu)$+jets",
    "TTbar": r"$t\bar{t}$+jets",
    "SingleTop": r"Single T",
    #     "WplusHToTauTau": "WplusHToTauTau",
    #     "WminusHToTauTau": "WminusHToTauTau",
    #     "ttHToTauTau": "ttHToTauTau",
    #     "GluGluHToTauTau": "GluGluHToTauTau",
    #     "ZHToTauTau": "ZHToTauTau",
    #     "VBFHToTauTau": "VBFHToTauTau"
    "WJetsLNu_unmatched": r"W$(\ell\nu)$+jets unmatched",
    "WJetsLNu_matched": r"W$(\ell\nu)$+jets matched",
    "EWKvjets": "EWK VJets",
    # TODO: make sure it's WZQQ is NLO in next iteration
    "DYJets": r"Z$(\ell\ell)$+jets",
    "WZQQ": r"V$(qq)$",
    "WZQQorDYJets": r"W$(qq)$/Z(inc.)+jets",
    "Fake": "Fake",
}

label_by_ch = {"mu": "Muon", "ele": "Electron"}

label_by_ch = {"mu": "Muon", "ele": "Electron"}


def plot_hists(
    h,
    years,
    channels,
    add_data,
    logy,
    add_soverb,
    only_sig,
    mult,
    outpath,
    text_="",
    blind_region=None,
    save_as=None,
    remove_samples=[],
    plot_tot_bkg_sig=False,
    plot_ratio_pulls=False,
    use_postfit_errorbars=False,
    postfit_errorbars_mc=None,
    postfit_errorbars_data=None,
    label_on_plot="",
):
    # luminosity
    luminosity = 0
    for year in years:
        lum = 0
        for ch in channels:
            with open("../fileset/luminosity.json") as f:
                lum += json.load(f)[ch][year] / 1000.0

        luminosity += lum / len(channels)

    # get samples existing in histogram
    samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

    for s in remove_samples:
        if s in samples:
            samples.remove(s)

    signal_labels = [label for label in samples if label in signals]
    bkg_labels = [label for label in samples if (label and label not in signal_labels and (label not in ["Data"]))]

    # get total yield of backgrounds per label
    # (sort by yield in fixed fj_pt histogram after pre-sel)
    order_dic = {}
    for bkg_label in bkg_labels:
        order_dic[plot_labels[bkg_label]] = h[{"Sample": bkg_label}].sum()

    # data
    if add_data:
        data = h[{"Sample": "Data"}]

    # signal
    signal = [h[{"Sample": label}] for label in signal_labels]
    signal_mult = [s * mult for s in signal]

    # background
    bkg = [h[{"Sample": label}] for label in bkg_labels]

    if plot_ratio_pulls:
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9, 9),
            gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
            sharex=True,
        )
    else:
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
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

    # sum all of the background
    if len(bkg) > 0:
        tot = bkg[0].copy()
        for i, b in enumerate(bkg):
            if i > 0:
                tot = tot + b

        tot_val = tot.values()
        tot_val_zero_mask = tot_val == 0  # check if this is for the ratio or not
        tot_val[tot_val_zero_mask] = 1

        tot_err_MC = np.sqrt(tot.values())

        if use_postfit_errorbars is True:
            tot_err_MC = postfit_errorbars_mc

    if add_data and data:
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

        if len(bkg) > 0:

            data_val = data.values()
            data_val[tot_val_zero_mask] = 1

            if plot_ratio_pulls:

                if use_postfit_errorbars is True:
                    sigma_data = postfit_errorbars_data
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

                pulls = (data_val - tot_val) / sigma_data

                hep.histplot(
                    pulls,
                    tot.axes[0].edges,
                    yerr=1,
                    ax=rax,
                    histtype="errorbar",
                    color="k",
                    capsize=4,
                    flow="none",
                )
                rax.stairs(
                    values=0 + tot_err_MC / sigma_data,
                    baseline=0 - tot_err_MC / sigma_data,
                    edges=tot.axes[0].edges,
                    **errps,
                    label=r"$\sigma_{syst}/\sigma_{data}$",
                )

                rax.axhline(0, ls="--", color="k")
                rax.set_ylim(-5, 5)
                rax.set_ylabel(r"Pull: $\frac{Data-MC}{\sigma_{data}}$", fontsize=18, labelpad=10)
                rax.legend(fontsize=14, loc="upper right")

            else:
                # from hist.intervals import ratio_uncertainty
                # yerr = ratio_uncertainty(data_val, tot_val, "poisson")
                yerr = np.sqrt(data_val) / tot_val

                hep.histplot(
                    data_val / tot_val,
                    tot.axes[0].edges,
                    yerr=yerr,
                    ax=rax,
                    histtype="errorbar",
                    color="k",
                    capsize=4,
                    flow="none",
                )
                rax.stairs(
                    values=1 + tot_err_MC / tot_val,
                    baseline=1 - tot_err_MC / tot_val,
                    edges=tot.axes[0].edges,
                    **errps,
                    label="Stat. unc.",
                )

                rax.axhline(1, ls="--", color="k")
                rax.set_ylim(0.2, 1.8)
                rax.set_ylabel("Data-MC", fontsize=20, labelpad=10)

    # plot the background
    if len(bkg) > 0 and not only_sig:
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
        if not plot_tot_bkg_sig:
            ax.stairs(
                values=tot.values() + tot_err_MC,
                baseline=tot.values() - tot_err_MC,
                edges=tot.axes[0].edges,
                **errps,
                # label="Stat. unc.",
                label="Syst. unc.",
            )

    # ax.text(0.5, 0.9, text_, fontsize=14, transform=ax.transAxes, weight="bold")

    # plot the signal (times 10)
    if plot_tot_bkg_sig:
        if len(signal) > 0:
            tot_signal = None
            for i, sig in enumerate(signal_mult):
                if tot_signal is None:
                    tot_signal = signal[i].copy()
                else:
                    tot_signal = tot_signal + signal[i]

            tot_signal *= mult_factor

            if mult is not None:
                if mult == 1:
                    siglabel = r"Background + Signal"
                else:
                    siglabel = r"Background + Signal $\times$" + f"{mult_factor}"

                # tot_signal += np.array(bkg).sum(axis=0)
                hep.histplot(
                    tot_signal,
                    ax=ax,
                    label=siglabel,
                    linewidth=2,
                    color="tab:red",
                    flow="none",
                )
                # add MC stat errors
                ax.stairs(
                    values=tot_signal.values() + np.sqrt(tot_signal.values()),
                    baseline=tot_signal.values() - np.sqrt(tot_signal.values()),
                    edges=sig.axes[0].edges,
                    **errps,
                )

    ax.set_ylabel("Events")

    ax.set_xlabel("")
    rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

    # get handles and labels of legend
    handles, labels = ax.get_legend_handles_labels()

    # append legend labels in order to a list
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
        title=text_,
        ncol=2,
        fontsize=14,
    )

    _, a = ax.get_ylim()
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-1, a * 15.7)
    else:
        ax.set_ylim(0, a * 1.7)

    # ax.set_xlim(45, 210)
    ax.set_xlim(75, 235)

    hep.cms.lumitext("%.0f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
    hep.cms.text("Work in Progress", ax=ax, fontsize=15)

    ax.text(0.05, 0.95, label_on_plot, transform=ax.transAxes, verticalalignment="top", fontweight="bold")

    # save plot
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if save_as:
        plt.savefig(f"{outpath}/stacked_hists_{save_as}.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"{outpath}/stacked_hists.pdf", bbox_inches="tight")
