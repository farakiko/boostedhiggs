import math
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from scipy.stats import f

fig, ax = plt.subplots(1, 1)

if __name__ == "__main__":

    thisdir = os.getcwd()

    gof_toys = []

    infile1 = ROOT.TFile.Open("higgsCombineToys.GoodnessOfFit.mH125.1.root")
    tree1 = infile1.Get("limit")
    for j in range(tree1.GetEntries()):
        tree1.GetEntry(j)
        gof_toys += [getattr(tree1, "limit")]

    ntoys = tree1.GetEntries()

    # Observed
    infile1 = ROOT.TFile.Open("higgsCombineObserved.GoodnessOfFit.mH125.1.root")
    tree1 = infile1.Get("limit")
    tree1.GetEntry(0)
    gof_obs = getattr(tree1, "limit")

    ashist = plt.hist(gof_toys, bins=20, histtype="step", color="black")
    ymax = 1.2 * max(ashist[0])

    plt.errorbar(
        (ashist[1][:-1] + ashist[1][1:]) / 2.0,
        ashist[0],
        yerr=np.sqrt(ashist[0]),
        linestyle="",
        color="black",
        marker="o",
        label=str(ntoys) + " toys",
    )
    mylabel = "obs = {:.2f}".format(gof_obs)

    pvalue = 1.0 * len([y for y in gof_toys if y > gof_obs]) / ntoys
    mylabel += ", p = {:.2f}".format(pvalue)
    print(pvalue)

    plt.ylim(0, ymax)
    plt.plot([gof_obs, gof_obs], [0, ymax], color="red", label=mylabel)
    plt.legend(loc="upper right", frameon=False)
    plt.xlabel("Goodness of fit (saturated)")

    plt.savefig(thisdir + "/gof.pdf", bbox_inches="tight")
    plt.show()
