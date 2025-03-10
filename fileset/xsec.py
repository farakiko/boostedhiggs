import json

"""
Cross Sections

Jennet has a nice notebook on x-sections:
https://github.com/jennetd/hbb-coffea/blob/master/xsec-json.ipynb

Other references:
- Latinos:
  https://github.com/latinos/LatinoAnalysis/blob/master/NanoGardener/python/framework/samples/samplesCrossSections2018.py
- XSDB: https://xsdb-temp.app.cern.ch/xsdb
"""

# Branching ratio of H to WW (LHC-XS-WG)
BR_HWW = 0.2137
# Branching ratio of WW to LNuQQ (0.43328544)
BR_WW_lnuqq = (0.1046 + 0.1050 + 0.1075) * (0.6832) * 2
# Branching ratio of WW to 4Q (0.46676224)
BR_WW_qqqq = 0.6832**2
# Branching ratio of WW to LNuLNu (0.1005)
BR_WW_lnulnu = (0.1046 + 0.1050 + 0.1075) ** 2
# Branching ratio of Z to LL
BR_Z_ll = 3 * 0.033658

BR_THadronic = 0.667
BR_TLeptonic = 1 - BR_THadronic

BR_HBB = 0.5809
BR_Htt = 0.0621

xs = {}

# VH - this needs to be checked for qqZH, also need to add ggZH
"""
pp->ZH 8.839E-01
From GenXsecAnalyzer: 7.891e-01 +/- 3.047e-03
cmsRun ana.py
inputFiles="file:root://cmsxrootd.fnal.gov//store/mc/RunIIAutumn18MiniAOD/ \
HZJ_HToWW_M125_13TeV_powheg_jhugen714_pythia8_TuneCP5/MINIAODSIM/ \
102X_upgrade2018_realistic_v15-v1/280000/B18DBE21-109B-6F4C-9920-A797AAC1D1CE.root" maxEvents=-1
"""

# GluGluToHToWWPt200
# - From powheg: 0.4716 pb
# - PKU uses 0.1014pb after BR
xs["GluGluHToWW_Pt-200ToInf_M-125"] = 0.4716 * BR_HWW

# GluGluHToWWToLNuQQ
# - From GenXsecAnalyzer: 28.87 pb
#    - Times BR: 28.87*0.2137*0.4332 = 2.67 pb
#  - From LHC-XS-WG:
#    - ggF: 48.58 pb (N3LO)
#           44.14 pb (NNLO + NNLL)
#      - Times BR: 48.58*0.2137*0.4332 = 4.497 pb
#  - From PKU:
#    - ggF: 48.61 pb
xs["GluGluHToWWToLNuQQ"] = 48.58 * BR_HWW * BR_WW_lnuqq
xs["GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8"] = 48.58 * BR_HWW * BR_WW_lnuqq

# - From AN-2023-102:
#  0.9913 / (BR_HWW *BR_WW_lnulnu) = 46.15
# - From AN-2020-046
# 1.0913430 / (BR_HWW *BR_WW_lnulnu) = 50.8
# xs["GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8"] = 0.9913

# VBFHToWWToLNuQQ-MH125
#  - From GenXsecAnalyzer: 3.892 pb
#  - From LHC-XS-WG:
#    - VBF: 3.782E pb (NNLO QCD + NLO EW)
#      - Times BR: 3.782*0.2137*0.21664272 = 0.1751 pb
vbf_xsec = 3.782 * BR_HWW * BR_WW_lnuqq
xs["VBFHToWWToLNuQQ-MH125"] = 3.782 * BR_HWW * BR_WW_lnuqq
xs["VBFHToWWToLNuQQ_M-125_withDipoleRecoil"] = vbf_xsec

xs["VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil"] = 3.782 * BR_HWW

# Using LHCXS numbers
xs["HWminusJ_HToWW_M-125"] = 0.5328 * BR_HWW
xs["HWplusJ_HToWW_M-125"] = 0.84 * BR_HWW
xs["HZJ_HToWW_M-125"] = 0.8839 * BR_HWW
xs["GluGluZH_HToWW_ZTo2L_M-125"] = 0.1227 * BR_Z_ll * BR_HWW  # 0.002639
xs["GluGluZH_HToWW_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.1227 * BR_HWW
xs["GluGluZH_HToWW_M-125"] = 0.1227 * BR_HWW
xs["ttHToNonbb_M125"] = 0.5071 * (1 - BR_HBB)

# QCD
xs["QCD_HT500to700"] = 3.033e04
xs["QCD_HT700to1000"] = 6.412e03
xs["QCD_HT1000to1500"] = 1.118e03
xs["QCD_HT1500to2000"] = 1.085e02
xs["QCD_HT2000toInf"] = 2.194e01

xs["QCD_Pt_120to170"] = 4.074e05
xs["QCD_Pt_170to300"] = 1.035e05
xs["QCD_Pt_300to470"] = 6.833e03
xs["QCD_Pt_470to600"] = 5.495e02
xs["QCD_Pt_600to800"] = 156.5
xs["QCD_Pt_800to1000"] = 2.622e01
xs["QCD_Pt_1000to1400"] = 7.475e00
xs["QCD_Pt_1400to1800"] = 6.482e-01
xs["QCD_Pt_1800to2400"] = 8.742e-02
xs["QCD_Pt_2400to3200"] = 5.237e-03
xs["QCD_Pt_3200toInf"] = 1.353e-04

# TTbar
# from XSDB
xs["TTTo2L2Nu"] = 87.314
xs["TTToHadronic"] = 380.094
xs["TTToSemiLeptonic"] = 364.351

# TTV
xs["TTZToQQ"] = 0.5113
xs["TTWJetsToQQ"] = 0.4377
xs["TTWJetsToLNu"] = 0.2176  # TuneCP5up!!!
xs["TTZToLLNuNu_M-10"] = 0.2439

# Single Top
xs["ST_s-channel_4f_hadronicDecays"] = 3.549e00 * BR_THadronic
xs["ST_s-channel_4f_leptonDecays"] = 3.549e00 * BR_TLeptonic
xs["ST_t-channel_antitop_4f_InclusiveDecays"] = 6.793e01
xs["ST_t-channel_antitop_5f_InclusiveDecays"] = 7.174e01
xs["ST_t-channel_top_4f_InclusiveDecays"] = 1.134e02
xs["ST_t-channel_top_5f_InclusiveDecays"] = 1.197e02
xs["ST_tW_antitop_5f_inclusiveDecays"] = 3.251e01
xs["ST_tW_antitop_5f_NoFullyHadronicDecays"] = 3.251e01 * BR_TLeptonic
xs["ST_tW_top_5f_inclusiveDecays"] = 3.245e01
xs["ST_tW_top_5f_NoFullyHadronicDecays"] = 3.245e01 * BR_TLeptonic

# W+jets W(qq)
xs["WJetsToQQ_HT-200to400"] = 2549.0
xs["WJetsToQQ_HT-400to600"] = 2.770e02
xs["WJetsToQQ_HT-600to800"] = 5.906e01
xs["WJetsToQQ_HT-800toInf"] = 2.875e01

# W+jets W(lv)
# from XSDB (miniaodv2) - numbers after # correspond to miniaodv1/preUL?
xs["WJetsToLNu_HT-70To100"] = 1264.0 * 1.21  # 1292.0
xs["WJetsToLNu_HT-100To200"] = 1256.0 * 0.993  # 1395.0
xs["WJetsToLNu_HT-200To400"] = 335.5 * 1.002  # 407.9
xs["WJetsToLNu_HT-400To600"] = 45.25 * 1.009  # 57.48
xs["WJetsToLNu_HT-600To800"] = 11.19 * 1.120  # 12.87
xs["WJetsToLNu_HT-800To1200"] = 4.933 * 1.202  # 5.366
xs["WJetsToLNu_HT-1200To2500"] = 1.16 * 1.332  # 1.074
xs["WJetsToLNu_HT-2500ToInf"] = 0.008001 * 4.2  # 0.026

xs["WJetsToLNu_TuneCP5_13TeV-madgraphMLM"] = 53940.0  # 52940.0

# W+jets W(lv) NLO (xsdb)
# WJetsToLNu_*J_TuneCP5_13TeV-amcatnloFXFX-pythia8
xs["WJetsToLNu_0J"] = 52780.0
xs["WJetsToLNu_1J"] = 8832.0
xs["WJetsToLNu_2J"] = 3276.0

# from xsdb, note this is sherpa
xs["WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr"] = 62070.0

# From running xsecanalyzer on one file ~4GB
xs["WJetsToLNu_LHEFilterPtW-250To400"] = 27.85
xs["WJetsToLNu_LHEFilterPtW-400To600"] = 3.622
xs["WJetsToLNu_LHEFilterPtW-600ToInf"] = 0.5501

xs["WJetsToLNu_Pt-100To250_MatchEWPDG20"] = 763.7
xs["WJetsToLNu_Pt-250To400_MatchEWPDG20"] = 27.55
xs["WJetsToLNu_Pt-400To600_MatchEWPDG20"] = 3.477
xs["WJetsToLNu_Pt-600ToInf_MatchEWPDG20"] = 0.5415

# Z+jets Z(qq)
xs["ZJetsToQQ_HT-200to400"] = 1012.0
xs["ZJetsToQQ_HT-400to600"] = 1.145e02
xs["ZJetsToQQ_HT-600to800"] = 2.541e01
xs["ZJetsToQQ_HT-800toInf"] = 1.291e01

# DY+jets
# ref: https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/205 (v11)
# they still need a k-factor depending on flavor composition?
# k-factor of 0.93 is NNLO correction
xs["DYJetsToLL_M-10to50"] = 15810.0
xs["DYJetsToLL_M-50"] = 5343.0
xs["DYJetsToLL_M-50_HT-70to100"] = 146.5
xs["DYJetsToLL_M-50_HT-100to200"] = 160.8
xs["DYJetsToLL_M-50_HT-200to400"] = 48.63
xs["DYJetsToLL_M-50_HT-400to600"] = 6.982
xs["DYJetsToLL_M-50_HT-600to800"] = 1.756
xs["DYJetsToLL_M-50_HT-800to1200"] = 0.8094
xs["DYJetsToLL_M-50_HT-1200to2500"] = 0.1931
xs["DYJetsToLL_M-50_HT-2500toInf"] = 0.003513

# Old NLO samples (invalid)
# xs["DYJetsToLL_Pt-50To100"] = 3.941e02
# xs["DYJetsToLL_Pt-100To250"] = 9.442e01
# xs["DYJetsToLL_Pt-250To400"] = 3.651e00
# xs["DYJetsToLL_Pt-400To650"] = 4.986e-01
# xs["DYJetsToLL_Pt-650ToInf"] = 4.678e-02

# NLO samples (xsdb)
xs["DYJetsToLL_LHEFilterPtZ-0To50"] = 1485.0
xs["DYJetsToLL_LHEFilterPtZ-50To100"] = 397.4
xs["DYJetsToLL_LHEFilterPtZ-100To250"] = 97.2
xs["DYJetsToLL_LHEFilterPtZ-250To400"] = 3.701
xs["DYJetsToLL_LHEFilterPtZ-400To650"] = 0.5086
xs["DYJetsToLL_LHEFilterPtZ-650ToInf"] = 0.04728

# VV
# NLO prediction from papers
xs["WW"] = 118.7
xs["WZ"] = 46.74
xs["ZZ"] = 16.91

# From VBS AN: AN2023_016_v11
# EWK Z
xs["EWKZ_ZToQQ"] = 10.67
xs["EWKZ_ZToLL"] = 6.22
xs["EWKZ_ZToNuNu"] = 10.72
# EWK W
xs["EWKWminus_WToQQ"] = 10.67
xs["EWKWplus_WToQQ"] = 10.67
xs["EWKWminus_WToLNu"] = 32.26
xs["EWKWplus_WToLNu"] = 39.33

# Higgs to BB
xs["GluGluHToBB"] = 4.716e-01 * BR_HBB
xs["VBFHToBB"] = 3.873e00 * BR_HBB
xs["VBFHToBBDipoleRecoilOn"] = xs["VBFHToBB"]

xs["WminusH_HToBB_WToQQ"] = 3.675e-01 * BR_HBB
xs["WplusH_HToBB_WToQQ"] = 5.890e-01 * BR_HBB
xs["WminusH_HToBB_WToLNu"] = 1.770e-01 * BR_HBB
xs["WplusH_HToBB_WToLNu"] = 2.832e-01 * BR_HBB

xs["ZH_HToBB_ZToQQ"] = 5.612e-01 * BR_HBB
xs["ZH_HToBB_ZToLL"] = 7.977e-02 * BR_HBB
xs["ZH_HToBB_ZToNuNu"] = 1.573e-01 * BR_HBB

xs["ggZH_HToBB_ZToNuNu"] = 1.222e-02 * BR_HBB
xs["ggZH_HToBB_ZToQQ"] = 4.319e-02 * BR_HBB
xs["ggZH_HToBB_ZToLL"] = 6.185e-03 * BR_HBB

xs["ttHToBB"] = 5.013e-01 * BR_HBB

# Higgs to Tau Tau
xs["GluGluHToTauTau"] = 48.58 * BR_Htt
xs["VBFHToTauTau"] = 3.770 * BR_Htt
xs["WminusHToTauTau"] = 0.5272 * BR_Htt
xs["WplusHToTauTau"] = 0.8331 * BR_Htt
xs["ZHToTauTau"] = 0.7544 * BR_Htt
xs["ttHToTauTau"] = 0.5033 * BR_Htt

print(xs)

with open("xsec_pfnano.json", "w") as outfile:
    json.dump(xs, outfile, indent=4)
