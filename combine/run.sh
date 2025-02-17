#!/bin/bash
# Based on: https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/combine/run_blinded.sh

####################################################################################################
# Script for fits
#
# 1) Combines cards and makes a workspace (--workspace / -w)
# 2) Background-only fit (--bfit / -b)
# 3) Expected asymptotic limits (--limits / -l)
# 4) Expected significance (--significance / --sig / -s)
# 5) Fit diagnostics (--dfit / -d)
# 6) Asimov fit (--dfitasimov / --dfita)
# 7) GoF on data (--gofdata / -g)
# 8) GoF on toys (--goftoys / -t),
# 9) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impactsf $nuisance), collect (--impactsc $nuisances)
# 10) Unfolding (--unfolding / -u),
# 11) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
#
# Specify seed with --seed (default 42) and number of toys with --numtoys (default 100)
#
# Usage ./run_blinded.sh [-wblsdgt] [--numtoys 100] [--seed 42]
####################################################################################################


####################################################################################################
# Read options
####################################################################################################


workspace=0
bfit=0
limits=0
significance=0
multisig=0
vbf=0
ggf=0
dfit=0
dfit_asimov=0
gofdata=0
goftoys=0
unfolding=0
impactsi=0
unblind=0
seed=444
numtoys=100
bias=-1
mintol=0.5 # --cminDefaultMinimizerTolerance
# maxcalls=1000000000  # --X-rtd MINIMIZER_MaxCalls

options=$(getopt -o "wblsdrgti" --long "workspace,bfit,limits,significance,multisig,vbf,ggf,dfit,dfitasimov,resonant,gofdata,goftoys,unfolding,impactsi,unblind,bias:,seed:,numtoys:,mintol:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        -s|--significance)
            significance=1
            ;;
        --multisig)
            multisig=1
            ;;
        --vbf)
            vbf=1
            ;;
        --ggf)
            ggf=1
            ;; 
        -u|--unfolding)
            unfolding=1
            ;;            
        -d|--dfit)
            dfit=1
            ;;
        --dfitasimov)
            dfit_asimov=1
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;                                   
        -i|--impactsi)
            impactsi=1
            ;;
        --unblind)
            unblind=1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --mintol)
            shift
            mintol=$1
            ;;
        --bias)
            shift
            bias=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

echo "Arguments: workspace=$workspace bfit=$bfit limits=$limits \
significance=$significance multisig=$multisig vbf=$vbf ggf=$ggf unfolding=$unfolding unblind=$unblind \
dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys"



####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is turned off)
####################################################################################################

dataset=data_obs
cards_dir="templates/v13"

if [ $unfolding = 1 ]; then
    cards_dir+="/datacards_unfolding"
else
    cards_dir+="/datacards"
fi

cp ${cards_dir}/testModel.root testModel.root # TODO: avoid this
CMS_PARAMS_LABEL="CMS_HWW_boosted"

# ####################################################################################################
# # Combine cards, text2workspace, fit, limits, significances, fitdiagnositcs, GoFs
# ####################################################################################################
# # # need to run this for large # of nuisances
# # # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735


# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

logsdir=${outdir}/logs
mkdir -p $logsdir
chmod +x ${logsdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

################# Edit below which cards you want to provide to combine
sr1="VBF"
sr2="ggFpt250to350"
sr3="ggFpt350to500"
sr4="ggFpt500toInf"
ccargs="SR1=${cards_dir}/${sr1}.txt SR2=${cards_dir}/${sr2}.txt SR3=${cards_dir}/${sr3}.txt SR4=${cards_dir}/${sr4}.txt"
# ccargs="SR1=${cards_dir}/${sr1}.txt"
# ccargs="SR2=${cards_dir}/${sr2}.txt SR3=${cards_dir}/${sr3}.txt SR4=${cards_dir}/${sr4}.txt"

cr1="TopCR"
cr2="WJetsCR"
ccargs+=" CR1=${cards_dir}/${cr1}.txt CR2=${cards_dir}/${cr2}.txt"
######################################################################

if [ $workspace = 1 ]; then
    echo "Combining cards:"
    for file in $ccargs; do
    echo "  ${file##*/}"
    done
    echo "-------------------------"
    combineCards.py $ccargs > $combined_datacard

    echo "Running text2workspace"

    if [ $unfolding = 1 ]; then
        echo "- Unfolding workspace"
        text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ggH_hww_200_300:r_ggH_pt200_300[1,-10,10]' --PO 'map=.*/ggH_hww_300_450:r_ggH_pt300_450[1,-10,10]' --PO 'map=.*/ggH_hww_450_Inf:r_ggH_pt450_inf[1,-10,10]' --PO 'map=.*/qqH_hww_mjj_1000_Inf:r_qqH_mjj_1000_inf[1,-10,10]' --PO 'map=.*/WH_hww:r_WH_hww[1,-10,10]' --PO 'map=.*/ZH_hww:r_ZH_hww[1,-10,10]' --PO 'map=.*/ttH_hww:r_ttH_hww[1,-10,10]' -o $ws 2>&1 | tee $logsdir/text2workspace.txt
    else
        echo "- Inclusive workspace"
        if [ $multisig = 1 ]; then
            echo "- Multiple POIs workspace"
            # seperate POIs (to make Table 30 in v11)
            text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ggH_hww:r_ggH_hww[1,0,10]' --PO 'map=.*/qqH_hww:r_qqH_hww[1,0,10]' --PO 'map=.*/WH_hww:r_WH_hww[1,0,10]' --PO 'map=.*/ZH_hww:r_ZH_hww[1,0,10]' --PO 'map=.*/ttH_hww:r_ttH_hww[1,0,10]' -o $ws 2>&1
        else
            echo "- Single POI workspace"
            # single POI
            text2workspace.py $combined_datacard --channel-masks -o $ws 2>&1 | tee $logsdir/text2workspace.txt    
        fi
    fi
    echo "-------------------------"
else
    if [ ! -f "$ws" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi


if [ $significance = 1 ]; then
    echo "Expected significance"

    if [ $multisig = 1 ]; then
        
        if [ $vbf = 1 ]; then
            
            echo "VBF significance"
            combine -M Significance -d $ws -t -1 --redefineSignalPOIs r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww
        
        elif [ $ggf = 1 ]; then
            
            echo "ggF Significance"
            combine -M Significance -d $ws -t -1 --redefineSignalPOIs r_ggH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww
        else
            echo "must provide either --ggf or --vbf"
        fi
    else
        echo "Total significance"
        combine -M Significance -d $ws -m 125 --expectSignal=1 --rMin -1 --rMax 5 -t -1
    fi
fi

if [ $dfit = 1 ]; then

    # combine -M MultiDimFit -d $ws -n _paramFit_test_wjetsnormSF --algo impact --redefineSignalPOIs r -P wjetsnormSF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit 1 --rMin -10 --rMax 10 -m 125 -v2 --setParameters ps_fsr_wjets=1 --freezeParameters ps_fsr_wjets
    
    # combine -M MultiDimFit -d $ws -n _paramFit_test_CMS_HWW_boosted_WJetsCR_mcstat_bin4 --algo impact --redefineSignalPOIs r -P CMS_HWW_boosted_WJetsCR_mcstat_bin4 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit 1 --rMin -20 --rMax 20 -m 125 -v2

    # combine -M MultiDimFit -d $ws --algo grid  -m 125 --rMin -10 --rMax 10 --points 20 -P CMS_HWW_boosted_taggereff --redefineSignalPOIs r --floatOtherPOIs 1 --setParameterRanges name=min,max[:CMS_HWW_boosted_taggereff=-3,3:]
    # combine -M MultiDimFit -d $ws --algo grid -n test_ -m 125 --rMin -10 --rMax 10 --points 30 --freezeParameters allConstrainedNuisances

    combine -M MultiDimFit -d $ws -n test_ --algo grid -P CMS_HWW_boosted_taggereff_ggF -m 125 --rMin 0 --rMax 10 --setParameterRanges name=min,max[:CMS_HWW_boosted_taggereff_ggF=-3,3:] --points 60 --floatOtherPOIs 1  --setParameters mask_SR1=1

    # plot 1D scan
    # python3 CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py higgsCombinetest_.MultiDimFit.mH125.root -o output --POI CMS_HWW_boosted_taggereff 
fi

if [ $dfit_asimov = 1 ]; then

    echo "Fit Diagnostics Asimov"
    combine -M FitDiagnostics -m 125 -d $ws \
    -t -1 --expectSignal=1 --saveWorkspace -n Asimov --ignoreCovWarning \
    --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnosticsAsimov.txt

fi

if [ $limits = 1 ]; then
    # echo "Expected limits"
    # combine -M AsymptoticLimits -m 125 -n "" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v 1 \
    # --saveWorkspace --saveToys --bypassFrequentistFit -s $seed \
    # --floatParameters r --toysFrequentist --run blind 2>&1 | tee $logsdir/AsymptoticLimits.txt

    combine -M AsymptoticLimits --run expected -d $ws -t -1  -v 1 --expectSignal 1
fi


if [ $impactsi = 1 ]; then

    echo "Initial fit for impacts"

    if [ $unblind = 1 ]; then
        echo Impacts unblinded
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doInitialFit --expectSignal 1
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --output impacts.json --expectSignal 1
        plotImpacts.py -i impacts.json -o impacts --blind


        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doInitialFit --expectSignal 1 --named CMS_pileup_2018
        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50 --named CMS_pileup_2018
        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --output impacts.json --expectSignal 1 --named CMS_pileup_2018
        # plotImpacts.py -i impacts.json -o impacts --blind

        #  --setParameters mask_SR1=1

        # combineTool.py -M Impacts -d $ws -m 125 --robustFit 1 --doInitialFit --named CMS_HWW_boosted_taggereff --setParameters r=0 --freezeParameters r
        # combineTool.py -M Impacts -d $ws -m 125 --robustFit 1 --doFits --parallel 50 --named CMS_HWW_boosted_taggereff --setParameters r=0 --freezeParameters r
        # combineTool.py -M Impacts -d $ws -m 125 --robustFit 1 --output impacts.json --named CMS_HWW_boosted_taggereff --setParameters r=0 --freezeParameters r
        # plotImpacts.py -i impacts.json -o impacts --blind        

        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doInitialFit --expectSignal 1 --named ps_fsr_wjets_2018,ps_fsr_wjets_2017,ps_fsr_wjets_2016,ps_fsr_wjets_2016APV
        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50 --named ps_fsr_wjets_2018,ps_fsr_wjets_2017,ps_fsr_wjets_2016,ps_fsr_wjets_2016APV
        # combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --output impacts.json --expectSignal 1 --named ps_fsr_wjets_2018,ps_fsr_wjets_2017,ps_fsr_wjets_2016,ps_fsr_wjets_2016APV
        # plotImpacts.py -i impacts.json -o impacts --blind
    else
        echo Impacts blinded
        combineTool.py -M Impacts -d $ws -t -1 --rMin -1 --rMax 2 -m 125 --robustFit 1 --doInitialFit --expectSignal 1
        combineTool.py -M Impacts -d $ws -t -1 --rMin -1 --rMax 2 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50
        combineTool.py -M Impacts -d $ws -t -1 --rMin -1 --rMax 2 -m 125 --robustFit 1 --output impacts.json --expectSignal 1
        plotImpacts.py -i impacts.json -o impacts      
    fi

fi

if [ $goftoys = 1 ]; then
    echo "GoF on toys"
    combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys | tee $logsdir/GoF_toys.txt
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_test --setParameters ttbarnormSF=1.0,wjetsnormSF=1.0 --freezeParameters ttbarnormSF,wjetsnormSF

    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_SR --setParameters ttbarnormSF=1.0,wjetsnormSF=1.0,mask_CR1=1,mask_CR2=1 --freezeParameters ttbarnormSF,wjetsnormSF | tee $logsdir/GoF_toys.txt

    # mask CR (!!!must freeze the rate params in the SRs; either to 1 or whatever they are when I fit to the CRs)
    # look at the postfit 

    
    # # mask SR1 --> masking VBF (!!!must freeze the rate constraints)
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_ggF --setParameters mask_SR1=1 | tee $logsdir/GoF_toys.txt

    # # mask SR2, SR3, SR4 --> masking ggF  (!!!must freeze the rate constraints)
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_VBF --setParameters mask_SR2=1,mask_SR3=1,mask_SR4=1 | tee $logsdir/GoF_toys.txt

    # # mask all signal regions
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_CR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1 --freezeParameters r | tee $logsdir/GoF_toys.txt
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_wjetsCR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1,mask_CR1=1 --freezeParameters r | tee $logsdir/GoF_toys.txt
    # combine -M GoodnessOfFit -d $ws --algo=saturated -t 100 -s 1 -m 125 --toysFrequentist -n Toys_topCR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1,mask_CR2=1 --freezeParameters r | tee $logsdir/GoF_toys.txt


fi

if [ $gofdata = 1 ]; then
    echo "GoF on data"
    combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed | tee $logsdir/GoF_data.txt
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_SR --setParameters ttbarnormSF=1.0,wjetsnormSF=1.0,mask_CR1=1,mask_CR2=1 --freezeParameters ttbarnormSF,wjetsnormSF

    # # mask SR1 --> masking VBF
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_ggF --setParameters mask_SR1=1 | tee $logsdir/GoF_data.txt

    # mask SR2, SR3, SR4 --> masking ggF
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_VBF --setParameters mask_SR2=1,mask_SR3=1,mask_SR4=1 | tee $logsdir/GoF_data.txt

    # mask all signal regions
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_CR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1 --freezeParameters r | tee $logsdir/GoF_data.txt
    
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_wjetsCR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1,mask_SR4=1,mask_CR1=1 --freezeParameters r | tee $logsdir/GoF_data.txt
    # combine -M GoodnessOfFit -d $ws --algo=saturated -s 1 -m 125 -n Observed_topCR --setParameters r=0,mask_SR1=1,mask_SR2=1,mask_SR3=1,mask_SR4=1,mask_SR4=1,mask_CR2=1 --freezeParameters r | tee $logsdir/GoF_data.txt


fi

if [ $unfolding = 1 ]; then

    combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_mjj_1000_inf=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww

fi
