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
obs=0
multisig=0
vbf=0
ggf=0
dfit=0
dfit_asimov=0
gofdata=0
goftoys=0
unfolding=0
channelcompatibility=0
impactsi=0
unblind=0
scanparam=0
seed=444
numtoys=100
bias=-1
mintol=0.5 # --cminDefaultMinimizerTolerance
# maxcalls=1000000000  # --X-rtd MINIMIZER_MaxCalls

options=$(getopt -o "wblsdrgti" --long "workspace,bfit,limits,significance,obs,multisig,vbf,ggf,dfit,dfitasimov,resonant,gofdata,goftoys,unfolding,channelcompatibility,impactsi,unblind,scanparam,bias:,seed:,numtoys:,mintol:" -- "$@")
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
        -sobs|--obs)
            obs=1
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
        -ch|--channelcompatibility)
            channelcompatibility=1
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
        --scanparam)
            scanparam=1
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
significance=$significance obs=$obs multisig=$multisig vbf=$vbf ggf=$ggf unfolding=$unfolding channelcompatibility=$channelcompatibility unblind=$unblind scanparam=$scanparam \
dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys"



####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is turned off)
####################################################################################################

dataset=data_obs
cards_dir="templates/v17/"

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

cr1="TopCR"
cr2="WJetsCR"
ccargs+=" CR1=${cards_dir}/${cr1}.txt CR2=${cards_dir}/${cr2}.txt"

# cr1="TopCR"
# cr2="WJetsCR1"
# cr3="WJetsCR2"
# cr4="WJetsCR3"
# ccargs+=" CR1=${cards_dir}/${cr1}.txt CR2=${cards_dir}/${cr2}.txt CR3=${cards_dir}/${cr3}.txt CR4=${cards_dir}/${cr4}.txt"
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
        text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks --PO verbose --PO 'map=.*/ggH_hww_200_300:r_ggH_pt200_300[1,-10,10]' --PO 'map=.*/ggH_hww_300_450:r_ggH_pt300_450[1,-10,10]' --PO 'map=.*/ggH_hww_450_Inf:r_ggH_pt450_inf[1,-10,10]' --PO 'map=.*/qqH_hww_mjj_1000_Inf:r_qqH_mjj_1000_inf[1,-10,10]' --PO 'map=.*/WH_hww:r_otherH[1,-10,10]' --PO 'map=.*/ZH_hww:r_otherH[1,-10,10]' --PO 'map=.*/ttH_hww:r_otherH[1,-10,10]' -o $ws 2>&1 | tee $logsdir/text2workspace.txt
    else
        if [ $channelcompatibility = 1 ]; then
            echo "- Channel compatibality workspace"
            text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose  --PO 'map=SR1/.*H_hww:r[1,-20,20]'  --PO 'map=SR2/.*H_hww:r[1,-20,20]'  --PO 'map=SR3/.*H_hww:r[1,-20,20]'  --PO 'map=SR4/.*H_hww:r[1,-20,20]' -o $ws 2>&1 | tee $logsdir/text2workspace.txt
            combine -M ChannelCompatibilityCheck -d $ws -m 125 -n HWW --setParameterRanges r=-20,20
        elif [ $multisig = 1 ]; then
            echo "- Two POIs workspace"
            text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks --PO verbose --PO 'map=.*/ggH_hww:r_ggH_hww[1,0,10]' --PO 'map=.*/qqH_hww:r_qqH_hww[1,0,10]' --PO 'map=.*/WH_hww:r_otherH[1,0,10]' --PO 'map=.*/ZH_hww:r_otherH[1,0,10]' --PO 'map=.*/ttH_hww:r_otherH[1,0,10]' -o $ws 2>&1
        else
            echo "- Single POI workspace"
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
    
    if [ $obs = 1 ]; then
        echo "Observed significance"

        if [ $multisig = 1 ]; then
            
            if [ $vbf = 1 ]; then
                
                echo "VBF significance"
                combine -M Significance -d $ws --redefineSignalPOIs r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            
            elif [ $ggf = 1 ]; then
                
                echo "ggF Significance"
                combine -M Significance -d $ws --redefineSignalPOIs r_ggH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            else
                echo "must provide either --ggf or --vbf"
            fi
        else
            echo "Total significance"
            combine -M Significance -d $ws

            # combine -M MultiDimFit --algo singles -d $ws  --rMax 10  --rMin -10
        fi

    else
        echo "Expected significance"

        if [ $multisig = 1 ]; then
            
            if [ $vbf = 1 ]; then
                
                echo "VBF significance"
                combine -M Significance -d $ws -t -1 --redefineSignalPOIs r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            
            elif [ $ggf = 1 ]; then
                
                echo "ggF Significance"
                combine -M Significance -d $ws -t -1 --redefineSignalPOIs r_ggH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            else
                echo "must provide either --ggf or --vbf"
            fi
        else
            echo "Total significance"
            combine -M Significance -d $ws -m 125 --expectSignal=1 --rMin -10 --rMax 10 -t -1 
            #--setParameters mask_SR2=1,mask_SR3=1,mask_SR4=1
        fi
    fi
fi

if [ $bfit = 1 ]; then

    combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded

    echo "Fit Shapes"
    PostFitShapesFromWorkspace --dataset data_obs -w $ws --output FitShapesB.root -m 125 -f fitDiagnosticsUnblinded.root:fit_b --postfit --print
fi


if [ $dfit = 1 ]; then


    # # scan weird NP (if you need to mask a region: --setParameters mask_SR1=1)
    # combine -M MultiDimFit -d $ws -n test_ --algo grid -P r_qqH_hww -m 125 --rMin -10 --rMax 10 --setParameterRanges name=min,max[:r_qqH_hww=-6,6:] --points 100 --floatOtherPOIs 1

    # plot 1D scan
    # python3 CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py higgsCombinetest_.MultiDimFit.mH125.root -o output --POI r_qqH_hww 

    if [ $unfolding = 1 ]; then 
        combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_unfolding --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_mjj_1000_inf=1,r_otherH=1 --freezeParameters r_otherH
        # to print the rs
        combine -M MultiDimFit --algo singles -d $ws --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_mjj_1000_inf=1,r_otherH=1 --freezeParameters r_otherH
    else

        if [ $multisig = 1 ]; then
            if [ $vbf = 1 ]; then
                
                echo "VBF fit"
                combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_multiPOI_vbf --redefineSignalPOIs r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
                # PostFitShapesFromWorkspace --dataset data_obs -w $ws --output FitShapesS.root -m 125 -f fitDiagnosticsUnblinded.root:fit_s --postfit --print

            elif [ $ggf = 1 ]; then
                
                combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_multiPOI_ggf --redefineSignalPOIs r_ggH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
                # PostFitShapesFromWorkspace --dataset data_obs -w $ws --output FitShapesS.root -m 125 -f fitDiagnosticsUnblinded.root:fit_s --postfit --print

            else
                echo "Do not rely on the printed value of r"
                combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_multiPOI --redefineSignalPOIs r_ggH_hww,r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            fi
        else
            combine -M FitDiagnostics -d $ws -m 125 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_singlePOI        
        fi

    fi
fi

if [ $dfit_asimov = 1 ]; then


    if [ $unfolding = 1 ]; then 
        combine -M FitDiagnostics -d $ws -m 125 -t -1 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Unblinded_unfolding --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_mjj_1000_inf=1,r_otherH=1 --freezeParameters r_otherH
        # to print the rs
        combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_mjj_1000_inf=1,r_otherH=1 --freezeParameters r_otherH
    else

        if [ $multisig = 1 ]; then
            if [ $vbf = 1 ]; then
                
                echo "VBF fit"
                combine -M FitDiagnostics -d $ws -m 125 \
                -t -1 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Blinded_multiPOI_vbf --redefineSignalPOIs r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
                # PostFitShapesFromWorkspace --dataset data_obs -w $ws --output FitShapesS.root -m 125 -f fitDiagnosticsUnblinded.root:fit_s --postfit --print

            elif [ $ggf = 1 ]; then
                
                combine -M FitDiagnostics -d $ws -m 125 \
                -t -1 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Blinded_multiPOI_ggf --redefineSignalPOIs r_ggH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
                # PostFitShapesFromWorkspace --dataset data_obs -w $ws --output FitShapesS.root -m 125 -f fitDiagnosticsUnblinded.root:fit_s --postfit --print

            else
                echo "Do not rely on the printed value of r"
                combine -M FitDiagnostics -d $ws -m 125 \
                -t -1 --rMin -10 --rMax 10 --ignoreCovWarning --cminDefaultMinimizerStrategy 0 --saveWithUncertainties --saveOverallShapes --saveShapes --saveNormalizations -n Blinded_multiPOI --redefineSignalPOIs r_ggH_hww,r_qqH_hww --setParameters r_ggH_hww=1,r_qqH_hww=1,r_otherH=1 --freezeParameters r_otherH
            fi
        else
            echo "Fit Diagnostics Asimov"
            combine -M FitDiagnostics -m 125 -d $ws \
            -t -1 --expectSignal=1 --saveWorkspace -n Asimov --ignoreCovWarning \
            --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnosticsAsimov.txt
        fi

    fi



fi


if [ $impactsi = 1 ]; then

    echo "Initial fit for impacts"

    if [ $unblind = 1 ]; then
        echo Impacts unblinded
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doInitialFit --expectSignal 1
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50
        combineTool.py -M Impacts -d $ws --rMin -10 --rMax 10 -m 125 --robustFit 1 --output impacts.json --expectSignal 1
        # plotImpacts.py -i impacts.json -o impacts --blind
        plotImpacts.py -i impacts.json -o impacts


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

if [ $scanparam = 1 ]; then

    # recall unfolding POIs [r_ggH_pt200_300, r_ggH_pt300_450, r_ggH_pt450_inf, r_qqH_mjj_1000_inf]
    POI=r
    echo Will run NLL scan for $POI

    # # scan weird NP (if you need to mask a region: --setParameters mask_SR1=1)
    # combine -M MultiDimFit -d $ws -n test_ --algo grid -P $POI -m 125 --rMin -10 --rMax 10 --setParameterRanges name=min,max[:$POI=-6,6:] --points 100 --floatOtherPOIs 1
    # plot 1D scan
    # python3 CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py higgsCombinetest_.MultiDimFit.mH125.root -o output --POI $POI 

    # expected below
    # combine -M MultiDimFit --algo grid -m 125 -n "Scan" -d $ws --bypassFrequentistFit --toysFrequentist -t -1 --expectSignal 1 --rMin 0 --rMax 2 --floatParameters r

    # plot1DScan.py "" -o scan
    python3 CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py higgsCombineScan.MultiDimFit.mH125.root -o output --main-label Expected


fi

if [ $limits = 1 ]; then
    echo "Observed limits"
    combine -M AsymptoticLimits -d $ws -m 125 -n "" --rMax 10 --saveWorkspace --saveToys -s "$seed" --toysFrequentist 2>&1
fi