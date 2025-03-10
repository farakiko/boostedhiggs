#!/usr/bin/python

import argparse
import json
import os
import pickle as pkl
import time

import pandas as pd
import uproot
from coffea import nanoevents, processor

nanoevents.PFNanoAODSchema.warn_missing_crossrefs = False

import sys

sys.path.insert(0, "")
sys.path.append("boostedhiggs/LundReweighting")
sys.path.append("boostedhiggs/LundReweighting/utils")


def main(args):
    # make directory for output
    if not os.path.exists("./outfiles"):
        os.makedirs("./outfiles")

    channels = ["ele", "mu"]
    if args.channels:
        channels = args.channels.split(",")

    # if --macos is specified in args, process only the files provided
    if args.macos:
        files = {}
        # files[args.sample] = [f"rootfiles2/rootfiles/{args.sample}/file{i+1}.root" for i in range(1)]

        files[args.sample] = [
            "/eos/uscms/store/group/lpcpfnano/cmantill/v2_3/2017/HWWRivet/GluGluHToWW_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/GluGluHToWW_Pt-200ToInf_M-125/241219_000454/0000/nano_mc2017_4.root"  # noqa
        ]

    # if --local is specified in args, process only the args.sample provided
    elif args.local:
        files = {}
        with open(f"fileset/pfnanoindex_{args.pfnano}_{args.year}.json", "r") as f:
            files_all = json.load(f)
            for subdir in files_all[args.year]:
                for key, flist in files_all[args.year][subdir].items():
                    if key in args.sample:
                        files[key] = ["root://cmseos.fnal.gov/" + f for f in flist]

    else:
        # get samples
        if "metadata" in args.config:
            with open(args.config, "r") as f:
                files = json.load(f)
        else:
            if not args.config or not args.configkey:
                raise Exception("No config or configkey provided for condor jobs")

            # hopefully this step is avoided in condor jobs that have metadata.json
            from condor.file_utils import loadFiles

            files, _ = loadFiles(
                args.config,
                args.configkey,
                args.year,
                args.pfnano,
                args.sample.split(","),
            )

            # print(files)

    if not files:
        print("Did not find files.. Exiting.")
        exit(1)

    # build fileset with files to run per job
    fileset = {}
    starti = args.starti
    job_name = "/" + str(starti * args.n)
    if args.n != -1:
        job_name += "-" + str(args.starti * args.n + args.n)
    for sample, flist in files.items():
        if args.sample:
            if sample not in args.sample.split(","):
                continue
        if args.n != -1:
            fileset[sample] = flist[args.starti * args.n : args.starti * args.n + args.n]
        else:
            fileset[sample] = flist

    print(
        len(list(fileset.keys())),
        "Samples in fileset to be processed: ",
        list(fileset.keys()),
    )
    print(fileset)
    print(f"Number of files: {len(fileset[list(fileset.keys())[0]])}")

    # define processor
    year = args.year.replace("APV", "")
    yearmod = ""
    if "APV" in args.year:
        yearmod = "APV"

    if args.processor == "hww":
        from boostedhiggs.hwwprocessor import HwwProcessor

        p = HwwProcessor(
            year=year,
            yearmod=yearmod,
            channels=channels,
            inference=args.inference,
            systematics=args.systematics,
            getLPweights=args.getLPweights,
            uselooselep=args.uselooselep,
            fakevalidation=args.fakevalidation,
            output_location="./outfiles" + job_name,
            no_trigger=args.no_trigger,
            no_selection=args.no_selection,
        )

    elif args.processor == "vjetsnlo":
        from boostedhiggs.wjetsnloprocessor import VjetsProcessor

        p = VjetsProcessor(
            year=year,
            yearmod=yearmod,
            channels=channels,
            inference=args.inference,
            systematics=args.systematics,
            getLPweights=args.getLPweights,
            uselooselep=args.uselooselep,
            fakevalidation=args.fakevalidation,
            output_location="./outfiles" + job_name,
            no_trigger=args.no_trigger,
            no_selection=args.no_selection,
        )

    elif args.processor == "lumi":
        from boostedhiggs.lumi_processor import LumiProcessor

        p = LumiProcessor(year=args.year, output_location=f"./outfiles/{job_name}")

    elif args.processor == "input":
        # define processor
        from boostedhiggs.inputprocessor import InputProcessor

        assert args.inference is True, "enable --inference to run skimmer"
        p = InputProcessor(year=args.year, output_location=f"./outfiles/{job_name}")

    elif args.processor == "fakes":
        # define processor
        from boostedhiggs.fakesprocessor import FakesProcessor

        p = FakesProcessor(year=year, yearmod=yearmod, output_location=f"./outfiles/{job_name}")

    elif args.processor == "zll":
        # define processor
        from boostedhiggs.zllprocessor import ZllProcessor

        p = ZllProcessor(year=year, yearmod=yearmod, output_location=f"./outfiles/{job_name}")

    else:
        from boostedhiggs.trigger_efficiencies_processor1 import (
            TriggerEfficienciesProcessor,
        )

        p = TriggerEfficienciesProcessor(year=year, yearmod=yearmod, channels=channels)

    tic = time.time()
    if args.executor == "dask":
        from coffea.nanoevents import NanoeventsSchemaPlugin
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files="boostedhiggs",
        )
        client = Client(cluster)
        nanoevents_plugin = NanoeventsSchemaPlugin()
        client.register_worker_plugin(nanoevents_plugin)
        cluster.adapt(minimum=1, maximum=30)

        print("Waiting for at least one worker")
        client.wait_for_workers(1)

        # does treereduction help?
        executor = processor.DaskExecutor(status=True, client=client, treereduction=2)

    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        if args.executor == "futures":
            executor = processor.FuturesExecutor(status=True)
        else:
            executor = processor.IterativeExecutor(status=True)

    nanoevents.PFNanoAODSchema.mixins["SV"] = "PFCand"

    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
    )

    out, metrics = run(fileset, "Events", processor_instance=p)

    elapsed = time.time() - tic
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

    if args.processor == "input":
        # merge parquet
        data = pd.read_parquet(f"./outfiles/{job_name}/parquet")
        data.to_parquet(f"./outfiles/{job_name}.parquet")

        # remove unmerged parquet files
        os.system("rm -rf ./outfiles/" + job_name)

    else:
        # dump to pickle
        filehandler = open("./outfiles/" + job_name + ".pkl", "wb")
        pkl.dump(out, filehandler)
        filehandler.close()

        if args.processor == "lumi":
            data = pd.read_parquet("./outfiles/" + job_name + "/parquet")
            data.to_parquet("./outfiles/" + job_name + ".parquet")
            # remove old parquet files
            os.system("rm -rf ./outfiles/" + job_name)

        elif args.processor != "trigger":
            # merge parquet
            for ch in channels:
                data = pd.read_parquet("./outfiles/" + job_name + ch + "/parquet")
                data.to_parquet("./outfiles/" + job_name + "_" + ch + ".parquet")
                # remove old parquet files
                os.system("rm -rf ./outfiles/" + job_name + ch)


if __name__ == "__main__":
    # e.g.
    # noqa: python run.py --executor iterative --year 2017 --processor trigger --pfnano v2_2 --n 100 --starti 0 --sample SingleElectron_Run2017C --local --channels ele --config samples_inclusive.yaml --key data

    # noqa: python run.py --year 2017 --processor hww --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele --config samples_inclusive.yaml --key mc --executor iterative

    # noqa LP: python run.py --executor iterative --year 2017 --processor hww --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele,mu --config samples_inclusive.yaml --key mc --getLPweights --inference
    # noqa LP: python run.py --executor iterative --year 2017 --processor hww --pfnano v2_2 --sample TTToSemiLeptonic --local --channels ele,mu --config samples_inclusive.yaml --key mc --getLPweights --n 1 --starti 0

    # noqa LP: python run.py --executor iterative --year 2017 --processor hww --pfnano v2_2 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele,mu --config samples_inclusive.yaml --key mc --getLPweights --n 1 --starti 0

    # noqa Fakes: python run.py --year 2017 --processor fakes --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele,mu --config samples_inclusive.yaml --key mc

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--starti", dest="starti", default=0, help="start index of files", type=int)
    parser.add_argument("--n", dest="n", default=-1, help="number of files to process", type=int)
    parser.add_argument("--config", dest="config", default=None, help="path to datafiles", type=str)
    parser.add_argument(
        "--key",
        dest="configkey",
        default=None,
        help="config key: [data, mc, ... ]",
        type=str,
    )
    parser.add_argument("--sample", dest="sample", default=None, help="specify sample", type=str)
    parser.add_argument("--processor", dest="processor", required=True, help="processor", type=str)
    parser.add_argument(
        "--chunksize",
        dest="chunksize",
        default=10000,
        help="chunk size in processor",
        type=int,
    )
    parser.add_argument("--channels", dest="channels", default=None, help="channels separated by commas")

    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument(
        "--pfnano",
        dest="pfnano",
        type=str,
        default="v2_2",
        help="pfnano version",
    )
    parser.add_argument("--macos", dest="macos", action="store_true")
    parser.add_argument("--local", dest="local", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")
    parser.add_argument("--no-inference", dest="inference", action="store_false")
    parser.add_argument("--systematics", dest="systematics", action="store_true")
    parser.add_argument("--no-systematics", dest="systematics", action="store_false")
    parser.add_argument("--getLPweights", dest="getLPweights", action="store_true")
    parser.add_argument("--no-getLPweights", dest="getLPweights", action="store_false")

    # for fake estimation
    parser.add_argument("--uselooselep", dest="uselooselep", action="store_true")
    parser.add_argument("--no-uselooselep", dest="uselooselep", action="store_false")
    parser.add_argument("--fakevalidation", dest="fakevalidation", action="store_true")
    parser.add_argument("--no-fakevalidation", dest="fakevalidation", action="store_false")

    parser.add_argument("--no-trigger", dest="no_trigger", action="store_true")
    parser.add_argument("--trigger", dest="no_trigger", action="store_false")
    parser.add_argument("--no-selection", dest="no_selection", action="store_true")
    parser.add_argument("--selection", dest="no_selection", action="store_false")

    parser.set_defaults(inference=False)
    args = parser.parse_args()

    main(args)
