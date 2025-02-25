#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantilla, Raghav Kansal, Farouk Mokhtar
"""
import argparse
import json
import os
from math import ceil

from file_utils import loadFiles


def main(args):
    try:
        proxy = os.environ["X509_USER_PROXY"]
    except ValueError:
        print("No valid proxy. Exiting.")
        exit(1)

    years = args.years.split(",")

    for year in years:

        locdir = "condor/" + args.tag + "_" + year
        username = os.environ["USER"]
        homedir = f"/store/user/{username}/boostedhiggs/"
        outdir = homedir + args.tag + "_" + year + "/"

        # make local directory
        logdir = locdir + "/logs"
        os.system(f"mkdir -p {logdir}")

        # copy the splitting file to the locdir
        os.system(f"cp pfnano_splitting.yaml {locdir}")
        os.system(f"cp {args.config} {locdir}")

        # and condor directory
        print("CONDOR work dir: " + outdir)
        os.system(f"mkdir -p /eos/uscms/{outdir}")

        # build metadata.json with samples
        slist = args.slist.split(",") if args.slist is not None else None
        files, nfiles_per_job = loadFiles(args.config, args.configkey, year, args.pfnano, slist)
        metadata_file = f"metadata_{args.configkey}.json"
        with open(f"{locdir}/{metadata_file}", "w") as f:
            json.dump(files, f, sort_keys=True, indent=2)
        print(files.keys())

        # submit a cluster of jobs per sample
        for sample in files.keys():
            print(f"Making directory /eos/uscms/{outdir}/{sample}")
            os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

            localcondor = f"{locdir}/{sample}.jdl"
            localsh = f"{locdir}/{sample}.sh"
            try:
                os.remove(localcondor)
                os.remove(localsh)
                os.remove(f"{locdir}/*.log")
            except Exception:
                pass

            tot_files = len(files[sample])
            if (args.maxfiles != -1) & (args.maxfiles < tot_files):
                tot_files = args.maxfiles

            if args.files_per_job:
                njobs = ceil(tot_files / args.files_per_job)
                files_per_job = str(args.files_per_job)
            else:
                njobs = ceil(tot_files / nfiles_per_job[sample])
                files_per_job = str(nfiles_per_job[sample])

            # make submit.txt with number of jobs
            if args.test:
                njobs = 1
            jobids = [str(jobid) for jobid in range(njobs)]
            jobids_file = os.path.join(locdir, f"submit_{sample}.txt")
            with open(jobids_file, "w") as f:
                f.write("\n".join(jobids))

            # make condor file
            condor_templ_file = open("condor/submit.templ.jdl")
            condor_file = open(localcondor, "w")
            for line in condor_templ_file:
                line = line.replace("DIRECTORY", locdir)
                line = line.replace("PREFIX", sample)
                line = line.replace("JOBIDS_FILE", jobids_file)
                line = line.replace("METADATAFILE", metadata_file)
                line = line.replace("PROXY", proxy)
                condor_file.write(line)
            condor_file.close()
            condor_templ_file.close()

            # make executable file
            sh_templ_file = open("condor/submit.templ.sh")
            eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"
            eosoutput_pkl = f"{eosoutput_dir}/"
            sh_file = open(localsh, "w")
            for line in sh_templ_file:
                line = line.replace("SCRIPTNAME", args.script)
                line = line.replace("YEAR", year)
                line = line.replace("PROCESSOR", args.processor)
                line = line.replace("METADATAFILE", metadata_file)
                line = line.replace("NUMJOBS", files_per_job)
                line = line.replace("SAMPLE", sample)
                line = line.replace("CHANNELS", args.channels)
                line = line.replace("EOSOUTPKL", eosoutput_pkl)
                line = line.replace("PFNANO", f"--pfnano {args.pfnano}")
                if args.inference:
                    line = line.replace("INFERENCE", "--inference")
                else:
                    line = line.replace("INFERENCE", "--no-inference")
                if args.systematics:
                    line = line.replace("SYSTEMATICS", "--systematics")
                else:
                    line = line.replace("SYSTEMATICS", "--no-systematics")
                if args.getLPweights:
                    line = line.replace("GETLPWEIGHTS", "--getLPweights")
                else:
                    line = line.replace("GETLPWEIGHTS", "--no-getLPweights")

                if args.uselooselep:
                    line = line.replace("LOOSELEP", "--uselooselep")
                else:
                    line = line.replace("LOOSELEP", "--no-uselooselep")

                if args.no_trigger:
                    line = line.replace("TRIGGER", "--no-trigger")
                else:
                    line = line.replace("TRIGGER", "--trigger")

                if args.no_selection:
                    line = line.replace("PRESELECTION", "--no-selection")
                else:
                    line = line.replace("PRESELECTION", "--selection")

                line = line.replace("LABEL", args.label)
                line = line.replace("REGION", args.region)

                sh_file.write(line)
            sh_file.close()
            sh_templ_file.close()

            os.system(f"chmod u+x {localsh}")
            if os.path.exists("%s.log" % localcondor):
                os.system("rm %s.log" % localcondor)

            # submit
            if args.submit:
                print("Submit ", localcondor)
                os.system("condor_submit %s" % localcondor)


if __name__ == "__main__":
    """
    # noqa: python condor/submit.py --years 2018,2017,2016,2016APV --tag test --config samples_inclusive.yaml --key mc --no-inference --channels mu,ele
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", dest="script", default="run.py", help="script to run", type=str)
    parser.add_argument("--years", dest="years", required=True, help="years separated by commas")
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--processor",
        dest="processor",
        default="hww",
        help="which processor",
        type=str,
        choices=["hww", "trigger", "lumi", "vh", "zll", "input", "fakes"],
    )
    parser.add_argument("--config", dest="config", required=True, help="path to config yaml", type=str)
    parser.add_argument("--key", dest="configkey", required=True, help="config key: [data, mc, ... ]", type=str)
    parser.add_argument("--slist", dest="slist", default=None, help="give sample list separated by commas")
    parser.add_argument("--test", dest="test", action="store_true", help="only 2 jobs per sample will be created")
    parser.add_argument("--submit", dest="submit", action="store_true", help="submit jobs when created")
    parser.add_argument("--files-per-job", default=None, help="# files per condor job", type=int)
    parser.add_argument("--channels", dest="channels", required=True, help="channels separated by commas")
    parser.add_argument("--pfnano", dest="pfnano", type=str, default="v2_2", help="pfnano version")
    parser.add_argument("--inference", dest="inference", action="store_true")
    parser.add_argument("--no-inference", dest="inference", action="store_false")
    parser.add_argument("--systematics", dest="systematics", action="store_true")
    parser.add_argument("--no-systematics", dest="systematics", action="store_false")
    parser.add_argument("--getLPweights", dest="getLPweights", action="store_true")
    parser.add_argument("--no-getLPweights", dest="getLPweights", action="store_false")
    parser.add_argument("--label", dest="label", default="H", help="jet label for inputskimmer", type=str)
    parser.add_argument("--region", dest="region", default="signal", help="specify region for selections", type=str)
    parser.add_argument("--maxfiles", default=-1, help="max number of files to run on", type=int)
    parser.add_argument("--uselooselep", dest="uselooselep", action="store_true")
    parser.add_argument("--no-uselooselep", dest="uselooselep", action="store_false")

    parser.set_defaults(inference=True)
    args = parser.parse_args()

    main(args)
