#!/usr/bin/env bash
# Run the W+jets ACCEPT decorrelation check on the v19 datacards using combine.
#
# Steps per mode (nominal, sr_vs_cr, per_sr, per_region):
#   1) text2workspace.py on combined_<mode>.txt
#   2) Asimov expected significance (combine -M Significance -t -1 --expectSignal=1)
#   3) Asimov expected upper limit  (combine -M AsymptoticLimits --run=expected -t -1)
# Outputs are written to combine/templates/v19/datacards/combined/decorrelation_check/
# and a summary table is printed at the end.
#
# Pre-requisites:
#   - combine + combineTool.py available in the current environment
#     (e.g. cmsenv inside a CMSSW area with HiggsAnalysis-CombinedLimit installed)
#   - the python script combine/decorrelate_wjets_accept.py has been run, OR
#     this script can run it for you (default).
#
# Usage:
#   ./combine/run_decorrelation_check.sh                 # run all modes
#   ./combine/run_decorrelation_check.sh nominal sr_vs_cr  # only these modes

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARDS_DIR="${REPO_ROOT}/combine/templates/v19/datacards"
COMB_DIR="${CARDS_DIR}/combined"
WORK_DIR="${COMB_DIR}/decorrelation_check"
SOURCE_CARD="${COMB_DIR}/combined.txt"

mkdir -p "${WORK_DIR}"

# ----------------------------------------------------------------------
# 1) Generate the decorrelated combined.txt files (idempotent)
# ----------------------------------------------------------------------
ALL_MODES=(nominal sr_vs_cr per_sr per_region)
MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then MODES=("${ALL_MODES[@]}"); fi

echo "=== Generating decorrelated datacards ==="
python3 "${REPO_ROOT}/combine/decorrelate_wjets_accept.py" \
    --input  "${SOURCE_CARD}" \
    --outdir "${WORK_DIR}" \
    --modes  "${MODES[@]}"

# combine needs to find testModel.root via the relative path embedded in the
# datacard ("templates/v19//datacards/testModel.root"). Easiest: cd to REPO_ROOT
# while running combine so those paths resolve.
cd "${REPO_ROOT}"

# ----------------------------------------------------------------------
# 2) For each mode: text2workspace + Significance + AsymptoticLimits
# ----------------------------------------------------------------------
declare -A SIG_RES
declare -A LIM_RES

for mode in "${MODES[@]}"; do
    DC="${WORK_DIR}/combined_${mode}.txt"
    WS="${WORK_DIR}/workspace_${mode}.root"
    LOG="${WORK_DIR}/log_${mode}.txt"
    echo
    echo "=== mode: ${mode} ==="
    echo "datacard:  ${DC}"
    echo "workspace: ${WS}"

    echo "-- text2workspace --"
    text2workspace.py "${DC}" --channel-masks -o "${WS}" 2>&1 | tee "${LOG}"

    echo "-- Significance (Asimov, expectSignal=1) --"
    combine -M Significance -d "${WS}" -m 125 --expectSignal=1 \
        --rMin -10 --rMax 10 -t -1 -n "_decorr_${mode}" 2>&1 | tee -a "${LOG}"
    SIG_RES[$mode]=$(grep -E "^Significance:" "${LOG}" | tail -1 | awk '{print $2}')

    echo "-- AsymptoticLimits (Asimov expected) --"
    combine -M AsymptoticLimits -d "${WS}" -m 125 --run=expected -t -1 \
        -n "_decorr_${mode}" 2>&1 | tee -a "${LOG}"
    LIM_RES[$mode]=$(grep -E "^Expected\s+50.0%:" "${LOG}" | tail -1 | awk '{print $5}')
done

# ----------------------------------------------------------------------
# 3) Summary
# ----------------------------------------------------------------------
echo
echo "=================================================================="
echo "  W+jets ACCEPT decorrelation check  (referee comment 25)"
echo "  Asimov dataset, expectSignal=1, v19 datacards"
echo "=================================================================="
printf "%-12s  %-20s  %-20s\n" "mode" "expected sig (sigma)" "expected upper limit"
printf "%-12s  %-20s  %-20s\n" "----" "--------------------" "--------------------"
for mode in "${MODES[@]}"; do
    printf "%-12s  %-20s  %-20s\n" "${mode}" "${SIG_RES[$mode]:-N/A}" "${LIM_RES[$mode]:-N/A}"
done
echo
echo "Logs:    ${WORK_DIR}/log_<mode>.txt"
echo "Cards:   ${WORK_DIR}/combined_<mode>.txt"
echo "Workspaces: ${WORK_DIR}/workspace_<mode>.root"
