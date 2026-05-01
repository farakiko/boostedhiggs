"""
Generate variants of the v19 combined datacard with the W+jets ACCEPT lnN's
(QCDscale_wjets_ACCEPT_CMS_HWW_boosted, PDF_wjets_ACCEPT_CMS_HWW_boosted)
decorrelated across regions in different ways. Used to answer JHEP referee
comment (25): does the relative-acceptance freedom matter for sigma(r)?

Each output is a self-contained combined.txt placed next to the original so
that text2workspace + combine can be re-run without touching anything else.

Usage (from the repo root):
    python combine/decorrelate_wjets_accept.py \
        --input  combine/templates/v19/datacards/combined/combined.txt \
        --outdir combine/templates/v19/datacards/combined \
        --modes nominal sr_vs_cr per_sr per_region

After running, datacards land at <outdir>/combined_<mode>.txt. The 'nominal'
mode is just a verbatim copy for sanity (workspace + fit should reproduce the
baseline numbers).

Notes on the decorrelation modes:
  - sr_vs_cr   : 2 nuisances per ACCEPT — one shared by all SRs, one for the CR
  - per_sr     : 5 nuisances per ACCEPT — independent per SR (4) + 1 for the CR
  - per_region : 6 nuisances per ACCEPT — one per region
The CR-side nuisance always gets absorbed by the floating wjetsnormSF; the
SR-side ones provide independent acceptance freedom in each SR.
"""
import argparse
import os
import shutil
import re
from typing import List, Dict


ACCEPT_NUISANCES = [
    "QCDscale_wjets_ACCEPT_CMS_HWW_boosted",
    "PDF_wjets_ACCEPT_CMS_HWW_boosted",
]
WJETS_PROCESS = "wjets"
# Regions of the v19 datacard, in order they appear in the combineCards.py header line.
# These are the "labels" set by the SR<i>=... CR<i>=... arguments at combineCards time.
SR_LABELS = ["SR1", "SR2", "SR3", "SR4"]  # VBF, ggFpt250to350, ggFpt350to500, ggFpt500toInf
CR_LABELS = ["CR1", "CR2"]  # TopCR, WJetsCR
# We only ever decorrelate W+jets nuisances. TopCR contains W+jets (small) but its
# float is not coupled to wjetsnormSF, so we treat it as part of the "CR" bucket.


def parse_header(lines: List[str]):
    """Find the 'bin' and 'process' rows that map columns to (bin, process)."""
    bin_row = None
    proc_row = None
    rate_row_idx = None
    for i, line in enumerate(lines):
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "bin":
            # The first 'bin' row is the short one (one entry per bin name).
            # The second 'bin' row repeats per (bin, process) — that's the one we want.
            if bin_row is None:
                bin_row = (i, toks[1:])
            else:
                bin_row = (i, toks[1:])  # keep the long one
        elif toks[0] == "process" and proc_row is None:
            # First 'process' row contains string names (ggH_hww, wjets, ...).
            # Second 'process' row contains integer ids. We want the string row.
            if all(re.fullmatch(r"-?\d+", t) for t in toks[1:]):
                continue
            proc_row = (i, toks[1:])
        elif toks[0] == "rate":
            rate_row_idx = i
    if bin_row is None or proc_row is None:
        raise ValueError("Could not locate the long 'bin' and 'process' rows in the datacard")
    return bin_row, proc_row, rate_row_idx


def column_assignment(bin_row, proc_row) -> List[Dict[str, str]]:
    """Return a list of dicts {bin, process} aligned with the value columns."""
    bins = bin_row[1]
    procs = proc_row[1]
    assert len(bins) == len(procs), f"bin/process row length mismatch ({len(bins)} vs {len(procs)})"
    return [{"bin": b, "process": p} for b, p in zip(bins, procs)]


def find_lnn_line(lines: List[str], nuisance_name: str):
    """Return (index, parsed_columns_after_lnN_keyword)."""
    for i, line in enumerate(lines):
        toks = line.split()
        if not toks or toks[0] != nuisance_name:
            continue
        if len(toks) < 2 or toks[1] != "lnN":
            continue
        return i, toks
    raise ValueError(f"Could not find lnN line for {nuisance_name}")


def build_buckets(mode: str) -> Dict[str, List[str]]:
    """Map a bucket-name -> list of bin labels covered by that bucket."""
    if mode == "nominal":
        return {"all": SR_LABELS + CR_LABELS}
    if mode == "sr_vs_cr":
        return {"SR": SR_LABELS, "CR": CR_LABELS}
    if mode == "per_sr":
        out = {f"SR_{lab}": [lab] for lab in SR_LABELS}
        out["CR"] = CR_LABELS
        return out
    if mode == "per_region":
        out = {f"R_{lab}": [lab] for lab in (SR_LABELS + CR_LABELS)}
        return out
    raise ValueError(f"Unknown mode: {mode}")


def emit_decorrelated_lines(
    orig_tokens: List[str], col_meta: List[Dict[str, str]], nuisance_name: str, buckets: Dict[str, List[str]]
) -> List[str]:
    """Replace one lnN line with N lines, one per bucket, copying the original
    1.05 value (or whatever is set) only into the columns whose bin label is in
    that bucket and whose process == 'wjets'."""
    # Locate where the value columns start within orig_tokens.
    # tokens[0] == nuisance_name, tokens[1] == 'lnN', then values.
    value_tokens = orig_tokens[2:]
    if len(value_tokens) != len(col_meta):
        raise ValueError(f"Value count {len(value_tokens)} != column count {len(col_meta)}")

    new_lines = []
    for bucket_name, bin_labels in buckets.items():
        keep = ["-"] * len(col_meta)
        for j, meta in enumerate(col_meta):
            if meta["process"] == WJETS_PROCESS and meta["bin"] in bin_labels:
                keep[j] = value_tokens[j]
        if all(v == "-" for v in keep):
            continue  # skip empty bucket (nothing to decorrelate here)
        # Width-aligned print: use the same column widths as the original tokens
        # so the datacard stays readable.
        col_strs = [_format_value(v, value_tokens[j]) for j, v in enumerate(keep)]
        new_name = f"{nuisance_name}_{bucket_name}" if bucket_name != "all" else nuisance_name
        new_lines.append(f"{new_name}   lnN   " + "   ".join(col_strs))
    return new_lines


def _format_value(v: str, ref: str) -> str:
    """Pad v to roughly the width of ref so columns line up (cosmetic only)."""
    width = max(len(v), len(ref), 4)
    return v.ljust(width)


def rewrite(input_path: str, output_path: str, mode: str):
    with open(input_path) as f:
        lines = f.read().splitlines()

    bin_row, proc_row, _ = parse_header(lines)
    col_meta = column_assignment(bin_row, proc_row)
    buckets = build_buckets(mode)

    # Replace each ACCEPT lnN line in-place with the bucketed lines.
    out_lines = list(lines)
    # Process from bottom to top so indices don't shift.
    for nuis in ACCEPT_NUISANCES:
        idx, toks = find_lnn_line(out_lines, nuis)
        new = emit_decorrelated_lines(toks, col_meta, nuis, buckets)
        out_lines[idx : idx + 1] = new

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"  [{mode:10s}] wrote {output_path}  " f"({len(buckets)} bucket(s) per ACCEPT -> {2*len(buckets)} lnN lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the source combined.txt")
    ap.add_argument("--outdir", required=True, help="Directory to write combined_<mode>.txt files")
    ap.add_argument(
        "--modes", nargs="+", default=["nominal", "sr_vs_cr", "per_sr", "per_region"], help="Decorrelation modes to emit"
    )
    args = ap.parse_args()

    print(f"Source: {args.input}")
    print(f"Output: {args.outdir}")
    for m in args.modes:
        out = os.path.join(args.outdir, f"combined_{m}.txt")
        rewrite(args.input, out, m)


if __name__ == "__main__":
    main()
