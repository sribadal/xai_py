"""
Backfill missing clinical fields into exported patient JSON files.
Links RSF exports via key_eid and ConDOR exports via key_nkr to the
PACAP test CSV, copying absent demographic, tumour, staging, treatment,
and lab fields into features_raw without overwriting existing values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

ENRICH_FIELDS = [
    "gesl",
    "leeft_cat",
    "cci_cat",
    "morf_cat",
    "diffgrad",
    "topo_sublok",
    "stadium",
    "ct",
    "cn",
    "cm",
    "perf_stat",
    "asa",
    "veneus_vaatbetr",
    "arterieel_vaatbetr",
    "ca199",
    "bilirubine",
    "meta_lever",
    "meta_long",
    "meta_perit",
    "meta_bot",
    "meta_lymf",
    "meta_hersenen",
    "meta_bijnier",
    "meta_overig",
    "chir",
    "chemo",
    "rt",
    "destr",
    "diag_basis",
]

LOOKUP_KEY = {
    "rsf": "key_eid",
    "condor": "key_nkr",
}


def build_lookups(pacap_csv):
    key_cols = list(LOOKUP_KEY.values())
    need_cols = key_cols + ENRICH_FIELDS
    header = pd.read_csv(pacap_csv, nrows=0).columns.tolist()
    usecols = [c for c in need_cols if c in header]
    available_fields = [c for c in ENRICH_FIELDS if c in header]
    df = pd.read_csv(pacap_csv, usecols=usecols, low_memory=False)

    def _clean_val(val):
        if pd.isna(val):
            return None
        if isinstance(val, float) and val == int(val):
            return int(val)
        return val

    lookups = {}

    for model_type, key_col in LOOKUP_KEY.items():
        if key_col not in df.columns:
            lookups[model_type] = {}
            continue

        sub = df.dropna(subset=[key_col]).copy()
        sub[key_col] = sub[key_col].astype("int64")

        lookup = {}
        for _, row in sub.iterrows():
            pid = int(row[key_col])
            fields = {
                col: _clean_val(row[col])
                for col in available_fields
                if col in row.index
            }
            lookup[pid] = fields

        lookups[model_type] = lookup

    return lookups


def enrich_patient_json(fpath, lookups, dry_run):
    with open(fpath) as f:
        record = json.load(f)

    pid = record.get("patient_id")
    if pid is None:
        return "skip:no-pid"

    model_type = record.get("model_type", "rsf")
    lookup = lookups.get(model_type)
    if lookup is None:
        return f"skip:unknown-model-type-{model_type}"

    try:
        pid_int = int(pid)
    except (ValueError, TypeError):
        return "skip:non-int-pid"

    if pid_int not in lookup:
        return "skip:not-in-pacap"

    features_raw = record.get("features_raw")
    if features_raw is None:
        features_raw = {}
        record["features_raw"] = features_raw

    enrichment = lookup[pid_int]
    added = []
    for field, value in enrichment.items():
        if field not in features_raw:
            features_raw[field] = value
            added.append(field)

    if not added:
        return "skip:already-complete"

    if dry_run:
        return f"dry-run:would-add={added}"

    with open(fpath, "w") as f:
        json.dump(record, f, indent=2)

    return f"enriched:{added}"


def process_directory(exports_dir, lookups, dry_run, label):
    if not os.path.isdir(exports_dir):
        return {}

    json_files = sorted(
        f
        for f in os.listdir(exports_dir)
        if f.startswith("patient_") and f.endswith(".json")
    )

    stats = {}
    for fname in json_files:
        fpath = os.path.join(exports_dir, fname)
        result = enrich_patient_json(fpath, lookups, dry_run)
        tag = result if result.startswith("skip") else result.split(":")[0]
        stats[tag] = stats.get(tag, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["rsf", "condor", "both"],
        default="both",
    )
    parser.add_argument(
        "--pacap-csv",
        type=str,
        default=os.path.join(ROOT, "data", "test_set_pacap.csv"),
    )
    parser.add_argument(
        "--rsf-dir",
        type=str,
        default=os.path.join(ROOT, "exports", "rsf"),
    )
    parser.add_argument(
        "--condor-dir",
        type=str,
        default=os.path.join(ROOT, "exports", "condor"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lookups = build_lookups(args.pacap_csv)

    targets = []
    if args.model_type in ("rsf", "both"):
        targets.append(("rsf", args.rsf_dir))
    if args.model_type in ("condor", "both"):
        targets.append(("condor", args.condor_dir))

    for label, exports_dir in targets:
        process_directory(exports_dir, lookups, args.dry_run, label)


if __name__ == "__main__":
    main()