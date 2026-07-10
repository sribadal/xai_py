"""
Single-run ConDOR model training.

Trains one ConDOR model with a patient-level train/val split, evaluates
on the validation sequences, and saves a checkpoint with model weights
and fitted encoders for downstream evaluation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xai_decision_making.experiments.qol_data_ablation_study import (
    ID_COL,
    build_sequences_from_merged,
    evaluate_sequences,
    load_merged,
    load_survival_model,
    precompute_mortality_risk_lookup,
    set_seed,
    train_one_fold,
)


def main():
    parser = argparse.ArgumentParser(description="Train a single ConDOR model.")
    parser.add_argument("--name", type=str, default="single_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--use-static-rdm", action="store_true")
    parser.add_argument("--use-static-tdm", action="store_true")
    parser.add_argument("--use-treatment", action="store_true")
    parser.add_argument("--use-ordinal", action="store_true")
    parser.add_argument("--use-mortality-risk", action="store_true")
    parser.add_argument("--survival-model-path", type=str, default=None)

    parser.add_argument("--pacap", type=str, default="data/dev_set_pacap.csv")
    parser.add_argument("--clinical", type=str, default="data/dev_set.csv")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path("artifacts") / "condor_models" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional mortality risk
    risk_lookup = None
    if args.use_mortality_risk:
        sm = load_survival_model(args.survival_model_path)
        risk_lookup = precompute_mortality_risk_lookup(sm)

    df = load_merged(args.pacap, args.clinical)

    # Patient-level train/val split
    rng = np.random.default_rng(args.seed)
    pids = df[ID_COL].dropna().unique()
    rng.shuffle(pids)

    n_val = max(1, int(len(pids) * args.val_frac))
    val_ids = set(pids[:n_val])
    train_ids = set(pids[n_val:])

    train_df = df[df[ID_COL].isin(train_ids)].copy()
    val_df = df[df[ID_COL].isin(val_ids)].copy()

    clin = pd.read_csv(args.clinical, low_memory=False)
    train_clin = clin[clin[ID_COL].isin(train_ids)].copy()
    val_clin = clin[clin[ID_COL].isin(val_ids)].copy()

    use_static_any = args.use_static_rdm or args.use_static_tdm

    # Fit encoders on training data only
    train_seqs, enc, static_dim = build_sequences_from_merged(
        train_df,
        train_clin,
        use_static=use_static_any,
        use_treatment=args.use_treatment,
        encoders=None,
        fit_encoders=True,
        risk_lookup=risk_lookup,
    )

    # Transform val with training encoders — no leakage
    val_seqs, _, _ = build_sequences_from_merged(
        val_df,
        val_clin,
        use_static=use_static_any,
        use_treatment=args.use_treatment,
        encoders=enc,
        fit_encoders=False,
        risk_lookup=risk_lookup,
    )

    print(
        f"Train sequences: {len(train_seqs)} | "
        f"Val sequences: {len(val_seqs)} | "
        f"static_dim={static_dim}"
    )

    if len(train_seqs) == 0 or len(val_seqs) == 0:
        raise RuntimeError(
            f"Empty sequences after split — "
            f"train={len(train_seqs)}, val={len(val_seqs)}. "
            "Try a smaller --val-frac or check your data paths."
        )

    models, conf = train_one_fold(
        fold_id=0,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        static_dim=static_dim,
        use_static_rdm=args.use_static_rdm,
        use_static_tdm=args.use_static_tdm,
        use_treatment=args.use_treatment,
        use_mortality_risk=args.use_mortality_risk,
        use_ordinal=args.use_ordinal,
        epochs=args.epochs,
        patience=args.patience,
    )

    metrics = evaluate_sequences(
        models=models,
        config=conf,
        sequences=val_seqs,
        use_static_rdm=args.use_static_rdm,
        use_static_tdm=args.use_static_tdm,
        use_treatment=args.use_treatment,
        use_mortality_risk=args.use_mortality_risk,
        use_ordinal=args.use_ordinal,
    )

    # Save checkpoint — encoders are saved so evaluate_condor.py can reuse them
    # and avoid static_dim mismatches at evaluation time.
    ckpt = {
        "ordinal": models[0].state_dict(),
        "rdm": models[1].state_dict(),
        "tdm": models[2].state_dict(),
        "config_args": {
            "static_dim": conf.static_dim,
            "use_static_rdm": args.use_static_rdm,
            "use_static_tdm": args.use_static_tdm,
            "use_treatment": args.use_treatment,
            "use_mortality_risk": args.use_mortality_risk,
        },
        "encoders": {
            "static_num_imputer": enc.num_imputer if enc is not None else None,
            "static_num_scaler": enc.num_scaler if enc is not None else None,
            "static_cat_imputer": enc.cat_imputer if enc is not None else None,
            "static_cat_encoder": enc.cat_encoder if enc is not None else None,
            "static_num_cols": enc.num_cols if enc is not None else [],
            "static_cat_cols": enc.cat_cols if enc is not None else [],
        },
        "metrics": metrics,
    }

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    print(f"\nSaved: {ckpt_path}")
    print("Val metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()