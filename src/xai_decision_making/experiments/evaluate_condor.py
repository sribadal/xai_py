"""
Single-checkpoint evaluation for trained ConDOR models.

Loads a saved .pt checkpoint, reconstructs model weights and static
encoders, runs inference on dev/test/black-envelope sequences, and
reports RMSE and distributional metrics (Wasserstein, JSD).
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from xai_decision_making.models.synthesis.condor_core import ConDORConfig, OrdinalRegression, RDM, TDM
from xai_decision_making.experiments.qol_data_ablation_study import (
    ID_COL,
    StaticEncoderBundle,
    build_sequences_from_merged,
    evaluate_sequences,
    load_merged,
    load_survival_model,
    precompute_mortality_risk_lookup,
)

def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def _find_latest_checkpoint():
    candidates = sorted(Path("outputs").glob("study_*/fold_*_rmse_*.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    candidates = sorted(Path("artifacts").glob("condor_models/**/checkpoint.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "No checkpoint found. Looked in outputs/study_*/fold_*.pt and "
        "artifacts/condor_models/**/checkpoint.pt"
    )

def evaluate_distribution_metrics(real_vals, gen_vals):
    mse = np.mean((real_vals - gen_vals) ** 2)
    rmse = float(np.sqrt(mse))
    wd = float(wasserstein_distance(real_vals, gen_vals))

    min_val = float(min(real_vals.min(), gen_vals.min()))
    max_val = float(max(real_vals.max(), gen_vals.max()))
    bins = np.linspace(min_val, max_val, 20)

    p_real, _ = np.histogram(real_vals, bins=bins, density=True)
    p_gen, _ = np.histogram(gen_vals, bins=bins, density=True)
    p_real = p_real + 1e-10
    p_gen = p_gen + 1e-10
    jsd = float(jensenshannon(p_real, p_gen))

    return rmse, wd, jsd

def main():
    parser = argparse.ArgumentParser(description="Evaluate a ConDOR checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--eval-set",
        type=str,
        default="test",
        choices=["dev", "test", "black_envelope"],
    )
    parser.add_argument("--use-ordinal", action="store_true")
    parser.add_argument("--use-mortality-risk", action="store_true")
    parser.add_argument("--survival-model-path", type=str, default=None)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    tmp_conf = ConDORConfig(static_dim=0)
    ckpt = _torch_load(ckpt_path, tmp_conf.device)

    config_args = ckpt.get("config_args", {})
    conf = ConDORConfig(**config_args)

    ord_net = OrdinalRegression(conf).to(conf.device)
    rdm_net = RDM(conf).to(conf.device)
    tdm_net = TDM(conf).to(conf.device)

    if "ordinal" in ckpt and ckpt["ordinal"] is not None:
        ord_net.load_state_dict(ckpt["ordinal"])
    rdm_net.load_state_dict(ckpt["rdm"])
    tdm_net.load_state_dict(ckpt["tdm"])

    models = (ord_net, rdm_net, tdm_net)

    use_static_rdm = bool(config_args.get("use_static_rdm", False))
    use_static_tdm = bool(config_args.get("use_static_tdm", False))
    use_treatment = bool(config_args.get("use_treatment", False))
    use_mortality_risk = bool(config_args.get("use_mortality_risk", False))
    use_static_any = use_static_rdm or use_static_tdm

    risk_lookup = None
    if args.use_mortality_risk or use_mortality_risk:
        sm = load_survival_model(args.survival_model_path)
        risk_lookup = precompute_mortality_risk_lookup(sm)

    enc = None
    if use_static_any and "encoders" in ckpt:
        e = ckpt["encoders"]

        num_imputer = e.get("static_num_imputer") or e.get("num_imputer")
        num_scaler = e.get("static_num_scaler") or e.get("num_scaler")
        cat_imputer = e.get("static_cat_imputer") or e.get("cat_imputer")
        cat_encoder = e.get("static_cat_encoder") or e.get("cat_encoder")

        if num_imputer is not None and num_scaler is not None and cat_encoder is not None:
            saved_num_cols = e.get("static_num_cols") or e.get("num_cols")
            saved_cat_cols = e.get("static_cat_cols") or e.get("cat_cols")

            if not saved_num_cols:
                num_cols = list(getattr(num_imputer, "feature_names_in_", []))
                if not num_cols:
                    raise RuntimeError(
                        "Cannot recover numeric column names: checkpoint has no "
                        "saved num_cols and num_imputer.feature_names_in_ is empty."
                    )
            else:
                num_cols = list(saved_num_cols)

            if not saved_cat_cols and cat_imputer is not None:
                cat_cols = list(getattr(cat_imputer, "feature_names_in_", []))
            elif saved_cat_cols:
                cat_cols = list(saved_cat_cols)
            else:
                cat_cols = []

            enc = StaticEncoderBundle(
                num_imputer=num_imputer,
                num_scaler=num_scaler,
                cat_imputer=cat_imputer,
                cat_encoder=cat_encoder,
                num_cols=num_cols,
                cat_cols=cat_cols,
            )
            print(f"Loaded encoders from checkpoint (static_dim={conf.static_dim})")
            print(f"  num_cols: {num_cols}")
            print(f"  cat_cols: {cat_cols}")
        else:
            print("WARNING: Encoder objects missing from checkpoint — fitting fresh encoders on eval data.")
            print("         This may cause a static_dim mismatch if the eval set differs from training.")

    if args.eval_set == "dev":
        pacap = "data/dev_set_pacap.csv"
        clin = "data/dev_set.csv"
    elif args.eval_set == "black_envelope":
        pacap = "data/black_envelop_test_set_pacap.csv"
        clin = "data/black_envelop_test_set.csv"
        if not (os.path.exists(pacap) and os.path.exists(clin)):
            raise FileNotFoundError(
                "Black envelope files not found. Expected:\n"
                f"  {pacap}\n  {clin}"
            )
    else:
        pacap = "data/test_set_pacap.csv"
        clin = "data/test_set.csv"

    import pandas as pd

    df = load_merged(pacap, clin)
    clin_df = pd.read_csv(clin, low_memory=False)

    seqs, _, _ = build_sequences_from_merged(
        df,
        clin_df,
        use_static=use_static_any,
        use_treatment=use_treatment,
        encoders=enc,
        fit_encoders=(enc is None and use_static_any),
        risk_lookup=risk_lookup,
    )

    print(f"Eval set: {args.eval_set} | sequences={len(seqs)}")

    metrics = evaluate_sequences(
        models=models,
        config=conf,
        sequences=seqs,
        use_static_rdm=use_static_rdm,
        use_static_tdm=use_static_tdm,
        use_treatment=use_treatment,
        use_mortality_risk=use_mortality_risk,
        use_ordinal=args.use_ordinal,
    )

    print(f"\nCheckpoint: {ckpt_path}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    x_obs = (df["qol_summary"].values.astype(float) * 100.0).clip(0, 100)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(x_obs, fill=True, label="Observed QoL", alpha=0.3)
    plt.title("Observed QoL distribution (sanity check)")
    plt.xlabel("QoL Summary Score (0-100)")
    plt.legend()

    out_plot = Path("outputs") / "analysis_plots"
    out_plot.mkdir(parents=True, exist_ok=True)
    plot_path = out_plot / "qol_observed_distribution.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()