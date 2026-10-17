#!/usr/bin/env python3
"""Reproducible, public-safe FlyRank CTR review-prioritization analysis.

Downloads the public anonymized FlyRank starter dataset to /tmp (never into this
repository), evaluates an interpretable baseline and a random-forest model with
client-grouped out-of-fold predictions, and writes only aggregate outputs.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PAPER_ASSETS = ROOT / "paper" / "assets"
RAW_URL = "https://huggingface.co/datasets/FlyRank/internship-starter/resolve/main/content_refresh_anonymized.csv"
RAW_PATH = Path(os.environ.get("FLYRANK_RAW_PATH", "/tmp/flyrank_content_refresh_anonymized.csv"))
SEED = 42
MIN_IMPRESSIONS = 50
MIN_SEGMENT_PAGES = 100

NUMERIC_FEATURES = [
    "avg_position",
    "search_volume",
    "competition",
    "word_count",
    "content_age_days",
    "days_since_last_update",
]
CATEGORICAL_FEATURES = [
    "content_type",
    "main_intent",
    "competition_level",
    "age_tier",
    "freshness_tier",
    "position_tier",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def download_data() -> Path:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists() or RAW_PATH.stat().st_size < 1_000_000:
        print(f"Downloading public starter data to {RAW_PATH} (outside repository)...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    return RAW_PATH


def weighted_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=float)
    denom = float(w.sum())
    mae = float(np.sum(w * np.abs(y - p)) / denom)
    rmse = float(np.sqrt(np.sum(w * (y - p) ** 2) / denom))
    return {"weighted_mae_pp": mae, "weighted_rmse_pp": rmse}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    PAPER_ASSETS.mkdir(parents=True, exist_ok=True)
    raw_path = download_data()
    df = pd.read_csv(raw_path)
    required = set(FEATURES + ["client_id", "impressions_last_30d", "clicks_last_30d"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in the public starter release: {missing}")

    # Evaluate only pages with at least 50 impressions in the target window.
    # The target is same-window CTR in percent; this is a contemporaneous
    # review-prioritization task, not a future forecast.
    eligible = df.loc[
        df["impressions_last_30d"].ge(MIN_IMPRESSIONS)
        & df["impressions_last_30d"].gt(0)
        & df["clicks_last_30d"].notna()
    ].copy()
    eligible["actual_ctr_pct"] = (
        100.0 * eligible["clicks_last_30d"] / eligible["impressions_last_30d"]
    )
    eligible["actual_ctr_pct"] = eligible["actual_ctr_pct"].clip(0, 100)
    X = eligible[FEATURES].copy()
    y = eligible["actual_ctr_pct"].to_numpy(dtype=float)
    weights = eligible["impressions_last_30d"].to_numpy(dtype=float)
    groups = eligible["client_id"].astype(str).to_numpy()

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=12,
            max_features=0.8,
            n_jobs=-1,
            random_state=SEED,
        )),
    ])

    # Out-of-fold estimates: entire pseudonymized clients are held out together.
    splitter = GroupKFold(n_splits=5)
    oof_model = np.full(len(eligible), np.nan, dtype=float)
    oof_baseline = np.full(len(eligible), np.nan, dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train = eligible.iloc[train_idx]
        test = eligible.iloc[test_idx]
        train_y = y[train_idx]
        train_w = weights[train_idx]
        train_imp_cap = float(np.quantile(train_w, 0.95))
        fit_w = np.sqrt(np.minimum(train_w, train_imp_cap))
        model.fit(
            X.iloc[train_idx], train_y,
            model__sample_weight=fit_w,
        )
        oof_model[test_idx] = model.predict(X.iloc[test_idx])

        global_mean = float(np.average(train_y, weights=train_w))
        tier_means = {}
        for tier, idxs in train.groupby("position_tier", dropna=False).groups.items():
            loc = train.index.get_indexer(idxs)
            # Translate original DataFrame indices into positions within train_idx.
            # Using index lookup keeps this mapping stable even after filtering.
            tier_means[str(tier)] = float(np.average(train_y[loc], weights=train_w[loc]))
        oof_baseline[test_idx] = [
            tier_means.get(str(tier), global_mean)
            for tier in test["position_tier"].tolist()
        ]
        fold_rows.append({
            "fold": fold,
            "test_pages": int(len(test_idx)),
            "model": weighted_metrics(y[test_idx], oof_model[test_idx], weights[test_idx]),
            "position_baseline": weighted_metrics(y[test_idx], oof_baseline[test_idx], weights[test_idx]),
        })

    if np.isnan(oof_model).any() or np.isnan(oof_baseline).any():
        raise RuntimeError("Incomplete out-of-fold predictions")
    model_metrics = weighted_metrics(y, oof_model, weights)
    baseline_metrics = weighted_metrics(y, oof_baseline, weights)
    mae_improvement_pct = 100.0 * (baseline_metrics["weighted_mae_pp"] - model_metrics["weighted_mae_pp"]) / baseline_metrics["weighted_mae_pp"]

    # Ranked actionable cohorts are aggregate-only; hashed page/client IDs are
    # deliberately excluded. Positive prediction gaps are descriptive proxies,
    # not causal or guaranteed recoverable clicks.
    scored = eligible[["content_type", "main_intent", "position_tier"]].copy()
    scored["pages"] = 1
    scored["impressions"] = weights
    scored["actual_clicks"] = eligible["clicks_last_30d"].to_numpy(dtype=float)
    scored["expected_clicks"] = oof_model * weights / 100.0
    segment = (
        scored.groupby(["content_type", "main_intent", "position_tier"], dropna=False)
        .agg(
            pages=("pages", "sum"),
            impressions=("impressions", "sum"),
            actual_clicks=("actual_clicks", "sum"),
            expected_clicks=("expected_clicks", "sum"),
        )
        .reset_index()
    )
    # Compare cohort-level expected and observed rates. Only net-positive
    # aggregate gaps qualify as opportunities, preventing mixed-sign row gaps
    # from making a cohort look underperforming when it is not.
    segment["actual_ctr_pct"] = 100.0 * segment["actual_clicks"] / segment["impressions"]
    segment["mean_expected_ctr_pct"] = 100.0 * segment["expected_clicks"] / segment["impressions"]
    segment["positive_gap_click_proxy"] = np.maximum(
        segment["expected_clicks"] - segment["actual_clicks"], 0.0
    )
    segment = segment.loc[
        segment["pages"].ge(MIN_SEGMENT_PAGES)
        & segment["positive_gap_click_proxy"].gt(0)
    ].copy()
    segment = segment.sort_values(
        ["positive_gap_click_proxy", "impressions"], ascending=[False, False]
    ).head(10)
    recommendations = []
    for _, row in segment.iterrows():
        tier = str(row["position_tier"])
        if tier in {"top_3", "page_1"}:
            action = "Review snippet-to-intent alignment and title/description choices; protect visibility and validate any change with a controlled test."
        elif tier == "striking":
            action = "Review intent fit, snippet clarity, and competing result formats; prioritize diagnostic review before rewriting."
        else:
            action = "Check relevance, coverage, and technical/indexing fundamentals first; a CTR-only edit may not address weak visibility."
        recommendations.append({
            "content_type": str(row["content_type"]),
            "main_intent": str(row["main_intent"]),
            "position_tier": tier,
            "eligible_pages": int(row["pages"]),
            "impressions_last_30d": int(row["impressions"]),
            "observed_ctr_pct": float(row["actual_ctr_pct"]),
            "expected_ctr_pct": float(row["mean_expected_ctr_pct"]),
            "positive_gap_click_proxy": float(row["positive_gap_click_proxy"]),
            "action": action,
        })

    # Aggregate performance by position tier for a descriptive chart.
    position = eligible.assign(
        clicks=eligible["clicks_last_30d"].to_numpy(dtype=float),
        imps=weights,
    ).groupby("position_tier", dropna=False).agg(
        pages=("imps", "size"), impressions=("imps", "sum"), clicks=("clicks", "sum")
    ).reset_index()
    position["observed_ctr_pct"] = 100.0 * position["clicks"] / position["impressions"]
    pos_order = ["top_3", "page_1", "striking", "page_3_5", "deep"]
    position["_order"] = position["position_tier"].astype(str).map({v: i for i, v in enumerate(pos_order)}).fillna(99)
    position = position.sort_values("_order")

    # Charts are built from aggregated counts and metrics only.
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
    labs = ["Position-tier baseline", "Random forest"]
    vals = [baseline_metrics["weighted_mae_pp"], model_metrics["weighted_mae_pp"]]
    bars = ax.bar(labs, vals, color=["#9ca3af", "#2d6a6a"], width=0.58)
    ax.set_ylabel("Impression-weighted MAE (percentage points; lower is better)")
    ax.set_title("Grouped out-of-fold CTR estimation error")
    ax.grid(axis="y", alpha=0.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PAPER_ASSETS / "model-vs-baseline.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
    ax.bar(position["position_tier"].astype(str), position["observed_ctr_pct"], color="#2d6a6a")
    ax.set_ylabel("Observed CTR (%)")
    ax.set_xlabel("Average-position tier")
    ax.set_title("Observed last-30-day CTR by position tier")
    ax.grid(axis="y", alpha=0.2)
    for i, v in enumerate(position["observed_ctr_pct"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PAPER_ASSETS / "ctr-by-position.png", bbox_inches="tight")
    plt.close(fig)

    ranked = pd.DataFrame(recommendations).head(6)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=180)
    if len(ranked):
        labels = [
            f"{r['content_type']} · {r['main_intent']} · {r['position_tier'].replace('_', ' ')}"
            for r in recommendations[:6]
        ][::-1]
        values = ranked["positive_gap_click_proxy"].to_numpy()[::-1]
        ax.barh(labels, values, color="#d58f48")
        ax.set_xlabel("Positive model-minus-observed gap, click-equivalent proxy")
        ax.set_title("Largest aggregated review cohorts (100+ pages each)")
        ax.grid(axis="x", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No cohorts met the publication threshold", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(PAPER_ASSETS / "ranked-review-cohorts.png", bbox_inches="tight")
    plt.close(fig)

    overall_ctr = 100.0 * float(eligible["clicks_last_30d"].sum()) / float(eligible["impressions_last_30d"].sum())
    result = {
        "dataset": {
            "name": "FlyRank/internship-starter",
            "release": "Public anonymized starter dataset; 30,000 rows, 53 columns, 32 pseudonymized clients",
            "source_url": "https://huggingface.co/datasets/FlyRank/internship-starter",
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "client_count": int(df["client_id"].nunique()),
            "eligible_pages": int(len(eligible)),
            "eligibility_rule": f"at least {MIN_IMPRESSIONS} impressions in impressions_last_30d",
            "relative_windows": "Target uses last 30 days; the dataset also supplies a previous 30-day comparison and 90-day aggregates. The starter card does not provide absolute calendar dates for these windows.",
            "public_safety": "No titles, URLs, domains, private queries, client names, raw rows, or hashed IDs are included in published outputs.",
        },
        "analysis": {
            "question": "Can page/context features estimate last-30-day CTR more accurately than a position-tier-only baseline, and which aggregate cohorts merit review?",
            "overall_weighted_ctr_pct_eligible": overall_ctr,
            "min_impressions": MIN_IMPRESSIONS,
            "split": "5-fold GroupKFold; all rows for a pseudonymized client stay in one fold; random_state=42 for the random forest.",
            "features": FEATURES,
            "excluded_features": ["client_id", "content_id", "click/impression-derived rates, session metrics, trend fields, product flags, health scores, model/provider metadata, and all other outcome-window metrics"],
            "model": "RandomForestRegressor, 250 trees, min_samples_leaf=12, max_features=0.8; one-hot encoded categorical fields and median/mode imputation; training weights are square-root impression weights capped at the training-fold 95th percentile.",
            "baseline": "Training-fold, impression-weighted mean last-30-day CTR within position_tier, with training-fold global mean fallback for unseen tiers.",
            "model_weighted_mae_pp": model_metrics["weighted_mae_pp"],
            "baseline_weighted_mae_pp": baseline_metrics["weighted_mae_pp"],
            "model_weighted_rmse_pp": model_metrics["weighted_rmse_pp"],
            "baseline_weighted_rmse_pp": baseline_metrics["weighted_rmse_pp"],
            "mae_improvement_vs_baseline_pct": mae_improvement_pct,
            "fold_metrics": fold_rows,
            "positions": [
                {"position_tier": str(r.position_tier), "pages": int(r.pages), "impressions": int(r.impressions), "observed_ctr_pct": float(r.observed_ctr_pct)}
                for r in position.itertuples(index=False)
            ],
            "ranked_aggregate_recommendations": recommendations,
            "interpretation": "Contemporaneous, group-held-out decision support only; not a forecast, causal estimate, Google-ranking explanation, or guaranteed click uplift. Positive residual click-equivalents are descriptive proxies.",
        },
    }
    out = WORK / "results.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": result["dataset"]["row_count"],
        "eligible_pages": result["dataset"]["eligible_pages"],
        "clients": result["dataset"]["client_count"],
        "weighted_ctr_pct": round(overall_ctr, 4),
        "baseline_weighted_mae_pp": round(baseline_metrics["weighted_mae_pp"], 5),
        "model_weighted_mae_pp": round(model_metrics["weighted_mae_pp"], 5),
        "mae_improvement_pct": round(mae_improvement_pct, 2),
        "recommendation_cohorts": len(recommendations),
        "outputs": [str(out), str(PAPER_ASSETS)],
    }, indent=2))


if __name__ == "__main__":
    main()
