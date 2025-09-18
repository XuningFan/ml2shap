#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_process_with_shap_and_metrics.py
- CPU threads cap (<=10 by default), single-GPU preference (if available), no nested parallelism
- SHAP explanations (summary/bar/dependence + numeric importance CSV)
- Per-model metrics export: AUC (95% CI via bootstrap), Accuracy, F1, PPV, NPV, Sensitivity, Specificity
"""

import os
import sys
import json
from datetime import datetime
import argparse
import logging
from contextlib import contextmanager
from typing import Dict, Tuple, Any, List

# =========================
# Global resource limits (set before heavy imports)
# =========================
DEFAULT_CPU_LIMIT = 10

def _coerce_cpu_limit(val: str) -> int:
    try:
        n = int(val)
    except Exception:
        n = DEFAULT_CPU_LIMIT
    n = max(1, min(n, os.cpu_count() or 1))
    return n

CPU_LIMIT = _coerce_cpu_limit(os.environ.get("CPU_LIMIT", str(DEFAULT_CPU_LIMIT)))

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_LIMIT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_LIMIT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_LIMIT))

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
)
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from joblib import Parallel, delayed

# SHAP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# === Added for imbalance handling (SMOTETomek) ===
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as e:
    raise ImportError(
        "需要安装 imbalanced-learn 才能使用 --use-smotetomek。\n"
        "请先运行：pip install -U imbalanced-learn"
    ) from e


try:
    from threadpoolctl import threadpool_limits
except Exception:
    def threadpool_limits(*args, **kwargs):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _N()

HAVE_XGB = False
HAVE_LGBM = False
HAVE_CAT = False

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    pass

# =========================
# Logging
# =========================
logger = logging.getLogger("batch")
handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# === Added: global toggle for SMOTETomek ===
GLOBAL_USE_SMOTETOMEK = False
# =========================
# Device and thread helpers
# =========================
def have_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())

@contextmanager
def limit_threads(n: int):
    with threadpool_limits(limits=n):
        yield

def _uses_gpu(model: Any) -> bool:
    """Heuristic: detect if the estimator is configured to use GPU."""
    try:
        from xgboost import XGBClassifier as _XGB
    except Exception:
        _XGB = tuple()
    try:
        from lightgbm import LGBMClassifier as _LGBM
    except Exception:
        _LGBM = tuple()
    try:
        from catboost import CatBoostClassifier as _CAT
    except Exception:
        _CAT = tuple()

    est = model
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import StackingClassifier
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    if isinstance(est, StackingClassifier):
        est = est.final_estimator

    try:
        if _XGB and isinstance(est, _XGB):
            tm = getattr(est, "tree_method", None)
            pred = getattr(est, "predictor", None)
            if (tm and "gpu" in str(tm)) or (pred and "gpu" in str(pred)):
                return True
        if _LGBM and isinstance(est, _LGBM):
            dev = getattr(est, "device", None)
            if dev and "gpu" in str(dev).lower():
                return True
        if _CAT and isinstance(est, _CAT):
            tt = getattr(est, "task_type", None)
            if tt and "gpu" in str(tt).upper():
                return True
    except Exception:
        pass
    return False

# =========================
# Models and grids
# =========================
def model_param_init(threads: int, use_gpu: bool) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[Any]]]]:
    models: Dict[str, Any] = {}
    param_grids: Dict[str, Dict[str, List[Any]]] = {}

    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])
    param_grids["Logistic Regression"] = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"]
    }

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1
    )
    param_grids["Random Forest"] = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    if HAVE_XGB:
        xgb_extra = {}
        if use_gpu:
            xgb_extra.update(dict(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0))
        else:
            xgb_extra.update(dict(n_jobs=1))
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=42, **xgb_extra
        )
        param_grids["XGBoost"] = {
            "n_estimators": [50, 100],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    else:
        logger.info("xgboost not found, skipping XGBoost.")

    if HAVE_LGBM:
        lgb_extra = dict(n_jobs=1)
        if use_gpu:
            lgb_extra.update(dict(device="gpu", gpu_platform_id=0, gpu_device_id=0))
        models["LightGBM"] = LGBMClassifier(verbose=-1, random_state=42, **lgb_extra)
        param_grids["LightGBM"] = {
            "n_estimators": [50, 100],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.1]
        }
    else:
        logger.info("lightgbm not found, skipping LightGBM.")

    if HAVE_CAT:
        cat_extra = {}
        if use_gpu:
            cat_extra.update(dict(task_type="GPU", devices="0", gpu_ram_part=0.5, border_count=32))
        else:
            cat_extra.update(dict(thread_count=1))
        models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=42, **cat_extra)
        param_grids["CatBoost"] = {
            "iterations": [100, 200],
            "depth": [4, 6],
            "learning_rate": [0.01, 0.1]
        }
    else:
        logger.info("catboost not found, skipping CatBoost.")

    models["MLP"] = MLPClassifier(random_state=42)
    param_grids["MLP"] = {
        "hidden_layer_sizes": [(100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01]
    }

    models["GaussianNB"] = GaussianNB()
    param_grids["GaussianNB"] = {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    }

    models["AdaBoost"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        random_state=42
    )
    param_grids["AdaBoost"] = {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 1.0]
    }

    base_estimators = []
    base_estimators.append(("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)))
    base_estimators.append(("lr", LogisticRegression(max_iter=2000)))
    if HAVE_LGBM:
        lgb_extra = dict(n_jobs=1)
        if use_gpu:
            lgb_extra.update(dict(device="gpu", gpu_platform_id=0, gpu_device_id=0))
        base_estimators.append(("lgb", LGBMClassifier(**lgb_extra)))

    models["Stacking"] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
        n_jobs=1
    )
    param_grids["Stacking"] = {
        "final_estimator__C": [0.1, 1.0, 10.0]
    }

    return models, param_grids

# =========================
# SHAP helpers
# =========================
def _is_tree_model(model) -> bool:
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = tuple()
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = tuple()
    try:
        from catboost import CatBoostClassifier
    except Exception:
        CatBoostClassifier = tuple()

    est = model
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    if isinstance(est, StackingClassifier):
        est = est.final_estimator

    return isinstance(est, (RandomForestClassifier, AdaBoostClassifier,)) \
        or (XGBClassifier and isinstance(est, XGBClassifier)) \
        or (LGBMClassifier and isinstance(est, LGBMClassifier)) \
        or (CatBoostClassifier and isinstance(est, CatBoostClassifier))

def _summarize_shap_importance(shap_values, feature_names):
    import numpy as _np
    import pandas as _pd
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    if isinstance(vals, list):
        per_class = []
        for arr in vals:
            per_class.append(_np.mean(_np.abs(arr), axis=0))
        per_class = _np.vstack(per_class).T
        overall = _np.mean(per_class, axis=1)
        df = _pd.DataFrame({"feature": feature_names, "mean_abs_shap": overall})
        for k in range(per_class.shape[1]):
            df[f"mean_abs_shap_class_{k}"] = per_class[:, k]
        df["rank"] = _np.argsort(-df["mean_abs_shap"].values) + 1
        df = df.sort_values("rank")
        return df

    vals = _np.array(vals)
    if vals.ndim == 3:
        per_class = _np.mean(_np.abs(vals), axis=0)
        overall = _np.mean(per_class, axis=1)
        df = _pd.DataFrame({"feature": feature_names, "mean_abs_shap": overall})
        for k in range(per_class.shape[1]):
            df[f"mean_abs_shap_class_{k}"] = per_class[:, k]
        df["rank"] = _np.argsort(-df["mean_abs_shap"].values) + 1
        df = df.sort_values("rank")
        return df
    elif vals.ndim == 2:
        overall = _np.mean(_np.abs(vals), axis=0)
        df = _pd.DataFrame({"feature": feature_names, "mean_abs_shap": overall})
        df["rank"] = _np.argsort(-df["mean_abs_shap"].values) + 1
        df = df.sort_values("rank")
        return df
    else:
        overall = _np.mean(_np.abs(vals), axis=0).reshape(-1)
        df = _pd.DataFrame({"feature": feature_names, "mean_abs_shap": overall})
        df["rank"] = _np.argsort(-df["mean_abs_shap"].values) + 1
        df = df.sort_values("rank")
        return df

def shap_explain_and_save(model, X_train, X_test, feature_names, outdir, model_name, max_bg=1000, max_show=20):
    os.makedirs(outdir, exist_ok=True)

    X_bg = shap.utils.sample(X_train, min(max_bg, X_train.shape[0]), random_state=42)
    X_explain = shap.utils.sample(X_test, min(2000, X_test.shape[0]), random_state=42)

    est = model
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import StackingClassifier
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    if isinstance(est, StackingClassifier):
        est = est.final_estimator

    explainer = None
    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        elif isinstance(est, (LogisticRegression,)):
            explainer = shap.LinearExplainer(est, X_bg, feature_perturbation="interventional")
        else:
            pred_fn = (lambda data: model.predict_proba(data) if hasattr(model, "predict_proba")
                       else model.decision_function(data))
            explainer = shap.KernelExplainer(pred_fn, shap.sample(X_bg, min(200, X_bg.shape[0])))
    except Exception as e:
        logger.warning("Failed to create SHAP explainer: %s", e)
        return

    try:
        shap_values = explainer(X_explain, check_additivity=False)
    except Exception:
        shap_values = explainer.shap_values(X_explain)

    try:
        shap_values.feature_names = list(feature_names)
    except Exception:
        pass

    # Export numeric importance
    try:
        imp_df = _summarize_shap_importance(shap_values, list(feature_names))
        csv_path = os.path.join(outdir, f"{model_name}_shap_importance.csv")
        imp_df.to_csv(csv_path, index=False)
        top_json = os.path.join(outdir, f"{model_name}_shap_importance_top20.json")
        imp_top = imp_df.head(20).to_dict(orient="records")
        with open(top_json, "w", encoding="utf-8") as f:
            json.dump(imp_top, f, ensure_ascii=False, indent=2)
        logger.info("Exported SHAP numeric importance: %s", csv_path)
    except Exception as e:
        logger.warning("Failed to export SHAP importance: %s", e)

    # Plots
    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, max_display=max_show)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_shap_summary.png"), dpi=180)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, plot_type="bar", max_display=max_show)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_shap_bar.png"), dpi=180)
    plt.close()

    # SHAP heatmap
    try:
        plt.figure()
        shap.plots.heatmap(shap_values, max_display=max_show, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{model_name}_shap_heatmap.png"), dpi=180)
        plt.close()
    except Exception as _e:
        logger.warning("SHAP heatmap failed for %s: %s", model_name, _e)

    try:
        vals = shap.utils.abs_mean(shap_values, axis=0)
        order = np.argsort(vals)[::-1][:min(6, X_explain.shape[1])]
    except Exception:
        order = np.arange(min(6, X_explain.shape[1]))

    for idx in order:
        feat = feature_names[idx] if feature_names is not None else f"f{idx}"
        plt.figure()
        shap.dependence_plot(idx,
                             shap_values.values if hasattr(shap_values, "values") else shap_values,
                             X_explain, feature_names=feature_names, show=False, interaction_index=None)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{model_name}_shap_dependence_{feat}.png"), dpi=180)
        plt.close()

    logger.info("Saved SHAP plots and importance to: %s", outdir)

# =========================
# Metrics helpers
# =========================
def _binary_confusion_counts(y_true, y_pred):
    # ensure binary labels 0/1
    labels = np.unique(y_true)
    if len(labels) != 2:
        raise ValueError("Binary metrics requested but found {} classes: {}".format(len(labels), labels))
    # map labels so that positive = 1
    if set(labels) != {0,1}:
        mapping = {labels[0]:0, labels[1]:1}
        y_true = np.vectorize(mapping.get)(y_true)
        y_pred = np.vectorize(mapping.get)(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def _binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)
    eps = 1e-12
    acc = (tp + tn) / max(tp+tn+fp+fn, eps)
    sens = tp / max(tp + fn, eps)        # recall, TPR
    spec = tn / max(tn + fp, eps)        # TNR
    ppv  = tp / max(tp + fp, eps)        # precision
    npv  = tn / max(tn + fn, eps)
    f1   = 2 * ppv * sens / max(ppv + sens, eps)
    return acc, f1, ppv, npv, sens, spec

def _bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.05, random_state=42):
    rng = np.random.RandomState(random_state)
    # ensure binary labels 0/1 and score is for positive class
    labels = np.unique(y_true)
    if len(labels) != 2:
        # For multi-class, compute macro-ovr AUC and bootstrap that
        auroc = roc_auc_score(y_true, y_score, multi_class="ovr")
        idx = np.arange(len(y_true))
        boots = []
        for _ in range(n_boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            try:
                boots.append(roc_auc_score(y_true[b], y_score[b], multi_class="ovr"))
            except Exception:
                continue
        if len(boots) == 0:
            return auroc, np.nan, np.nan
        lo = np.percentile(boots, 100*alpha/2)
        hi = np.percentile(boots, 100*(1-alpha/2))
        return auroc, lo, hi

    # Binary case
    # If y_score has shape (n_samples,2) -> use positive column
    score = y_score
    if score.ndim == 2 and score.shape[1] >= 2:
        score = score[:,1]
    auroc = roc_auc_score(y_true, score)
    idx = np.arange(len(y_true))
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            boots.append(roc_auc_score(y_true[b], score[b]))
        except Exception:
            continue
    if len(boots) == 0:
        return auroc, np.nan, np.nan
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return auroc, lo, hi

def _optimal_roc_point(y_true, y_score):
    """Return (fpr*, tpr*, thr*) maximizing Youden's J (TPR - FPR)."""
    from sklearn.metrics import roc_curve
    import numpy as _np
    score = y_score
    if score.ndim == 2 and score.shape[1] >= 2:
        score = score[:, 1]
    fpr, tpr, thr = roc_curve(y_true, score)
    j = tpr - fpr
    k = int(_np.nanargmax(j))
    return float(fpr[k]), float(tpr[k]), float(thr[k])

def plot_roc_with_ci(y_true, y_score, model_name, outdir, ci_tuple=None):
    """
    Plot ROC curve for a single model with AUC and 95% CI annotation and optimal point (Youden's J).
    Saves to <outdir>/<model_name>_roc.png
    """
    import os
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.metrics import roc_curve, roc_auc_score
    os.makedirs(outdir, exist_ok=True)
    
    ROC_COLOR_PALETTE = [
    "#5566BE",  # 蓝
    "#C85365",  # 绯红
    "#E0B7B5",  # 浅橙粉
    "#DFCDBC",  # 米杏
    "#7CADCD",  # 浅天蓝
    "#B4AFD1",  # 薰衣草紫
    ]
    score = y_score
    if score is None:
        return
    if hasattr(score, "ndim") and score.ndim == 2 and score.shape[1] >= 2:
        score = score[:, 1]

    fpr, tpr, thr = roc_curve(y_true, score)
    auc_val = roc_auc_score(y_true, score)

    # Choose a random color from palette (deterministic by model name for stability)
    rng = _np.random.default_rng(abs(hash(model_name)) % (2**32))
    _color = rng.choice(ROC_COLOR_PALETTE)

    # CI
    if ci_tuple is None or any([c != c for c in ci_tuple]):
        auc_val, lo, hi = _bootstrap_auc_ci(y_true, score, n_boot=1000, alpha=0.05, random_state=42)
    else:
        lo, hi = ci_tuple[0], ci_tuple[1]

    # find optimal point (Youden J)
    j = tpr - fpr
    k = int(_np.nanargmax(j))
    fpr_star, tpr_star = fpr[k], tpr[k]

    _plt.figure(figsize=(4, 4), dpi=180)
    _plt.plot(fpr, tpr, linewidth=2, color=_color)
    _plt.fill_between(fpr, _np.maximum.accumulate(tpr*0), tpr, alpha=0.12, color=_color)
    _plt.plot([0, 1], [0, 1], '--', linewidth=1)
    _plt.scatter([fpr_star], [tpr_star], s=40, color='red', zorder=3)

    _plt.title(model_name)
    _plt.xlabel("False Positive Rate")
    _plt.ylabel("True Positive Rate")

    # AUC/CI text (no legend box)
    legend_text = f"AUC = {auc_val:.3f}\n95% CI: [{lo:.3f}, {hi:.3f}]"
    _ax = _plt.gca()
    _ax.text(0.98, 0.08, legend_text, fontsize=8, ha='right', va='bottom', transform=_ax.transAxes)

    # Optimal point text below the AUC box
    _ax.text(0.98, 0.02, f"Optimal point (FPR={fpr_star:.3f}, TPR={tpr_star:.3f})",
             fontsize=8, ha='right', va='bottom', transform=_ax.transAxes)

    _plt.tight_layout()
    outpath = os.path.join(outdir, f"{model_name.replace(' ', '_')}_roc.png")
    _plt.savefig(outpath)
    _plt.clf()
    _plt.close()


def get_best_model(model, X, y, param_grid, threads: int) -> Any:
    jobs = min(threads, 8)
    if _uses_gpu(model):
        jobs = 1
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc_ovr"
    }
    # === Added: richer scoring for imbalance ===
    scoring.update({
        "average_precision": "average_precision",
        "f1": "f1",
    })
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="roc_auc",  # default; may override below
        cv=cv,
        verbose=1,
        n_jobs=jobs,
        pre_dispatch="1*n_jobs",
        return_train_score=True
    )
    # === Added: set refit to PR AUC when SMOTETomek is enabled ===
    try:
        if GLOBAL_USE_SMOTETOMEK:
            grid_search.set_params(scoring=scoring, refit="average_precision")
            logger.info("Refit metric set to average_precision (PR AUC) due to SMOTETomek.")
        else:
            grid_search.set_params(scoring=scoring)  # keep default refit in constructor
    except Exception as _e:
        logger.warning("Could not adjust refit to average_precision; keeping default. %s", _e)

    with limit_threads(threads):
        try:
            grid_search.fit(X, y)
        except Exception as e:
            logger.warning("Parallel GridSearch failed (%s). Retrying with n_jobs=1 + threading backend...", type(e).__name__)
            try:
                grid_search.set_params(n_jobs=1)
            except Exception:
                pass
            try:
                from joblib import parallel_backend
                with parallel_backend("threading", n_jobs=1):
                    grid_search.fit(X, y)
            except Exception as e2:
                logger.error("GridSearch retry also failed: %s", e2)
                raise

    logger.info("Best params: %s", json.dumps(grid_search.best_params_, ensure_ascii=False))
    logger.info("Best CV AUROC: %.4f", grid_search.best_score_)
    results = grid_search.cv_results_
    mean_acc = np.max(results["mean_test_accuracy"])
    mean_auc = np.max(results["mean_test_roc_auc"])
    logger.info("CV Accuracy max: %.4f | AUROC max: %.4f", mean_acc, mean_auc)
    return grid_search.best_estimator_

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Any],
    grids: Dict[str, Dict[str, List[Any]]],
    threads: int,
    run_models: List[str],
    outdir: str
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    metrics_rows = []  # collect per-model metrics
    metrics_dir = os.path.join(outdir, "metrics")
    shap_base_dir = os.path.join(outdir, "shap")
    logs_dir = os.path.join(outdir, "logs")
    models_dir = os.path.join(outdir, "models")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(shap_base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "training.log")
    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as _f:
            _f.write(f"[{ts}] {msg}\n")

    _log("Starting training and evaluation run")

    for name in run_models:
        if name not in models:
            logger.warning("Model %s not found, skip.", name)
            continue

        logger.info("====== Training: %s ======", name)
        model = models[name]

        best_model = model
        if name in grids:
            best_model = get_best_model(model, X_train, y_train, grids[name], threads=threads)
        else:
            logger.info("No param grid for %s, skip tuning.", name)

        with limit_threads(threads):
            best_model.fit(X_train, y_train)
            # probabilities/scores
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)
                y_score = y_proba[:,1] if y_proba.ndim == 2 and y_proba.shape[1] >= 2 else y_proba.ravel()
            elif hasattr(best_model, "decision_function"):
                y_score = best_model.decision_function(X_test)
            else:
                y_score = None
            y_pred = best_model.predict(X_test)

        # classification report logging
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

        # metrics (binary preferred; for multi-class, we compute macro AUC and macro F1/accuracy only)
        row = {"model": name}
        try:
            if len(np.unique(y_test)) == 2:
                acc, f1, ppv, npv, sens, spec = _binary_metrics(y_test, y_pred)
                row.update({
                    "Accuracy": acc,
                    "F1": f1,
                    "PPV": ppv,
                    "NPV": npv,
                    "Sensitivity": sens,
                    "Specificity": spec
                })
                if y_score is not None:
                    auc, lo, hi = _bootstrap_auc_ci(y_test, y_score, n_boot=1000, alpha=0.05, random_state=42)
                    row.update({"AUC": auc, "AUC_CI95_L": lo, "AUC_CI95_U": hi})
                else:
                    row.update({"AUC": np.nan, "AUC_CI95_L": np.nan, "AUC_CI95_U": np.nan})
            else:
                # multi-class fallback
                acc = accuracy_score(y_test, y_pred)
                f1m = f1_score(y_test, y_pred, average="macro")
                row.update({"Accuracy": acc, "F1": f1m})
                if y_score is not None:
                    # if decision_function/proba gives 1D, cannot compute multi-class AUC
                    y_score_arr = y_score
                    if y_score_arr.ndim == 1:
                        row.update({"AUC": np.nan, "AUC_CI95_L": np.nan, "AUC_CI95_U": np.nan})
                    else:
                        auc, lo, hi = _bootstrap_auc_ci(y_test, y_score_arr, n_boot=1000, alpha=0.05, random_state=42)
                        row.update({"AUC": auc, "AUC_CI95_L": lo, "AUC_CI95_U": hi})
                # PPV/NPV/Sens/Spec are not uniquely defined for multi-class; omit
        except Exception as e:
            logger.warning("Metric computation failed for %s: %s", name, e)

        metrics_rows.append(row)

        # ROC plot per model (binary only)
        try:
            if len(np.unique(y_test)) == 2 and y_score is not None:
                roc_dir = os.path.join(outdir, "roc")
                ci = None
                if "AUC_CI95_L" in row and "AUC_CI95_U" in row and row.get("AUC_CI95_L")==row.get("AUC_CI95_L"):
                    ci = (row.get("AUC_CI95_L"), row.get("AUC_CI95_U"))
                plot_roc_with_ci(y_test, y_score, name, roc_dir, ci_tuple=ci)
        except Exception as e:
            logger.warning("ROC plot failed for %s: %s", name, e)

        # SHAP explain
        try:
            feat_names = list(X.columns)
            shap_dir = os.path.join(shap_base_dir, name.replace(" ", "_"))
            shap_explain_and_save(best_model, X_train, X_test, feat_names, shap_dir, name)
            # Save best params
            safe_name = name.replace(" ", "_")
            params_out = os.path.join(models_dir, f"{safe_name}_best_params.json")
            try:
                if hasattr(best_model, "best_params_"):
                    best_params = best_model.best_params_
                elif hasattr(best_model, "get_params"):
                    best_params = best_model.get_params()
                else:
                    best_params = {}
            except Exception as _e:
                best_params = {"_warning": f"could not extract params: {type(_e).__name__}: {_e}"}
            try:
                with open(params_out, "w", encoding="utf-8") as _pf:
                    json.dump(best_params, _pf, ensure_ascii=False, indent=2)
                _log(f"Saved best params for {name} -> {params_out}")
            except Exception as _e:
                _log(f"Failed to save best params for {name}: {type(_e).__name__}: {_e}")
        except Exception as e:
            logger.warning("SHAP failed for %s: %s", name, e)

    # Save metrics summary
    dfm = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(metrics_dir, "metrics_summary.csv")
    dfm.to_csv(csv_path, index=False)
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dfm.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics summary to: %s", csv_path)

# =========================
# Data loading
# =========================
def load_data(csv_path: str = "", target_col: str = "") -> Tuple[pd.DataFrame, np.ndarray]:
    """Load CSV, preserve original column names exactly; if target_col not given, use last column."""
    df = pd.read_csv(csv_path)
    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]
    y = df[target_col].values
    x = df.drop(columns=[target_col])
    # Explicitly preserve names
    x.columns = list(x.columns)
    return x, y

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Batch training with strict CPU/GPU limits + SHAP + metrics.")
    p.add_argument("--csv", type=str, default="", help="CSV path (optional; synthetic data if omitted)")
    p.add_argument("--target", type=str, default="", help="Target column name (for CSV mode)")
    p.add_argument("--models", type=str, default="Logistic Regression,Random Forest,XGBoost,LightGBM,CatBoost,MLP,GaussianNB,AdaBoost,Stacking",
    help="Models to run, comma-separated. Default runs all.")
    p.add_argument("--cpu-limit", type=int, default=CPU_LIMIT, help=f"CPU threads cap (default {CPU_LIMIT})")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU-only (ignore GPU)")
    p.add_argument("--outdir", type=str, default="outputs", help="Base directory for all outputs (metrics, SHAP, etc.)")

    p.add_argument("--use-smotetomek", action="store_true",
    help="Enable SMOTETomek resampling inside CV folds (and refit on PR AUC/F1)")

    return p.parse_args()

def main():
    args = parse_args()
    threads = _coerce_cpu_limit(str(args.cpu_limit))

    gpu_ok = (not args.cpu_only) and have_torch_cuda()
    if gpu_ok:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        logger.info("GPU available, using device 0.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU mode (or forced CPU).")

    x , y = load_data(args.csv, args.target)
    models, grids = model_param_init(threads=threads, use_gpu=gpu_ok)

    # === Added: expose flag to module scope ===
    global GLOBAL_USE_SMOTETOMEK
    GLOBAL_USE_SMOTETOMEK = args.use_smotetomek

    # === Optional: wrap estimators with SMOTETomek inside CV folds ===
    if args.use_smotetomek:
        smt = SMOTETomek(
            smote=SMOTE(random_state=42, k_neighbors=5),
            tomek=TomekLinks(n_jobs=1)
        )
        new_models = {}
        new_grids = {}
        for _name, _est in models.items():
            # Wrap: resample -> est
            new_models[_name] = ImbPipeline([("resample", smt), ("est", _est)])
        for _name, _grid in grids.items():
            _ng = {}
            for _k, _v in _grid.items():
                _ng["est__" + _k] = _v
            # Tune SMOTE neighbors and sampling ratio (milder than 1:1)
            _ng["resample__smote__k_neighbors"] = [3, 5, 7]
            _ng["resample__sampling_strategy"] = [0.5, 0.7, "auto"]
            new_grids[_name] = _ng
            logger.info("SMOTETomek enabled: models wrapped for in-fold resampling; PR-oriented refit will be used.")
            models, grids = new_models, new_grids

    run_models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_models = [m for m in run_models if m in models]

    logger.info("Models to train: %s", ", ".join(run_models))
    logger.info("CPU threads cap: %d", threads)

    train_and_evaluate(x, y, models, grids, threads=threads, run_models=run_models, outdir=args.outdir)

if __name__ == "__main__":
    main()
