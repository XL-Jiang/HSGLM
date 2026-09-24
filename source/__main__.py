import copy
import itertools
import json
import hydra
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig, open_dict, OmegaConf
from dataset import dataset_factory
from models import model_factory
from components import optimizers_factory, logger_factory
from source.utils.feature_visualizer import plot_tsne_scatter
from sklearn.metrics import roc_auc_score
from training import training_factory
from datetime import datetime
import os
import time
import numpy as np
import random
import torch

def _aggregate_site_records(site_records):
    from collections import defaultdict

    by_site = defaultdict(list)
    for rec in site_records:
        site = rec.get('site')
        if site is None:
            continue
        by_site[site].append(rec)

    site_summary = []
    for site, recs in sorted(by_site.items()):
        metrics = [
            {k: v for k, v in r.items() if k not in ('seed', 'site')}
            for r in recs
        ]
        summary = _aggregate_metrics(metrics, skip_keys=(), ddof=1)

        item = {'site': site, 'n': len(recs)}
        for k, v in summary.items():
            item[f'mean_{k}'] = v['mean']
            item[f'std_{k}'] = v['std']
        site_summary.append(item)

    return site_summary

def _default_prefix(cfg):
    return (
        cfg.get('evaluation', {}).get('unique_id_prefix', None)
        or cfg.get('dataset', {}).get('name', 'ABIDE2_AAL_90_HO_96_CC200')
    )


def _get_cfg_list(cfg, *keys):
    cur = cfg
    for k in keys:
        if cur is None:
            return None
        if hasattr(cur, 'get'):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    if cur is None:
        return None
    return list(cur)


def _collect_fold_ids_for_seed(cfg, SEED, prefix=None):
    if prefix is None:
        prefix = _default_prefix(cfg)

    kfold_enable = cfg.get('kfold', {}).get('enable', False)
    cv_mode = cfg.get('cv_mode', 'kfold')

    if not kfold_enable:
        id_list = cfg.get('evaluation', {}).get('unique_id_list', None)
        if not id_list:
            return []
        seeds = list(cfg.seeds)
        try:
            idx = seeds.index(SEED)
        except ValueError:
            return []
        if idx >= len(id_list):
            return []
        return [id_list[idx]]

    if cv_mode == 'loso':
        sites = _get_cfg_list(cfg, 'loso', 'sites')
        if sites is None:
            sites = _get_cfg_list(cfg, 'kfold', 'sites')
        if sites is None:
            sites = _get_cfg_list(cfg, 'dataset', 'sites')

        if sites:
            return [f"{prefix}_seed{SEED}_{site}" for site in sites]

        log_path = Path(cfg.log_path)
        pattern = f"{prefix}_seed{SEED}_*"
        dirs = sorted([
            p.name for p in log_path.glob(pattern)
            if p.is_dir() and not p.name.startswith('summary')
        ])
        return dirs

    n_folds = cfg.get('kfold', {}).get('n_splits', 5)
    return [f"{prefix}_seed{SEED}_fold{fold_idx}" for fold_idx in range(n_folds)]




def evaluate_efficiency_from_saved(save_dir):

    save_dir = Path(save_dir)
    csv_file = save_dir / "efficiency_per_epoch.csv"
    if not csv_file.exists():
        print(f"[Warn] 未找到 efficiency_per_epoch.csv：{csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[Warn] 读取 {csv_file} 失败：{e}")
        return None

    if len(df) == 0:
        return None

    metric_cols = [
        "Train Time (s/epoch)",
        "GPU Peak Allocated (MB)",
        "GPU Peak Reserved (MB)",
        "GPU Global Used (MB)",
        "GPU Global Used (%)",
    ]
    const_cols = ["Train Samples (N)", "Batch Size", "Total Params (N)", "Trainable Params (N)"]

    result = {}
    for c in metric_cols:
        if c in df.columns:
            result[c] = float(df[c].mean())
    for c in const_cols:
        if c in df.columns:
            result[c] = float(df[c].iloc[0])
    result["n_measure_epochs"] = int(len(df))
    return result
def _is_nan(x):
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def compute_metrics_from_cm(cm):
    """
    从混淆矩阵计算全部指标。
    cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    """
    TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = TP + TN + FP + FN

    ACC       = (TP + TN) / total if total > 0 else 0.0
    SEN       = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    SPE       = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    NPV       = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    F1        = (2 * Precision * SEN / (Precision + SEN)) if (Precision + SEN) > 0 else 0.0
    BA        = (SEN + SPE) / 2.0

    denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    MCC = ((TP * TN - FP * FN) / denom) if denom > 0 else 0.0

    return {
        'ACC': ACC, 'SEN': SEN, 'SPE': SPE, 'Precision': Precision,
        'NPV': NPV, 'F1': F1, 'BA': BA, 'MCC': MCC
    }

def plot_tsne_from_existing_folds(
    cfg,
    seeds=None,
    zscore_per_fold=True,
    random_state=42,
):
    if seeds is None:
        seeds = list(cfg.seeds)
    prefix = _default_prefix(cfg)

    for SEED in seeds:
        fold_ids = _collect_fold_ids_for_seed(cfg, SEED, prefix)

        fusions_list, labels_list = [], []
        for fold_id in fold_ids:
            save_dir = Path(cfg.log_path) / fold_id
            f_file = save_dir / "best_test_fusions.npy"
            l_file = save_dir / "best_test_labels.npy"
            if not (f_file.exists() and l_file.exists()):
                print(f"[tsne] 跳过（缺文件）: {save_dir}")
                continue

            f = np.load(f_file).astype(np.float64)
            l = np.load(l_file).astype(int).ravel()

            if zscore_per_fold:
                f = (f - f.mean(axis=0, keepdims=True)) \
                    / (f.std(axis=0, keepdims=True) + 1e-8)

            fusions_list.append(f)
            labels_list.append(l)

        if not fusions_list:
            print(f"[tsne] seed {SEED} 无可用折")
            continue

        F = np.vstack(fusions_list)
        Y = np.concatenate(labels_list).ravel()

        summary_dir = Path(cfg.log_path) / f"summary_seed_{SEED}"
        summary_dir.mkdir(parents=True, exist_ok=True)

        f_path = summary_dir / f"seed_{SEED}_all_folds_fusions.npy"
        l_path = summary_dir / f"seed_{SEED}_all_folds_labels.npy"

        backup = {
            'f': np.load(f_path) if f_path.exists() else None,
            'l': np.load(l_path) if l_path.exists() else None,
        }
        plot_tsne_scatter(F, Y, save_dir=summary_dir, seed=SEED,
                          save_npy=True, random_state=random_state)
        if backup['f'] is not None:
            np.save(f_path, backup['f'])
        elif f_path.exists():
            f_path.unlink()
        if backup['l'] is not None:
            np.save(l_path, backup['l'])
        elif l_path.exists():
            l_path.unlink()

        print(f"[tsne] seed {SEED} 绘图 → {summary_dir}")

def evaluate_fold_from_saved(save_dir):
    save_dir = Path(save_dir)
    cm_file = save_dir / "confusion_matrix.npy"
    if not cm_file.exists():
        print(f"[Warn] 未找到混淆矩阵：{cm_file}")
        return None

    cm = np.load(cm_file)
    metrics = compute_metrics_from_cm(cm)

    probs_file  = save_dir / "test_probs.npy"
    labels_file = save_dir / "test_labels_for_auc.npy"
    if probs_file.exists() and labels_file.exists():
        probs  = np.load(probs_file)
        labels = np.load(labels_file)
        if probs.ndim == 2:
            probs = probs[:, 1]
        try:
            metrics['AUC'] = float(roc_auc_score(labels, probs))
        except Exception as e:
            print(f"[Warn] 计算 AUC 失败 {save_dir}: {e}")
            metrics['AUC'] = float('nan')
    else:
        print(f"[Warn] 未找到 probs/labels，AUC 置为 nan：{save_dir}")
        metrics['AUC'] = float('nan')

    return metrics


def _aggregate_metrics(metrics_list, skip_keys=('fold',), ddof=1):
    if not metrics_list:
        return {}


    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    for k in skip_keys:
        all_keys.discard(k)

    summary = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in metrics_list
                if key in m and not _is_nan(m[key])]
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=ddof))
        elif len(vals) == 1:
            mean, std = float(vals[0]), float('nan')
        else:
            mean, std = float('nan'), float('nan')
        summary[key] = {
            'mean': mean,
            'std':  std,
            'per_seed': [float(v) for v in vals]
        }
    return summary


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def model_training(cfg: DictConfig, SEED, hyperparams=None):
    with open_dict(cfg):
        cfg.seed = SEED
        param_suffix = ""
        if hyperparams:
            for key, value in hyperparams.items():
                if '.' in key:
                    parts = key.split('.')
                    current = cfg
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], value)
                else:
                    setattr(cfg, key, value)
            param_suffix = "_" + "_".join([f"{k.split('.')[-1]}={v}" for k, v in hyperparams.items()])
    dataloaders = dataset_factory(cfg, SEED)
    raw_base_id = (
            cfg.get('evaluation', {}).get('unique_id_prefix', None)
            or cfg.get('dataset', {}).get('name', 'ABIDE2_AAL_90_HO_96_CC200')
    )
    base_id = raw_base_id + param_suffix

    if isinstance(dataloaders, list) and len(dataloaders) > 0 \
            and isinstance(dataloaders[0], tuple):

        fold_metrics_list = []
        fold_fusions_list = []
        fold_labels_list = []

        for fold_idx, item in enumerate(dataloaders):

            if len(item) == 3:
                train_loader, val_loader, test_loader = item
                site_name = None
            else:
                train_loader, val_loader, test_loader, site_name = item

            with open_dict(cfg):
                if cfg.get('cv_mode', 'kfold') == 'loso' and site_name is not None:
                    cfg.unique_id = f"{base_id}_seed{SEED}_{site_name}"
                    cfg.confusion_matrix_path = cfg.unique_id
                else:
                    cfg.unique_id = f"{base_id}_seed{SEED}_fold{fold_idx}"
                    cfg.confusion_matrix_path = cfg.unique_id

                cfg.unique_id1 = base_id

            print(f"==================== Seed {SEED} | Fold/Site {fold_idx} ====================")
            if site_name is not None:
                print(f"Site: {site_name}")

            logger = logger_factory(cfg)
            model = model_factory(cfg)

            if cfg.state == 'testing':
                print("############testing##############")
                cfg.training.epochs = 1
                save_dir = Path(cfg.log_path) / cfg.unique_id
                model_weight_path = save_dir / "model.pt"
                if not os.path.exists(model_weight_path):
                    raise FileNotFoundError(f"模型权重文件不存在：{model_weight_path}")
                state_dict = torch.load(model_weight_path)
                model.load_state_dict(state_dict)
                logger.info(f"成功加载模型权重：{model_weight_path}")

            optimizers = optimizers_factory(model, cfg.optimizer)
            training = training_factory(
                cfg, model, optimizers,
                [train_loader, val_loader, test_loader],
                logger
            )

            training.train()

            fold_save_dir = Path(cfg.log_path) / cfg.unique_id
            metrics = evaluate_fold_from_saved(fold_save_dir)

            if metrics is not None:
                metrics['fold'] = fold_idx
                if site_name is not None:
                    metrics['site'] = site_name
                fold_metrics_list.append(metrics)

            fusion_file = fold_save_dir / "best_test_fusions.npy"
            label_file = fold_save_dir / "best_test_labels.npy"
            if fusion_file.exists() and label_file.exists():
                fold_fusions_list.append(np.load(fusion_file))
                fold_labels_list.append(np.load(label_file))
            else:
                print(f"Warning: 折/站点 {fold_idx} 的特征文件不存在，跳过该折。")

        if len(fold_fusions_list) == len(dataloaders):
            combined_fusions = np.vstack(fold_fusions_list)
            combined_labels = np.concatenate(fold_labels_list)
            summary_dir = Path(cfg.log_path) / f"summary_seed_{SEED}"
            plot_tsne_scatter(
                combined_fusions, combined_labels,
                save_dir=summary_dir, seed=SEED
            )

        return fold_metrics_list

    else:
        with open_dict(cfg):
            cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
            if not hasattr(cfg, 'unique_id1'):
                cfg.unique_id1 = f"experiment_seed{SEED}"

        logger = logger_factory(cfg)
        model = model_factory(cfg)

        if cfg.state == 'testing':
            print("############testing##############")
            cfg.training.epochs = 1
            save_dir = Path(cfg.log_path) / cfg.unique_id
            model_weight_path = save_dir / "model.pt"
            if not os.path.exists(model_weight_path):
                raise FileNotFoundError(f"模型权重文件不存在：{model_weight_path}")
            state_dict = torch.load(model_weight_path)
            model.load_state_dict(state_dict)
            logger.info(f"成功加载模型权重：{model_weight_path}")

        optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
        training = training_factory(cfg, model, optimizers, dataloaders, logger)

        training.train()

        save_dir = Path(cfg.log_path) / cfg.unique_id
        metrics = evaluate_fold_from_saved(save_dir)
        return metrics


def run_single_experiment(cfg: DictConfig, params=None):
    if params:
        with open_dict(cfg):
            for key, value in params.items():
                if '.' in key:
                    parts = key.split('.')
                    current = cfg
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], value)
                else:
                    setattr(cfg, key, value)

    per_seed_metrics = []
    all_site_records = []

    seeds = cfg.seeds
    kfold_enable = cfg.get('kfold', {}).get('enable', False)
    cv_mode = cfg.get('cv_mode', 'kfold')

    for SEED in seeds:
        print(f"\n########## SEED = {SEED} ##########")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if kfold_enable:
            fold_metrics = model_training(cfg, SEED, params)

            if fold_metrics:
                if cv_mode == 'loso':
                    for m in fold_metrics:
                        if 'site' in m:
                            rec = {'seed': int(SEED), 'site': m['site']}
                            for k, v in m.items():
                                if k not in ('fold', 'site'):
                                    rec[k] = v
                            all_site_records.append(rec)

                fold_summary = _aggregate_metrics(
                    fold_metrics,
                    skip_keys=('fold', 'site'),
                    ddof=0
                )
                seed_mean = {k: v['mean'] for k, v in fold_summary.items()}
                per_seed_metrics.append(seed_mean)

                print(f"[SEED={SEED}] 折/站点内平均: " + " | ".join(
                    f"{k}={v:.4f}" for k, v in seed_mean.items()
                ))
        else:
            metrics = model_training(cfg, SEED, params)
            if metrics is not None:
                per_seed_metrics.append(metrics)

    metric_summary = _aggregate_metrics(
        per_seed_metrics,
        skip_keys=(),
        ddof=1
    )

    result = {
        "params": params if params else "default",
        "seeds": list(seeds),
        "per_seed_metrics": per_seed_metrics,
        "metric_summary": metric_summary,
    }

    if cv_mode == 'loso' and all_site_records:
        result['site_summary'] = _aggregate_site_records(all_site_records)

    return result

def evaluate_from_results(cfg: DictConfig, seeds=None):
    if seeds is None:
        seeds = list(cfg.seeds)

    kfold_enable = cfg.get('kfold', {}).get('enable', False)
    cv_mode = cfg.get('cv_mode', 'kfold')
    prefix = _default_prefix(cfg)

    per_seed_metrics = []
    all_site_records = []

    per_fold_efficiency_records = []
    per_seed_efficiency = []

    for SEED in seeds:
        fold_metrics = []
        fold_efficiency = []
        fold_ids = _collect_fold_ids_for_seed(cfg, SEED, prefix)

        if not fold_ids:
            print(f"[Warn] seed {SEED} 未找到任何保存目录，跳过")
            continue

        for fold_idx, unique_id in enumerate(fold_ids):
            save_dir = Path(cfg.log_path) / unique_id
            metrics = evaluate_fold_from_saved(save_dir)

            if metrics is None:
                print(f"[Warn] 无法读取 {save_dir}，跳过")

                eff = evaluate_efficiency_from_saved(save_dir)
                if eff is not None:
                    eff_rec = dict(eff)
                    eff_rec['seed'] = int(SEED)
                    eff_rec['fold'] = fold_idx
                    fold_efficiency.append(eff)
                    per_fold_efficiency_records.append(eff_rec)
                continue

            if kfold_enable:
                if cv_mode == 'loso':
                    site = unique_id.replace(f"{prefix}_seed{SEED}_", "")
                    metrics['site'] = site
                    metrics['fold'] = fold_idx
                else:
                    try:
                        metrics['fold'] = int(unique_id.split('_fold')[-1])
                    except Exception:
                        metrics['fold'] = fold_idx
            else:
                metrics['fold'] = 0

            fold_metrics.append(metrics)

            if cv_mode == 'loso' and 'site' in metrics:
                rec = {'seed': int(SEED), 'site': metrics['site']}
                for k, v in metrics.items():
                    if k not in ('fold', 'site'):
                        rec[k] = v
                all_site_records.append(rec)

            eff = evaluate_efficiency_from_saved(save_dir)
            if eff is not None:
                eff_rec = dict(eff)
                eff_rec['seed'] = int(SEED)
                eff_rec['fold'] = fold_idx
                if 'site' in metrics:
                    eff_rec['site'] = metrics['site']
                fold_efficiency.append(eff)
                per_fold_efficiency_records.append(eff_rec)

            tag = metrics.get('site', f"fold{metrics.get('fold')}")
            msg = " | ".join(
                f"{k}={v:.4f}"
                for k, v in metrics.items()
                if k not in ('fold', 'site')
            )
            print(f"[SEED={SEED}][{tag}] {msg}")

        if fold_metrics:
            fold_summary = _aggregate_metrics(
                fold_metrics,
                skip_keys=('fold', 'site'),
                ddof=0
            )
            seed_mean = {k: v['mean'] for k, v in fold_summary.items()}
            per_seed_metrics.append(seed_mean)

            print(f"[SEED={SEED}] 折/站点内平均: " + " | ".join(
                f"{k}={v:.4f}" for k, v in seed_mean.items()
            ))
        if fold_efficiency:
            eff_sum = _aggregate_metrics(fold_efficiency, skip_keys=(), ddof=0)
            seed_eff_mean = {k: v['mean'] for k, v in eff_sum.items()}
            per_seed_efficiency.append(seed_eff_mean)

            print(f"[SEED={SEED}] 折/站点内效率平均: " + " | ".join(
                f"{k}={v:.4f}" for k, v in seed_eff_mean.items()
            ))

    if not per_seed_metrics:
        print("[Error] 没有收集到任何折/站点的指标，请检查保存路径是否正确")
        return None

    metric_summary = _aggregate_metrics(
        per_seed_metrics,
        skip_keys=(),
        ddof=1
    )

    efficiency_summary = (
        _aggregate_metrics(per_seed_efficiency, skip_keys=(), ddof=1)
        if per_seed_efficiency else {}
    )

    efficiency_all_folds_summary = (
        _aggregate_metrics(per_fold_efficiency_records,
                           skip_keys=('seed', 'fold', 'site'),
                           ddof=1)
        if per_fold_efficiency_records else {}
    )

    result = {
        "mode": "evaluation",
        "seeds": list(seeds),
        "n_folds": (
            cfg.get('kfold', {}).get('n_splits', 5)
            if kfold_enable and cv_mode != 'loso' else None
        ),
        "per_seed_metrics": per_seed_metrics,
        "metric_summary": metric_summary,

        "per_fold_efficiency": per_fold_efficiency_records,
        "per_seed_efficiency": per_seed_efficiency,
        "efficiency_summary": efficiency_summary,
        "efficiency_all_folds_summary": efficiency_all_folds_summary
    }

    if cv_mode == 'loso' and all_site_records:
        result['site_summary'] = _aggregate_site_records(all_site_records)

    return result

def save_evaluation_result(cfg: DictConfig, result):

    eval_cfg = cfg.get('evaluation', {})
    default_name = f"{getattr(cfg, 'unique_id1', 'evaluation_result')}_evaluation.json"
    save_name = eval_cfg.get('result_file', default_name)

    out_dir = eval_cfg.get('output_dir', None)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_name = str(Path(out_dir) / save_name)

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump([result], f, indent=4, ensure_ascii=False)

    print(f"\n评估结果已保存至: {save_name}")
    return save_name


def generate_hyperparameter_grid(param_grid):
    param_grid_dict = OmegaConf.to_container(param_grid, resolve=True)
    keys = param_grid_dict.keys()
    values = param_grid_dict.values()
    param_combinations = [dict(zip(keys, combination))
                          for combination in itertools.product(*values)]
    return param_combinations


def _print_metric_summary(result):
    ms = result.get('metric_summary', {})
    print("\n=== 指标汇总（折内平均 → 种子间 mean ± std）===")
    for k in ['ACC', 'SEN', 'SPE', 'Precision', 'NPV', 'F1', 'BA', 'MCC', 'AUC']:
        if k in ms:
            print(f"{k:10s}: {ms[k]['mean']:.4f} ± {ms[k]['std']:.4f}")

    eff_seed = result.get('efficiency_summary', {})
    eff_fold = result.get('efficiency_all_folds_summary', {})

    if eff_seed:
        print("\n=== 效率汇总（先折内平均 → 种子间 mean ± std）===")
        for k, v in eff_seed.items():
            print(f"{k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")

    if eff_fold:
        print("\n=== 效率汇总（所有种子所有折合并 mean ± std）===")
        for k, v in eff_fold.items():
            print(f"{k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")


def run_grid_search(cfg: DictConfig, param_grid_config):
    param_combinations = generate_hyperparameter_grid(param_grid_config)
    all_results = []

    base_prefix = _default_prefix(cfg)
    results_file = f"grid_search_results_{base_prefix}.json"
    print(f"当前网格搜索结果将实时写入: {results_file}")

    for i, params in enumerate(param_combinations):
        print(f"\n=== 超参数组合 {i + 1}/{len(param_combinations)} ===")
        print(f"参数: {params}")
        current_cfg = copy.deepcopy(cfg)

        result = run_single_experiment(current_cfg, params)
        all_results.append(result)

        results_file = "grid_search_results.json"
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        _print_metric_summary(result)

    if all_results:
        def _acc_of(r):
            return r.get('metric_summary', {}).get('ACC', {}).get('mean', -1.0)

        best_result = max(all_results, key=_acc_of)
        print("\n=== 最佳超参数组合 ===")
        print(f"参数: {best_result['params']}")
        _print_metric_summary(best_result)

    with open(results_file, 'w', encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    return all_results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    state = getattr(cfg, 'state', 'training')

    if state == 'evaluation':
        print("=" * 60)
        print("评估模式：仅从已保存的混淆矩阵/概率计算指标")
        print("=" * 60)


        plot_tsne_from_existing_folds(
            cfg,
            seeds=list(cfg.seeds),
            random_state=42,
        )

        result = evaluate_from_results(cfg)

        if result is None:
            print("[Error] 评估失败，未生成结果")
            return

        _print_metric_summary(result)

        site_summary = result.get('site_summary', [])
        if site_summary:
            print("\n=== LOSO 站点汇总（跨种子 mean ± std）===")
            for item in site_summary:
                print(
                    f"{item['site']}: "
                    f"ACC={item.get('mean_ACC', float('nan')):.4f}"
                    f"±{item.get('std_ACC', float('nan')):.4f}, "
                    f"AUC={item.get('mean_AUC', float('nan')):.4f}"
                    f"±{item.get('std_AUC', float('nan')):.4f}"
                )

        save_evaluation_result(cfg, result)
        return

    if hasattr(cfg, 'grid_search') and getattr(cfg.grid_search, 'enable', False):
        print("开始网格搜索...")

        if not hasattr(cfg.grid_search, 'config_file') or cfg.grid_search.config_file is None:
            print("错误：未指定网格搜索配置文件！")
            return

        grid_search_config_path = os.path.join("conf", "grid_search",
                                               cfg.grid_search.config_file)

        if not os.path.exists(grid_search_config_path):
            print(f"错误：网格搜索配置文件不存在：{grid_search_config_path}")
            return

        try:
            param_grid_config = OmegaConf.load(grid_search_config_path)
            print(f"已加载网格搜索配置：{cfg.grid_search.config_file}")
            run_grid_search(cfg, param_grid_config)
        except Exception as e:
            print(f"加载或运行网格搜索时出错：{e}")
            return

    else:
        print("运行单次实验（多种子 × k 折）...")
        result = run_single_experiment(cfg)

        _print_metric_summary(result)

        save_name = getattr(cfg, 'unique_id1', 'experiment_result')
        results_file = f"{save_name}.json"
        with open(results_file, 'w', encoding="utf-8") as f:
            json.dump([result], f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()