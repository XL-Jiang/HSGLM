import os
import pickle
from sklearn.decomposition import PCA
from typing import Union, List, Dict, Optional, Tuple, Literal
import torch
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
def plot_tsne_scatter(fusions: np.ndarray, labels: np.ndarray,
                      save_dir: Path, seed: int,
                      save_npy: bool = True,
                      random_state: int = 42):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    if save_npy:
        np.save(save_dir / f"seed_{seed}_all_folds_fusions.npy", fusions)
        np.save(save_dir / f"seed_{seed}_all_folds_labels.npy", labels)

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, init='pca', learning_rate='auto')
    fusions_2d = tsne.fit_transform(fusions)

    # 3. 绘制散点图
    plt.figure(figsize=(8, 6), dpi=300)

    classes = np.unique(labels)
    # colors = ['#1f77b4', '#ff7f0e']  #蓝橙
    class_names = {0: 'HC', 1: 'ASD'}  #

    colors = ['#FB8072', '#80B1D3']  # 红蓝 （0 1）
    # colors = ['#800080', '#FFDD00']  # 黄紫 （0 1）
    for cls in classes:
        cls_int = int(cls)
        mask = (labels == cls)
        plt.scatter(
            fusions_2d[mask, 0],
            fusions_2d[mask, 1],
            c=colors[cls_int % len(colors)],
            label=class_names.get(cls_int, f'Class {cls_int}'),
            alpha=1,  # 透明度
            edgecolors='none',
            s=100
        )

    # plt.title(f'Integrated 5-Fold t-SNE Feature Map (Seed {seed})', fontsize=13, fontweight='bold')
    # plt.xlabel('t-SNE Dimension 1', fontsize=11)
    # plt.ylabel('t-SNE Dimension 2', fontsize=11)

    plt.xticks([])
    plt.yticks([])
    # plt.legend(frameon=True, loc='best', fontsize=10) #图例
    # plt.grid(True, linestyle='--', alpha=0.3) #网格

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    fig_path = save_dir / f"tsne_scatter_seed_{seed}.tiff"
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"--> [Seed {seed}] 5折整合 t-SNE 散点图已保存至: {fig_path}")