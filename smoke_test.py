import torch
import torch.nn as nn
import sys
sys.path.append('.')
from source.models.HSGLM.HSGLM import HSGLM
from pathlib import Path
from omegaconf import OmegaConf
PROJECT_ROOT = Path(__file__).resolve().parent

def load_config_for_smoke_test():
    """从 YAML 加载配置，缩小规模以加速测试"""
    # 1) 加载真实配置
    model_cfg = OmegaConf.load(PROJECT_ROOT / "source/conf/model/HSGLM.yaml")
    dataset_cfg = OmegaConf.load(PROJECT_ROOT / "source/conf/dataset/ABIDE2.yaml")

    # 2) 缩小规模（不改变结构相关维度）
    dataset_cfg.timeseries_sz = 30
    dataset_cfg.windows_sz = 5
    dataset_cfg.non_imaging_feature_sz = 3
    dataset_cfg.node_sz = [48, 90, 200]

    model_cfg.hidden_dim = 32
    model_cfg.kernel_size = 3
    model_cfg.topk = 10
    model_cfg.num_heads = 1
    model_cfg.mamba_layers = 1

    # 3) 合并
    config = OmegaConf.create({
        "model": model_cfg,
        "dataset": dataset_cfg,
    })
    return config


def main():
    print("=" * 60)
    print("HSGLM Smoke Test")
    print("=" * 60)

    config = load_config_for_smoke_test()
    print("\n[Config] model keys:", list(config.model.keys()))
    print("[Config] dataset keys:", list(config.dataset.keys()))

    # 构造小 batch
    B, W, C = 2, 5, 30
    N1, N2, N3 = config.dataset.node_sz
    BOLD1 = torch.randn(B, W, N1, C)
    BOLD2 = torch.randn(B, W, N2, C)
    BOLD3 = torch.randn(B, W, N3, C)
    phd   = torch.randn(B, config.dataset.non_imaging_feature_sz)
    labels = torch.tensor([0, 1], dtype=torch.long)

    # 实例化
    print("\n[1/4] 实例化模型...")
    model = HSGLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"     ✓ 参数量: {n:,}")

    # 前向
    print("\n[2/4] 前向传播...")
    model.eval()
    with torch.no_grad():
        logits, fusion,attention = model(BOLD1, BOLD2, BOLD3, phd)
    assert logits.shape == (B, 2)
    assert not torch.isnan(logits).any()
    print(f"     ✓ logits shape: {logits.shape}")

    # 反向
    print("\n[3/4] 反向传播...")
    model.train()
    logits, _,_ = model(BOLD1, BOLD2, BOLD3, phd)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print("\n[DEBUG] 检查各层参数的梯度情况:")
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        print(f"层: {name} | 参数量: {param.numel()} | 有梯度: {has_grad}")

    print("\n" + "=" * 60)
    print("✓ 冒烟测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()