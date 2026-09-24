import pandas as pd
from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
import logging
import time
from sklearn.metrics import confusion_matrix

def get_gpu_mem_in_mb():
    """返回当前进程已占用的显存（MB）"""
    return torch.cuda.memory_allocated() / 1024**2

def get_gpu_mem_in_pct():
    """返回当前占用比例（0~1）"""
    free, total = torch.cuda.mem_get_info()
    return 1 - free / total


class training_monitoring:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.optimizers = optimizers
        self.min_epochs = cfg.training.min_epochs
        self.patience = cfg.training.patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_labels = self.train_dataloader.dataset.tensors[4]
        if train_labels.dim() > 1:
            train_labels = train_labels.argmax(dim=1)
        N = train_labels.shape[0]
        N0 = (train_labels == 0).sum().item()
        N1 = (train_labels == 1).sum().item()
        class_weights = torch.tensor(
            [N / (2.0 * N0), N / (2.0 * N1)],
            dtype=torch.float32
        ).to(self.device)
        self.logger.info(f"Class weights (N0={N0}, N1={N1}): {class_weights.tolist()}")

        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction='sum',
            weight=class_weights
        )

        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.confusion_matrix_path = Path(cfg.log_path) / cfg.confusion_matrix_path
        self.save_test_attn_weights = cfg.save_test_attn_weights

        # ===== 效率测量配置 =====
        self.warmup_epochs = getattr(cfg.training, "warmup_epochs", 50)
        self.measure_epochs = getattr(cfg.training, "measure_epochs", 10)
        self.precision = getattr(cfg.training, "precision", "float32")
        self.batch_size = getattr(cfg.dataset, "batch_size", None)

        self.perf_records = []
        self.init_meters()
        self.mseLoss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()
        print([name for name, _ in self.model.named_modules()])

    @staticmethod
    def _sync():
        """确保 CUDA 异步 kernel 执行完毕，计时准确"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _gpu_snapshot(self):
        """当前进程 / 全局 GPU 显存快照（MB、百分比）"""
        if not torch.cuda.is_available():
            return dict(
                proc_alloc_mb=0.0,
                proc_reserved_mb=0.0,
                peak_alloc_mb=0.0,
                peak_reserved_mb=0.0,
                global_used_mb=0.0,
                global_used_pct=0.0,
            )

        proc_alloc_mb = torch.cuda.memory_allocated() / 1024**2
        proc_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

        free, total = torch.cuda.mem_get_info()
        global_used_mb = (total - free) / 1024**2
        global_used_pct = (total - free) / total

        return dict(
            proc_alloc_mb=proc_alloc_mb,
            proc_reserved_mb=proc_reserved_mb,
            peak_alloc_mb=peak_alloc_mb,
            peak_reserved_mb=peak_reserved_mb,
            global_used_mb=global_used_mb,
            global_used_pct=global_used_pct,
        )

    def _save_perf_records(self):
        """把逐轮效率数据保存到 self.save_path（CSV）"""
        if not self.perf_records:
            return
        self.save_path.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(self.perf_records)

        # 只保留需要的列，并带明确单位
        df = df[[
            "train_time_sec",
            "num_train_samples",
            "batch_size",
            "gpu_peak_allocated_mb",
            "gpu_peak_reserved_mb",
            "gpu_global_used_mb",
            "gpu_global_used_pct",
        ]].rename(columns={
            "train_time_sec": "Train Time (s/epoch)",
            "num_train_samples": "Train Samples (N)",
            "batch_size": "Batch Size",
            "gpu_peak_allocated_mb": "GPU Peak Allocated (MB)",
            "gpu_peak_reserved_mb": "GPU Peak Reserved (MB)",
            "gpu_global_used_mb": "GPU Global Used (MB)",
            "gpu_global_used_pct": "GPU Global Used (%)",
        })

        df.to_csv(self.save_path / "efficiency_per_epoch.csv", index=False)
        self.logger.info(
            f"Per-epoch efficiency saved to {self.save_path / 'efficiency_per_epoch.csv'}"
        )

    # ================================================================

    def init_meters(self):
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def save_test_predictions(self, probs, labels):
        """保存测试集 softmax 概率与标签，供训练结束后独立计算 AUC"""
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "test_probs.npy", np.asarray(probs))
        np.save(self.save_path / "test_labels_for_auc.npy", np.asarray(labels))
        self.logger.info(f"Test probs & labels saved to {self.save_path}")

    def save_confusion_matrix(self, cm, epoch):
        """
        cm: np.ndarray, 形状 (2,2)，行=真实标签, 列=预测标签
        """
        self.save_path.mkdir(exist_ok=True, parents=True)


        np.save(self.save_path / "confusion_matrix.npy", cm)


        cm_df = pd.DataFrame(
            cm,
            index=["True_0", "True_1"],
            columns=["Pred_0", "Pred_1"]
        )
        cm_df.to_csv(self.save_path / "confusion_matrix.csv")
        self.logger.info(f"Confusion matrix saved to {self.save_path}")

    def train_per_epoch(self, optimizer):
        self.model.train()
        for features1, features2, features3, non_imaging, label in self.train_dataloader:
            label = label.float().cuda()

            predict, _, _ = self.model(features1.cuda(), features2.cuda(), features3.cuda(), non_imaging.cuda())
            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        fusion_features = []

        self.model.eval()
        with torch.no_grad():
            for features1, features2, features3, non_imaging, label in dataloader:
                predict, fusion, attention = self.model(features1.cuda(), features2.cuda(), features3.cuda(),
                                                        non_imaging.cuda())

                label = label.float().cuda()

                loss = self.loss_fn(predict, label)
                loss_meter.update_with_weight(loss.item(), label.shape[0])
                top1 = accuracy(predict, label[:, 1])[0]
                acc_meter.update_with_weight(top1, label.shape[0])

                result += F.softmax(predict, dim=1)[:, 1].tolist()
                labels += label[:, 1].tolist()

                fusion_features.append(fusion.detach().cpu().numpy())

        auc = roc_auc_score(labels, result)
        result_np, labels_np = np.array(result), np.array(labels)
        result_np[result_np > 0.5] = 1
        result_np[result_np <= 0.5] = 0

        metric = precision_recall_fscore_support(labels_np, result_np, average='micro')

        report = classification_report(labels_np, result_np, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        cm = confusion_matrix(labels_np, result_np, labels=[0, 1])

        all_fusions = np.concatenate(fusion_features, axis=0) if len(fusion_features) > 0 else np.array([])

        return [auc] + list(metric) + recall, cm, all_fusions, labels_np, result_np

    def save_test_attention_weights(self):
        """
        只收集测试集所有被试的注意力权重，保存为 npy。
        输出：
            attnWeights_test_HO.npy    形状 (N_test, W, 96)
            attnWeights_test_AAL.npy   形状 (N_test, W, 90)
            attnWeights_test_CC200.npy 形状 (N_test, W, 200)
            labels_test.npy            形状 (N_test,)
        所有张量按被试顺序对齐。
        """
        self.model.eval()

        attn_ho_list = []
        attn_aal_list = []
        attn_cc200_list = []
        labels_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                features1, features2, features3, non_imaging, label = batch
                _, _, attention = self.model(
                    features1.cuda(), features2.cuda(),
                    features3.cuda(), non_imaging.cuda()
                )
                attn_ho_list.append(attention['node-attention1'].cpu())
                attn_aal_list.append(attention['node-attention2'].cpu())
                attn_cc200_list.append(attention['node-attention3'].cpu())
                labels_list.append(label.cpu())

        attn_ho = torch.cat(attn_ho_list, dim=0).numpy()
        attn_aal = torch.cat(attn_aal_list, dim=0).numpy()
        attn_cc200 = torch.cat(attn_cc200_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "attnWeights_test_AAL.npy", attn_ho)
        np.save(self.save_path / "attnWeights_test_HO.npy", attn_aal)
        np.save(self.save_path / "attnWeights_test_CC200.npy", attn_cc200)
        np.save(self.save_path / "labels_test.npy", labels)

        self.logger.info(
            f"Test attention saved: HO {attn_ho.shape}, "
            f"AAL {attn_aal.shape}, CC200 {attn_cc200.shape}, "
            f"labels {labels.shape}"
        )

    def save_result(self,  test_fusions=None, test_labels=None):
        """保存训练过程、最佳模型权重以及对应的测试集 fusion 特征和标签"""
        self.save_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), self.save_path / "model.pt")

        if test_fusions is not None and test_labels is not None:
            np.save(self.save_path / "best_test_fusions.npy", test_fusions)
            np.save(self.save_path / "best_test_labels.npy", test_labels)
            self.logger.info(
                f"Test fusion features saved to 'best_test_fusions.npy' Shape: {test_fusions.shape}")
            self.logger.info(
                f"Test labels saved to 'best_test_labels.npy' Shape: {test_labels.shape}")

    def train(self):
        self.current_step = 0
        best_val_AUC = 0
        best_test_ACC = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0

        # ===== 早停状态 =====
        best_val_loss = float('inf')
        epochs_no_improve = 0

        num_train_samples = len(self.train_dataloader.dataset)

        for epoch in range(self.epochs):
            is_warmup = epoch < self.warmup_epochs
            is_measure = (self.warmup_epochs <= epoch <
                          self.warmup_epochs + self.measure_epochs)

            self._sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            epoch_start_time = time.time()

            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])

            self._sync()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            gpu_stat = self._gpu_snapshot()

            if is_measure:
                self.perf_records.append({

                    "train_time_sec": epoch_time,
                    "num_train_samples": num_train_samples,
                    "batch_size": self.batch_size,
                    "gpu_peak_allocated_mb": gpu_stat["peak_alloc_mb"],
                    "gpu_peak_reserved_mb": gpu_stat["peak_reserved_mb"],
                    "gpu_global_used_mb": gpu_stat["global_used_mb"],
                    "gpu_global_used_pct": gpu_stat["global_used_pct"] * 100.0,
                })
                self._save_perf_records()

            val_result, _, _, _, _ = self.test_per_epoch(
                self.val_dataloader, self.val_loss, self.val_accuracy
            )

            test_result, test_cm, test_fusions, test_labels, test_probs = self.test_per_epoch(
                self.test_dataloader, self.test_loss, self.test_accuracy
            )

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Phase:{"warmup" if is_warmup else ("measure" if is_measure else "post")}',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}',
                f'Val Loss{self.val_loss.avg: .3f}',
                f'Val AUC:{val_result[0]:.4f}',
            ]))

            current_val_loss = self.val_loss.avg
            if epoch < self.min_epochs:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
            else:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1


            if val_result[0] >= best_val_AUC:
                self.save_result( test_fusions=test_fusions, test_labels=test_labels)
                best_val_AUC = val_result[0]
                best_test_ACC = self.test_accuracy.avg
                best_test_AUC = test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
                self.save_confusion_matrix(test_cm, epoch)
                self.save_test_predictions(test_probs, test_labels)
                if self.save_test_attn_weights:
                    self.save_test_attention_weights()

            if epochs_no_improve >= self.patience:
                break

        self._save_perf_records()
        return [best_test_ACC, best_test_AUC, best_test_sen, best_test_spec]