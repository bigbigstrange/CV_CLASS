"""Training and evaluation loops for Tier 1 models (SegDecNet + SuperSimpleNet)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from engine.losses import CombinedSegLoss, SegDecLoss, SSNLoss
from engine.metrics import (
    ApMetricsAccumulator,
    aggregate_metrics,
    compute_batch_iou,
    compute_batch_metrics,
    search_best_threshold,
    search_best_threshold_from_scores,
)
from models.factory import create_model


class SegmentationTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = create_model(config).to(self.device)
        self.model_type = config.model_type
        self.criterion = self._build_criterion()
        self.optimizer, self.scheduler = self._build_optimizer()
        self.history: list[dict] = []
        self.start_epoch = 1
        self.best_f1 = -1.0
        self.best_metric_value = -1.0
        self.eval_threshold = config.eval_threshold
        self._interrupted = False

        self._load_history_file()
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from, resume_training=True)
        self._sync_best_from_history()

    def _build_criterion(self):
        if self.model_type == "segdec":
            return SegDecLoss(
                seg_pos_weight=self.config.seg_pos_weight,
                decision_pos_weight=self.config.decision_pos_weight,
                decision_weight=self.config.decision_loss_weight,
            ).to(self.device)
        if self.model_type == "ssn":
            th = self.config.ssn_config.get("seg_loss_th", 0.5)
            return SSNLoss(seg_loss_th=th)
        return CombinedSegLoss(
            pos_weight=self.config.pos_weight,
            bce_weight=self.config.bce_weight,
            dice_weight=self.config.dice_weight,
            focal_weight=self.config.focal_weight,
            focal_gamma=self.config.focal_gamma,
            focal_alpha=self.config.focal_alpha,
        ).to(self.device)

    def _build_optimizer(self):
        if self.model_type == "ssn":
            optim, sched = self.model.get_optimizers(epochs=self.config.epochs)
            return optim, sched
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer, None

    @property
    def history_path(self) -> Path:
        return self.config.checkpoint_dir / "history.json"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.config.checkpoint_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.config.checkpoint_dir / "best.pt"

    def _metric_kwargs(self) -> dict:
        return {
            "threshold": self.eval_threshold,
            "min_defect_pixels": self.config.min_defect_pixels,
        }

    def _get_best_metric_value(self, metrics: dict) -> float:
        name = self.config.best_metric
        if name == "f1":
            return float(metrics.get("f1", -1.0))
        if name in {"ap_det", "ap"}:
            return float(metrics.get(name, metrics.get("ap_det", metrics.get("ap", -1.0))))
        return float(metrics.get(name, -1.0))

    def _sync_best_from_history(self) -> None:
        for record in self.history:
            val = record.get("val")
            if not val:
                continue
            self.best_f1 = max(self.best_f1, float(val.get("f1", -1.0)))
            self.best_metric_value = max(
                self.best_metric_value,
                self._get_best_metric_value(val),
            )

    def _load_history_file(self) -> None:
        if not self.history_path.exists():
            return
        with self.history_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            self.history = data

    def _ssn_mask(self, masks: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(
            masks,
            size=(self.model.fh, self.model.fw),
            mode="bilinear",
            align_corners=True,
        )
        return torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))

    def _forward_train(self, batch: dict) -> tuple[torch.Tensor, dict]:
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        if self.model_type == "segdec":
            outputs = self.model(images)
            loss = self.criterion(outputs["seg_logits"], outputs["decision_logits"], masks, labels)
            return loss, outputs

        if self.model_type == "ssn":
            mask_down = self._ssn_mask(masks)
            is_segmented = batch["is_segmented"].to(self.device)
            anomaly_map, anomaly_score, mask_out, label_out = self.model(
                images, mask_down, labels
            )
            loss = self.criterion(anomaly_map, anomaly_score, mask_out, label_out, is_segmented)
            batch_size = masks.size(0)
            seg_up = F.interpolate(
                anomaly_map[:batch_size],
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return loss, {
                "seg_logits": seg_up,
                "decision_logits": anomaly_score[:batch_size],
            }

        logits = self.model(images)
        loss = self.criterion(logits, masks)
        return loss, {"seg_logits": logits, "decision_logits": None}

    @torch.no_grad()
    def _forward_eval(self, batch: dict) -> dict:
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        if self.model_type == "ssn":
            anomaly_map, anomaly_score = self.model(images)
            seg_up = F.interpolate(anomaly_map, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            return {
                "seg_logits": seg_up,
                "decision_logits": anomaly_score,
                "masks": masks,
                "labels": labels,
            }

        if self.model_type == "segdec":
            outputs = self.model(images)
            outputs["masks"] = masks
            outputs["labels"] = labels
            return outputs

        logits = self.model(images)
        return {"seg_logits": logits, "decision_logits": None, "masks": masks, "labels": labels}

    def _compute_loss_eval(self, batch: dict, outputs: dict) -> float:
        masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        if self.model_type == "segdec":
            return float(
                self.criterion(outputs["seg_logits"], outputs["decision_logits"], masks, labels).item()
            )
        if self.model_type == "ssn":
            from engine.losses import focal_loss_bce

            mask_down = self._ssn_mask(masks)
            seg_down = F.interpolate(
                outputs["seg_logits"],
                size=(self.model.fh, self.model.fw),
                mode="bilinear",
                align_corners=False,
            )
            seg_loss = focal_loss_bce(torch.sigmoid(seg_down), mask_down)
            cls_loss = focal_loss_bce(torch.sigmoid(outputs["decision_logits"]), labels)
            return float((seg_loss + cls_loss).item())
        return float(self.criterion(outputs["seg_logits"], masks).item())

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        losses: list[float] = []
        metrics_list: list[dict[str, float]] = []

        for batch in loader:
            self.optimizer.zero_grad()
            loss, outputs = self._forward_train(batch)
            loss.backward()

            if self.model_type == "ssn" and self.config.ssn_clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            losses.append(float(loss.item()))

            labels = batch["label"].to(self.device)
            masks = batch["mask"].to(self.device)
            metrics_list.append(
                compute_batch_metrics(
                    outputs["seg_logits"].detach(),
                    masks,
                    labels,
                    decision_logits=outputs.get("decision_logits"),
                    **self._metric_kwargs(),
                )
            )

        metrics = aggregate_metrics(metrics_list)
        metrics["loss"] = sum(losses) / len(losses)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, search_threshold: bool | None = None) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        ap_acc = ApMetricsAccumulator(
            max_pixel_samples=self.config.pixel_ap_max_samples,
            score_mode=self.config.image_score_mode,
            topk=self.config.image_score_topk,
            target_fpr=self.config.target_fpr,
        )
        use_decision_for_search = self.model_type in {"segdec", "ssn"}
        do_search = self.config.threshold_search if search_threshold is None else search_threshold

        for batch in loader:
            outputs = self._forward_eval(batch)
            losses.append(self._compute_loss_eval(batch, outputs))
            ap_acc.add_batch(
                outputs["seg_logits"],
                outputs["masks"],
                outputs["labels"],
                outputs.get("decision_logits"),
            )

        ap_metrics = ap_acc.finalize()
        labels_np = np.concatenate(ap_acc.labels)
        det_scores_np = np.concatenate(ap_acc.det_scores)

        if do_search and use_decision_for_search:
            self.eval_threshold, metrics = search_best_threshold_from_scores(
                det_scores_np,
                labels_np,
                threshold_min=self.config.threshold_min,
                threshold_max=self.config.threshold_max,
                threshold_step=self.config.threshold_step,
                criterion=self.config.threshold_criterion,
                min_recall=self.config.threshold_min_recall,
            )
        elif do_search:
            seg_logits_list: list[torch.Tensor] = []
            mask_list: list[torch.Tensor] = []
            label_list: list[torch.Tensor] = []
            decision_list: list[torch.Tensor] = []
            for batch in loader:
                outputs = self._forward_eval(batch)
                seg_logits_list.append(outputs["seg_logits"].cpu())
                mask_list.append(outputs["masks"].cpu())
                label_list.append(outputs["labels"].cpu())
                if outputs.get("decision_logits") is not None:
                    decision_list.append(outputs["decision_logits"].cpu())
            self.eval_threshold, metrics = search_best_threshold(
                torch.cat(seg_logits_list, dim=0),
                torch.cat(mask_list, dim=0),
                torch.cat(label_list, dim=0),
                threshold_min=self.config.threshold_min,
                threshold_max=self.config.threshold_max,
                threshold_step=self.config.threshold_step,
                min_defect_pixels=self.config.min_defect_pixels,
                decision_logits=torch.cat(decision_list, dim=0) if decision_list else None,
            )
        else:
            self.eval_threshold, metrics = search_best_threshold_from_scores(
                det_scores_np,
                labels_np,
                threshold_min=self.eval_threshold,
                threshold_max=self.eval_threshold + 1e-6,
                threshold_step=self.config.threshold_step,
                criterion=self.config.threshold_criterion,
                min_recall=self.config.threshold_min_recall,
            )

        iou_values: list[float] = []
        for batch in loader:
            outputs = self._forward_eval(batch)
            iou_values.append(
                compute_batch_iou(outputs["seg_logits"], outputs["masks"], self.eval_threshold)
            )
        metrics["iou"] = sum(iou_values) / len(iou_values) if iou_values else 0.0
        metrics.update(ap_metrics)
        metrics["loss"] = sum(losses) / len(losses)
        metrics["threshold"] = self.eval_threshold
        metrics["image_score_mode"] = self.config.image_score_mode
        metrics["threshold_criterion"] = self.config.threshold_criterion
        return metrics

    def _refresh_best_checkpoint_if_needed(self) -> None:
        if self.best_metric_value <= 0 or not self.history:
            return

        stored = -1.0
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(
                self.best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            stored = self._get_best_metric_value(checkpoint.get("metrics", {}))

        if self.best_metric_value <= stored + 1e-6:
            return

        best_record = max(
            (record for record in self.history if record.get("val")),
            key=lambda record: self._get_best_metric_value(record["val"]),
        )
        best_epoch = int(best_record["epoch"])
        if best_epoch != self.start_epoch - 1:
            return

        self.save_checkpoint(
            self.best_checkpoint_path,
            best_epoch,
            best_record["val"],
            kind="best",
        )
        print(
            f"[checkpoint] refreshed best.pt from epoch {best_epoch} "
            f"({self.config.best_metric}={self.best_metric_value:.4f})"
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[dict]:
        self._refresh_best_checkpoint_if_needed()
        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                train_metrics = self.train_epoch(train_loader)
                record: dict = {"epoch": epoch, "train": train_metrics}

                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader, search_threshold=True)
                    record["val"] = val_metrics
                    metric_value = self._get_best_metric_value(val_metrics)
                    if metric_value > self.best_metric_value:
                        self.best_f1 = max(self.best_f1, float(val_metrics["f1"]))
                        self.best_metric_value = metric_value
                        self.save_checkpoint(self.best_checkpoint_path, epoch, val_metrics, kind="best")
                        print(
                            f"[checkpoint] new best by {self.config.best_metric}="
                            f"{metric_value:.4f} (epoch {epoch})"
                        )
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} | "
                        f"val loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} "
                        f"rec={val_metrics['recall']:.4f} "
                        f"ap={val_metrics.get('ap_det', val_metrics.get('ap', 0.0)):.4f} "
                        f"r@fpr={val_metrics.get('recall_at_fpr', 0.0):.4f} "
                        f"iou={val_metrics['iou']:.4f} thr={val_metrics['threshold']:.2f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f}"
                    )

                self._append_history(record)
                self.save_checkpoint(self.last_checkpoint_path, epoch, record, kind="last")
                self.save_history()

                if self.scheduler is not None:
                    self.scheduler.step()

                if epoch % self.config.save_every == 0 or epoch == self.config.epochs:
                    self.save_checkpoint(
                        self.config.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                        epoch,
                        record,
                        kind="epoch",
                    )
        except KeyboardInterrupt:
            self._interrupted = True
            print("\n[interrupt] training stopped by user, saving resume checkpoint...")
            self._save_interrupt_checkpoint()
            raise

        if not self._interrupted:
            if not self.best_checkpoint_path.exists() and self.history:
                last_record = self.history[-1]
                metrics = last_record.get("val", last_record.get("train", {}))
                epoch = int(last_record["epoch"])
                self.save_checkpoint(self.best_checkpoint_path, epoch, metrics, kind="best")
                print(f"[checkpoint] best.pt was missing, saved from epoch {epoch}")

            print(
                f"Training finished. best_{self.config.best_metric}={self.best_metric_value:.4f} "
                f"best_f1={self.best_f1:.4f} eval_threshold={self.eval_threshold:.2f}"
            )
            print(f"  best : {self.best_checkpoint_path}")
            print(f"  last : {self.last_checkpoint_path}")
        return self.history

    def _append_history(self, record: dict) -> None:
        epoch = record["epoch"]
        self.history = [item for item in self.history if item.get("epoch") != epoch]
        self.history.append(record)
        self.history.sort(key=lambda item: item["epoch"])

    def _save_interrupt_checkpoint(self) -> None:
        epoch = self.start_epoch - 1
        if self.history:
            epoch = self.history[-1]["epoch"]
            metrics = self.history[-1]
        else:
            metrics = {"interrupted": True}
        self.save_checkpoint(self.last_checkpoint_path, epoch, metrics, kind="last")
        self.save_history()
        print(f"[interrupt] saved to {self.last_checkpoint_path}")

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict, kind: str = "last") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = dict(self.config.__dict__)
        config_dict["checkpoint_dir"] = str(config_dict["checkpoint_dir"])
        config_dict["eval_threshold"] = self.eval_threshold
        if config_dict.get("resume_from") is not None:
            config_dict["resume_from"] = str(config_dict["resume_from"])

        payload = {
            "epoch": epoch,
            "best_f1": self.best_f1,
            "best_metric": self.config.best_metric,
            "best_metric_value": self.best_metric_value,
            "eval_threshold": self.eval_threshold,
            "kind": kind,
            "stage": self.config.stage,
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": config_dict,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()

        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(path)

    def load_checkpoint(
        self,
        path: Path,
        resume_training: bool = True,
        reset_best_f1: bool = False,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if reset_best_f1:
            self.best_f1 = -1.0
            self.best_metric_value = -1.0
        else:
            self.best_f1 = float(checkpoint.get("best_f1", self.best_f1))
            self.best_metric_value = float(
                checkpoint.get("best_metric_value", self.best_metric_value)
            )

        self.eval_threshold = float(
            checkpoint.get(
                "eval_threshold",
                checkpoint.get("config", {}).get("eval_threshold", self.eval_threshold),
            )
        )
        self.config.eval_threshold = self.eval_threshold

        if resume_training and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        else:
            self.start_epoch = 1

        print(
            f"Loaded checkpoint: {path} | "
            f"model={checkpoint.get('model_type', self.model_type)} | "
            f"epoch={checkpoint.get('epoch')} | "
            f"best_{self.config.best_metric}={self.best_metric_value:.4f} | "
            f"best_f1={self.best_f1:.4f} | "
            f"eval_threshold={self.eval_threshold:.2f} | "
            f"resume from epoch {self.start_epoch}"
        )

    def save_history(self) -> None:
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2, ensure_ascii=False)
