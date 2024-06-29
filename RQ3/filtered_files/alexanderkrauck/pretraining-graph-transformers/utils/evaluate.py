"""
Utility functions/classes for doing evaluation on the model.

Copyright (c) 2023 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

from sklearn.metrics import roc_auc_score
import numpy as np
from functools import partial

import tracemalloc
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
import wandb


from transformers import TrainerCallback, TrainerControl


#generated by chatgpt4
class CustomEarlyStoppingCallback(TrainerCallback):

    def __init__(self, early_stopping_patience=-1, early_stopping_threshold=0.01, **kwargs):
        self.best_score = None
        self.patience = early_stopping_patience
        self.bad_epochs = 0
        self.threshold = early_stopping_threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # metric_for_best_model is the evaluation metric set when defining TrainingArguments
        if self.patience == -1:
            return

        metric_name = args.metric_for_best_model
        if "eval_"+metric_name in metrics:
            score = metrics["eval_"+metric_name]
        elif "evel_loss" in metrics:
            score = metrics["evel_loss"]
        else:
            return

        if self.best_score is None or self.is_metric_better(args, self.best_score, score):
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            if self.score_diff_threshold_reached(args, self.best_score, score):
                control.should_training_stop = True

    def is_metric_better(self, args, best_metric, current_metric):
        if args.greater_is_better:
            return current_metric > best_metric
        else:
            return current_metric < best_metric

    def score_diff_threshold_reached(self, args, best_score, current_score):
        if args.greater_is_better:
            return (best_score - current_score) >= self.threshold
        else:
            return (current_score - best_score) >= self.threshold


def stable_sigmoid(x, C=40.0):
      # clamp range; you might need to adjust this depending on your use case
    x_clamped = np.clip(x, -C, C)
    return 1 / (1 + np.exp(-x_clamped))

def multi_label_metrics(eval_pred: tuple, label_names):
    """
    Computes ROC AUC and accuracy per label, as well as their averages.

    Args:
        eval_pred: Tuple of (logits, labels) from the evaluation dataloader.
        label_names: List of label names.

    Returns:
        metrics: Dictionary of metrics.
    """

    logits, labels = eval_pred
    logits = logits[0]
    probs = stable_sigmoid(logits)

    # Create mask for valid (non-NaN) labels
    valid_labels_mask = ~np.isnan(labels)

    roc_auc_per_label = {}
    accuracy_per_label = {}
    for label_idx in range(labels.shape[1]):  # Iterate over each label
        # Use the mask to select only valid labels and corresponding probabilities
        valid_probs = probs[valid_labels_mask[:, label_idx], label_idx]
        valid_labels = labels[valid_labels_mask[:, label_idx], label_idx]

        if len(valid_labels) > 0:  # Only compute metrics if there are valid labels
            roc_auc_per_label[f"{label_names[label_idx]}_roc_auc"] = roc_auc_score(
                valid_labels, valid_probs
            )

            predicted_labels = valid_probs > 0.5
            accuracy_per_label[f"{label_names[label_idx]}_accuracy"] = (
                predicted_labels == valid_labels
            ).mean()

    # Organize metrics into a nested dictionary

    mean_metrics = {
        "mean_roc_auc": np.mean(list(roc_auc_per_label.values())),
        "mean_accuracy": np.mean(list(accuracy_per_label.values())),
    }

    return {**mean_metrics, **roc_auc_per_label, **accuracy_per_label}

def regression_metrics(eval_pred: tuple, label_names, target_scaler):
    logits, labels = eval_pred
    logits = logits[0]
    maes = []
    return_metrics = {}
    if isinstance(label_names, str):
        label_names = [label_names]
    for idx, label_name in enumerate(label_names):
        mask = ~np.isnan(labels[:, idx])
        masked_labels = labels[:, idx][mask]
        masked_logits = logits[:, idx][mask]
        mse = np.mean(np.square(masked_logits - masked_labels))
        mae = np.mean(np.abs(masked_logits - masked_labels))
        maes.append(mae)
        return_metrics[f"{label_name}_mse"] = mse
        return_metrics[f"{label_name}_mae"] = mae
    if len(maes) > 1:
        return_metrics["mean_mae"] = np.mean(maes)

    if target_scaler is not None:
        unscaled_logits = target_scaler.inverse_transform(logits)
        unsclaed_targets = target_scaler.inverse_transform(labels)
        for idx, label_name in enumerate(label_names):
            mask = ~np.isnan(labels[:, idx])
            masked_labels = unscaled_logits[:, idx][mask]
            masked_logits = unsclaed_targets[:, idx][mask]
            mse = np.mean(np.square(masked_logits - masked_labels))
            mae = np.mean(np.abs(masked_logits - masked_labels))
            return_metrics[f"{label_name}_unscaled_mse"] = mse
            return_metrics[f"{label_name}_unscaled_mae"] = mae

    return return_metrics


def prepare_evaluation_for_training(pretraining: bool, dataset_name: str, target_scaler = None, **kwargs):
    if not pretraining:
        if dataset_name in ["tox21_original", "tox21"]:
            return partial(
                multi_label_metrics,
                label_names=[
                    "NR.AhR",
                    "NR.AR",
                    "NR.AR.LBD",
                    "NR.Aromatase",
                    "NR.ER",
                    "NR.ER.LBD",
                    "NR.PPAR.gamma",
                    "SR.ARE",
                    "SR.ATAD5",
                    "SR.HSE",
                    "SR.MMP",
                    "SR.p53",
                ],
            )
        if dataset_name == "ZINC":
            return partial(
                regression_metrics,
                label_names="penalized logP",
                target_scaler = target_scaler
            )
            return None
        if dataset_name == "qm9":
            return partial(
                regression_metrics,
                label_names=[
                    "A",
                    "B",
                    "C",
                    "mu",
                    "alpha",
                    "homo",
                    "lumo",
                    "gap",
                    "r2",
                    "zpve",
                    "u0",
                    "u298",
                    "h298",
                    "g298",
                    "cv",
                    "u0_atom",
                    "u298_atom",
                    "h298_atom",
                    "g298_atom",
                ],
                target_scaler = target_scaler
            )
        raise ValueError("Invalid dataset name for fine tuning.")
    else:
        if dataset_name == "pcqm4mv2":
            return None
        if dataset_name == "pcba":
            return None
        if dataset_name == "qm9":
            return None
        raise ValueError("Invalid dataset name for pretraining.")


# Define your custom callback
class MemoryProfilerCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Check the current training step
        if state.global_step % 500 == 0:
            # Start tracing memory allocations
            tracemalloc.start()

            # Get snapshot of current memory consumption
            snapshot = tracemalloc.take_snapshot()

            # Display the top 10 lines consuming the memory
            top_stats = snapshot.statistics("lineno")

            for i, stat in enumerate(top_stats[:30]):
                wandb.log({f"memory_stat_{i}": str(stat)}, step=state.global_step)
                print(stat)

            # Stop tracing memory allocations
            tracemalloc.stop()