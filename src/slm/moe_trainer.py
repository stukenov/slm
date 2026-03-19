"""Domain-aware MoE Trainer with curriculum learning and router metrics.

Features:
- Domain-aware batching via custom data collator
- Curriculum schedule: pure domain → mixed batches over training
- Per-domain eval loss tracking
- Router entropy and expert utilization logging
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


class DomainBatchSampler(Sampler):
    """Sampler that creates domain-pure or mixed batches based on curriculum phase.

    Curriculum schedule (based on fraction of total steps):
      0.0 - 0.2:  100% domain-pure batches
      0.2 - 0.6:  70% domain / 30% mixed
      0.6 - 1.0:  50% domain / 50% mixed
    """

    def __init__(self, domain_ids: list[int], batch_size: int, total_steps: int, seed: int = 42):
        self.batch_size = batch_size
        self.total_steps = max(total_steps, 1)
        self.seed = seed

        # Group indices by domain
        self.domain_indices: dict[int, list[int]] = defaultdict(list)
        for idx, dom in enumerate(domain_ids):
            self.domain_indices[dom].append(idx)

        self.all_indices = list(range(len(domain_ids)))
        self.domains = sorted(self.domain_indices.keys())
        self._current_step = 0

    def set_step(self, step: int):
        self._current_step = step

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self._current_step)

        # Determine curriculum phase
        frac = self._current_step / self.total_steps
        if frac < 0.2:
            domain_prob = 1.0
        elif frac < 0.6:
            domain_prob = 0.7
        else:
            domain_prob = 0.5

        # Shuffle all domain index pools
        shuffled_pools = {}
        for dom, indices in self.domain_indices.items():
            pool = indices.copy()
            rng.shuffle(pool)
            shuffled_pools[dom] = pool

        all_shuffled = self.all_indices.copy()
        rng.shuffle(all_shuffled)

        total_samples = len(self.all_indices)
        yielded = 0
        mixed_ptr = 0

        while yielded < total_samples:
            if rng.random() < domain_prob:
                # Domain-pure batch
                dom = rng.choice(self.domains)
                pool = shuffled_pools[dom]
                if len(pool) < self.batch_size:
                    rng.shuffle(pool)
                    shuffled_pools[dom] = pool
                batch = pool[:self.batch_size]
                pool[:self.batch_size] = []
            else:
                # Mixed batch
                end = min(mixed_ptr + self.batch_size, total_samples)
                batch = all_shuffled[mixed_ptr:end]
                mixed_ptr = end
                if mixed_ptr >= total_samples:
                    rng.shuffle(all_shuffled)
                    mixed_ptr = 0

            yield from batch
            yielded += len(batch)

    def __len__(self):
        return len(self.all_indices)


class MoEDomainTrainer(Trainer):
    """Custom Trainer for domain-aware MoE training."""

    def __init__(self, *args, domain_curriculum: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_curriculum = domain_curriculum
        self._domain_sampler = None

    def get_train_dataloader(self) -> DataLoader:
        if not self.domain_curriculum or "domain_id" not in self.train_dataset.column_names:
            return super().get_train_dataloader()

        domain_ids = self.train_dataset["domain_id"]
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) // self.args.per_device_train_batch_size * int(self.args.num_train_epochs)
        )

        self._domain_sampler = DomainBatchSampler(
            domain_ids=domain_ids,
            batch_size=self.args.per_device_train_batch_size,
            total_steps=total_steps,
            seed=self.args.seed,
        )

        # Remove domain_id from the dataset for the collator
        train_ds = self.train_dataset.remove_columns(["domain_id"])

        return DataLoader(
            train_ds,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._domain_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Update sampler step for curriculum
        if self._domain_sampler is not None:
            self._domain_sampler.set_step(self.state.global_step)

        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Log router metrics periodically
        if self.state.global_step % (self.args.logging_steps * 5) == 0:
            self._log_router_metrics(model)

        return loss

    def _log_router_metrics(self, model):
        """Compute and log router entropy and expert utilization."""
        try:
            router_entropies = []
            expert_counts = defaultdict(int)
            total_tokens = 0

            for layer in model.model.layers:
                if hasattr(layer.mlp, "gate"):
                    gate = layer.mlp.gate
                    if hasattr(gate, "weight"):
                        # Approximate: compute softmax entropy of router weights
                        w = gate.weight.data.float()
                        probs = torch.softmax(w.mean(dim=1), dim=0)
                        entropy = -(probs * (probs + 1e-10).log()).sum().item()
                        router_entropies.append(entropy)

            if router_entropies:
                avg_entropy = sum(router_entropies) / len(router_entropies)
                max_entropy = math.log(model.config.num_local_experts)
                self.log({
                    "router/avg_entropy": avg_entropy,
                    "router/normalized_entropy": avg_entropy / max_entropy,
                })
        except Exception:
            pass  # Don't crash training for logging failures

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Per-domain eval if dataset has domain_id
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is not None and "domain_id" in eval_ds.column_names:
            self._per_domain_eval(eval_ds, metric_key_prefix)

        return output

    def _per_domain_eval(self, eval_ds, prefix: str):
        """Evaluate loss per domain."""
        domain_ids = eval_ds["domain_id"]
        unique_domains = sorted(set(domain_ids))

        for dom in unique_domains:
            indices = [i for i, d in enumerate(domain_ids) if d == dom]
            if len(indices) < 10:
                continue
            subset = eval_ds.select(indices).remove_columns(["domain_id"])
            try:
                metrics = super().evaluate(subset, metric_key_prefix=f"{prefix}_domain{dom}")
                logger.info("Domain %d eval_loss: %.4f (%d samples)",
                            dom, metrics.get(f"{prefix}_domain{dom}_loss", 0), len(indices))
            except Exception as e:
                logger.warning("Domain %d eval failed: %s", dom, e)
