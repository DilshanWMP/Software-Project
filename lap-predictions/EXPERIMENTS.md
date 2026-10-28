# EXPERIMENTS.md
Experiment log for `lap-predictions` project. One row per training run.

> All test-set evaluations use the **frozen test split** from `data/split_manifest.csv`
> (SHA-256: `d8aeea938c7638874b18c11bf970250ac0c4578120de2332163fa350f2f5e90b`).

---

## Run Log

| Run ID | Date | Stage | Arch | Head LR | FT LR | Head Ep | FT Ep (stopped) | Val BalAcc | Test BalAcc | ROC-AUC | Notes |
|--------|------|-------|------|---------|-------|---------|-----------------|------------|-------------|---------|-------|
| `v1-baseline` | 2026-10-28 | F3 | `efficientnet_b0` | 1e-3 | 1e-4 | 5 | 10 | 0.8056 | TBD (F4) | TBD (F4) | Baseline; label smoothing=0.05; drop_rate=0.25; WeightedRandomSampler; seed=42; early stopping at ep 10 (best Val BalAcc 0.8056) |

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Two-step transfer learning | Protects ImageNet features from large initial gradient; head-only warm-up stabilises final linear layer before full unfreeze |
| Label smoothing 0.05 | Prevents overconfidence on small dataset |
| drop_rate 0.25 | Light regularisation for EfficientNet-B0 on N~200 train set |
| WeightedRandomSampler | ~70/30 class imbalance; ensures minority class seen equally per epoch |
| Cosine Annealing (Step B) | Avoids sharp LR drops; compatible with early stopping |
| Temperature calibration on val logits | Separates calibration set from test; prevents leakage |
| Threshold @ 90% sensitivity | Clinical requirement: missing a tool is more dangerous than a false alarm |
