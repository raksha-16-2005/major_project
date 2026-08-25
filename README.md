# Part 2 — Student Model & Knowledge Distillation

**Owner:** [Teammate 2]
**Resume headline:** *Implemented the core ML of a medical-imaging distillation system — a ResNet-50
student trained with a temperature-scaled Hinton knowledge-distillation loss (multi-class + multi-label
branches) and a unified, auto-tuning trainer.*

**Code in this folder (`code/`):**
`student.py` · `distill.py` · `train.py` · `utils.py` (shared infra) · `02_train_baseline.py` · `03_train_distill.py`

> This is the **heart of the project**. If you own this part, you own "the model and the distillation."

---

## 1. What this part does

This part trains the **student network** in two modes and compares them:
- **`baseline`** — a ResNet-50 trained from scratch on hard labels only (standard supervised learning).
- **`distill`** — the same ResNet-50 trained with the **Hinton knowledge-distillation loss**, learning
  from both the hard labels *and* the frozen DINOv2 teacher's soft logits (produced in Part 1).

The scientific question this part answers: **does distilling DINOv2's knowledge into a small ResNet-50
beat training that ResNet-50 alone?** The unified trainer produces both models so the comparison is
apples-to-apples (same architecture, same schedule, only the loss differs).

---

## 2. How it works internally

### `student.py` — the student network
- `build_student(n_classes)` = torchvision **ResNet-50** with `weights=None` (from scratch) and its
  final `fc` replaced by `nn.Linear(2048 → n_classes)` so the head matches each dataset's class count.
- `gradcam_target_layer()` returns `model.layer4[-1]` — the last conv block — which Part 3 hooks for
  Grad-CAM++. (Defined here because it's a property of the architecture.)

### `distill.py` — `HintonKDLoss` (the intellectual core)
The loss is the classic Hinton (2015) formulation:

```
L = α · CE(student, hard_labels)  +  (1 − α) · T² · KL( softmax(student/T) ‖ softmax(teacher/T) )
```

- **`α` (alpha = 0.2)** weights the hard-label cross-entropy; **(1−α) = 0.8** weights the soft
  teacher-matching term. So the student leans mostly on the teacher's soft targets.
- **`T` (temperature = 2.0)** softens both distributions so the student learns the teacher's *relative*
  class confidences ("dark knowledge"), not just the argmax.
- **The `T²` factor** rescales the KL gradient, which otherwise shrinks by 1/T² — a detail from the
  original paper that keeps the soft-loss magnitude comparable to the hard loss.
- **Two branches:**
  - *Multi-class* (`_forward_multiclass`): CE + temperature-scaled **KL divergence** between softmaxes.
  - *Multi-label* (`_forward_multilabel`, used by ChestMNIST): CE → **BCE-with-logits**, and KL is
    replaced by **BCE between temperature-softened sigmoid distributions** (`sigmoid(teacher/T)` as
    pseudo-soft targets, `×T²`). This is the correct generalisation of soft-target distillation to
    independent per-label Bernoullis.

### `train.py` — the unified trainer (`train_student(mode, subset, cfg)`)
One function trains either mode; the only difference is the loss and the data it feeds.

- **`_DistillDataset`** — the key performance fix. Distillation needs each image paired with its
  index-aligned teacher logit. The naive approach (single-threaded per-sample index lookup) ran at
  **~70 s/batch** on the cluster. Wrapping the base dataset so `__getitem__` returns `(x, y, teacher_logit[i])`
  lets a **normal multi-worker DataLoader** yield the triple — restoring full training speed. MedMNIST
  samples are deterministic by index, so the pairing is exact.
- **`auto_tune_batch_size`** (from `utils.py`) — probes a forward+backward pass at 64→32→16→8 and
  returns the largest that fits in VRAM, so the same code runs on an A100 or a small GPU without OOM.
- **Mixed precision** (fp16 autocast + `GradScaler`) on CUDA for speed/memory.
- **Optimiser/schedule:** Adam, lr=1e-3, `MultiStepLR` milestones [5,7] × 0.1, 10 epochs.
- **Checkpointing:** best model by **validation AUC** is saved; a `{mode}_history.json` records
  per-epoch loss/AUC/acc and the best result (used later by Part 3's evaluation and Part 4's report).
- **Idempotent/resumable:** if the checkpoint + history already exist, it returns the cached best.
- `_eval_loader` / `_sk_auc` compute val AUC robustly via sklearn (handling binary, multi-class-OVR,
  and multi-label-macro cases).

### `utils.py` — shared infrastructure (lives here, used by all parts)
Seeding, logging (`get_logger`), config loading (`load_config` resolves all paths relative to `demo/`),
`per_subset_paths` (the canonical cache/checkpoint/log directory layout), VRAM probe, and the
`auto_tune_batch_size` tuner above.

### Scripts
- `02_train_baseline.py` — trains the from-scratch ResNet-50 for each subset.
- `03_train_distill.py` — trains the distilled ResNet-50 using cached teacher logits.

---

## 3. Real results — does distillation win?

Distillation is **genuinely mixed** — and being honest about that is a strength at viva/interview.
It **helps on accuracy in 5/12 datasets** and is roughly a wash on AUC.

| Dataset | Baseline Acc | Distill Acc | Δ Acc | Verdict |
|---|---|---|---|---|
| pathmnist | 0.8877 | **0.9400** | **+0.0523** | ✅ distill wins |
| dermamnist | 0.7097 | **0.7421** | **+0.0324** | ✅ distill wins |
| retinamnist | 0.5225 | **0.5575** | **+0.0350** | ✅ distill wins |
| pneumoniamnist | 0.8365 | **0.8590** | **+0.0225** | ✅ distill wins |
| organsmnist | 0.7914 | **0.7952** | +0.0038 | ✅ distill wins (marginal) |
| octmnist | 0.7680 | 0.6690 | −0.0990 | ❌ baseline wins |
| breastmnist | 0.8654 | 0.8013 | −0.0641 | ❌ baseline wins |
| tissuemnist | 0.7043 | 0.6514 | −0.0529 | ❌ baseline wins |
| bloodmnist | 0.9597 | 0.9310 | −0.0287 | ❌ baseline wins |
| organamnist | 0.9442 | 0.9355 | −0.0087 | ❌ baseline wins |
| organcmnist | 0.9136 | 0.9094 | −0.0042 | ≈ tie |
| chestmnist (ML) | 0.9480 | 0.9477 | −0.0003 | ≈ tie |

**The pattern:** distillation wins exactly where Part 1's **teacher-probe beat the baseline**
(path/derma/retina). Where the frozen teacher is *worse* than a from-scratch ResNet (oct/breast/tissue),
distilling from it **drags the student down**. Distillation transfers the teacher's knowledge — including
its weaknesses. **A student can't out-learn a teacher that's worse than learning from scratch.**

---

## 4. Likely interview questions (with strong answers)

**Q: Explain the distillation loss term by term.**
A: `L = α·CE + (1−α)·T²·KL`. CE is the normal hard-label loss. The KL term matches the student's
temperature-softened softmax to the teacher's — transferring "dark knowledge" (relative class
similarities). T=2 softens both; the T² factor undoes the 1/T² gradient shrink so the soft loss stays
comparably weighted. α=0.2 means 80% of the signal comes from the teacher.

**Q: What is temperature and why T² ?**
A: Temperature flattens the softmax so small logit differences become learnable signal. Because
softmax(z/T) gradients scale like 1/T², Hinton multiplies the distillation loss by T² to keep its
magnitude on par with the hard-label loss regardless of T.

**Q: How do you distill for a multi-label task like ChestMNIST?**
A: KL over a single softmax doesn't apply — each of the 14 findings is an independent Bernoulli. I
replace CE with BCE-with-logits and the soft term with BCE between temperature-softened sigmoids of
teacher vs student logits, scaled by T². That's the correct per-label generalisation.

**Q: Distillation didn't always win. Isn't that a failure?**
A: No — it's the finding. Distillation is bounded by teacher quality. It wins precisely on the datasets
where the frozen DINOv2 teacher's linear probe beats the from-scratch baseline, and loses where the
teacher is weaker. That's a clean, defensible, causal story, not noise.

**Q: What was the 70 s/batch bug?**
A: I paired teacher logits to images with a single-threaded per-sample index pass, which serialised
the whole loader. I wrapped the dataset (`_DistillDataset`) so each `__getitem__` returns the image, its
label, and its cached logit — letting a multi-worker DataLoader parallelise it back to normal speed.

**Q: Why auto-tune batch size?**
A: To run the same code across GPUs without hand-editing. It tries a real forward+backward at
descending batch sizes and returns the first that fits, catching `OutOfMemoryError` and retrying smaller.

---

## 5. How to run this part
```bash
python code/02_train_baseline.py     # from-scratch ResNet-50 per subset
python code/03_train_distill.py      # distilled ResNet-50 (needs Part 1's cached logits)
```
Hyperparameters (T, α, lr, epochs, milestones) live in `demo/config.yaml`.
