# Part 1 — Data Pipeline & DINOv2 Teacher

**Owner:** [Teammate 1]
**Resume headline:** *Built the data-ingestion and foundation-model (DINOv2) teacher layer of a
medical-imaging knowledge-distillation system spanning all 12 MedMNIST datasets.*

**Code in this folder (`code/`):**
`data.py` · `cache.py` · `teacher.py` · `00_install_check.py` · `01_cache_teacher.py` · `04_train_teacher_probe.py`

---

## 1. What this part does (the one-paragraph version)

This is the **front of the pipeline**. It (a) loads every MedMNIST dataset into a uniform tensor
format the rest of the system can consume, (b) wraps a **frozen DINOv2 vision-transformer** as the
"teacher", and (c) pre-computes and caches the teacher's features and soft-label logits to disk so
the expensive transformer only runs **once**, not every training epoch. Everything downstream —
student training, distillation, evaluation, the router — reads the artifacts this part produces.

The teacher is **DINOv2 ViT-S/14** (22M params, 384-dim CLS embedding), loaded **frozen** — it is
never fine-tuned. A tiny linear "probe" head is trained on top of its cached features to turn those
features into class logits, which become the *soft targets* for distillation.

---

## 2. How it works internally

### `data.py` — MedMNIST loaders (28×28 → 224×224, gray→3ch, ImageNet norm)
- MedMNIST images are 28×28. DINOv2 needs a multiple of its patch size (14) — we use **224×224**.
  So every image is **bilinearly upsampled 28→224**.
- Grayscale datasets (e.g. chest, pneumonia, OCT, organ*, tissue) are **expanded to 3 channels** so
  the same 3-channel backbone works everywhere, then normalised with **ImageNet mean/std**.
- `get_info(subset)` reads MedMNIST's `INFO` dict for task type, channel count, class count, labels.
- Tasks differ: most are **multi-class**, but **ChestMNIST is multi-label** (14 findings, each 0/1),
  and some are binary/ordinal. `is_multilabel()` gates this, and `collate_targets` squeezes labels
  to `long` for cross-entropy (multi-class) or keeps them `float` for BCE (multi-label).

**Two production-grade bug fixes worth knowing (great interview material):**
1. **`/dev/shm` overflow on Kubeflow.** Notebook containers cap shared memory at ~64 MB. PyTorch's
   default DataLoader worker sharing strategy (`file_descriptor`) routes worker tensors through
   `/dev/shm`, which overflows mid-epoch and the kernel `SIGKILL`s a worker ("Killed"). Fix:
   `torch.multiprocessing.set_sharing_strategy("file_system")` — route through disk-backed temp files.
2. **Windows DataLoader pickling.** A lambda `collate_fn` can't be pickled by Windows workers, so
   `collate_targets` is wrapped in a picklable `_CollateFn` class.

### `teacher.py` — frozen DINOv2 backbone + linear probe head
- **Why load DINOv2 through `timm`, not `torch.hub`?** The official `facebookresearch/dinov2` repo
  uses Python 3.10+ syntax (`float | None`) and won't import on the cluster's **Python 3.8**. `timm`
  ships the same pretrained DINOv2 weights and is import-safe on 3.8. This one-line decision unblocked
  the whole cluster run — a classic "the model isn't the hard part, the environment is" story.
- `load_dinov2()` returns the backbone with **all params frozen** (`requires_grad=False`, `.eval()`),
  `num_classes=0` so it emits the **CLS embedding (B, 384)**, with `dynamic_img_size=True`.
- `LinearHead` is a single `nn.Linear(384 → n_classes)` — the only trainable teacher part.

### `cache.py` — precompute features + teacher logits (the performance heart of this part)
- `cache_features()` forwards every image through the frozen teacher **once per split** and saves
  `features_{split}.pt` + `labels_{split}.pt`. Because the teacher is frozen, its output never
  changes across epochs — caching turns a per-epoch transformer forward pass into a **one-time cost**.
- `cache_teacher_logits()` then trains the `LinearHead` on the cached **train** features (256-batch
  Adam, `epochs` from config), and dumps `logits_{split}.pt` for **all** splits.
- **Why store logits, not softmax probabilities?** Because the Hinton KD loss (Part 2) applies a
  **temperature T** at training time — `softmax(logits / T)`. Storing raw logits keeps T tunable
  later; storing pre-softened probs would bake T in permanently.
- Idempotent: every step checks `has_features` / `has_logits` and skips if already cached, so reruns
  are cheap and crash-resumable.

### Scripts
- `00_install_check.py` — environment/dependency sanity check.
- `01_cache_teacher.py` — runs feature + logit caching for the configured subsets.
- `04_train_teacher_probe.py` — trains/persists the linear probe (the `teacher_probe` model that
  appears as a third column in the results — DINOv2 features + linear head, no ResNet at all).

---

## 3. Real results this part produces

The **teacher_probe** row in the results *is* this part's output — it measures how good the frozen
DINOv2 features are on their own. This is the single most important diagnostic in the whole project,
because **distillation only helps when the teacher_probe beats the from-scratch baseline.**

| Dataset | Baseline Acc | **Teacher-probe Acc** | Teacher better? |
|---|---|---|---|
| dermamnist | 0.7097 | **0.7526** | ✅ yes → distill later wins |
| retinamnist | 0.5225 | **0.5300** | ✅ yes → distill later wins |
| pathmnist | 0.8877 | **0.9153** | ✅ yes → distill later wins |
| octmnist | 0.7680 | 0.5820 | ❌ no → distill later hurts |
| breastmnist | 0.8654 | 0.8013 | ❌ no → distill later hurts |
| tissuemnist | 0.7043 | 0.5952 | ❌ no → distill later hurts |

That correlation is the project's headline finding, and it starts *here*.

---

## 4. Likely interview questions (with strong answers)

**Q: Why freeze the teacher instead of fine-tuning it?**
A: DINOv2 is a self-supervised foundation model with strong general visual features. Freezing keeps
those features intact, makes the teacher a fixed reference, and — crucially — lets us **cache** its
outputs once. Fine-tuning would force re-computing features every epoch and risk overfitting a 22M-param
transformer on tiny 28×28 medical datasets.

**Q: Why cache features to disk? What does it buy you?**
A: The teacher is frozen, so its output for a given image is constant. Caching converts a per-epoch
ViT forward pass (the most expensive op) into a one-time precompute. Student training then reads a
`.pt` tensor instead of running the transformer — this is what made 10-epoch training feasible on the
cluster and cut distillation from ~70 s/batch to normal speed.

**Q: Why store logits and not probabilities?**
A: Hinton distillation softens targets with a temperature T: `softmax(logits/T)`. Storing raw logits
keeps T a free hyperparameter at student-training time; storing softmaxed probs would freeze T.

**Q: MedMNIST is 28×28 — why upsample to 224?**
A: DINOv2 uses patch-14 tokens and expects image sizes that are multiples of 14; 224 is the standard.
We bilinearly upsample and expand grayscale to 3 channels so one backbone handles all 12 datasets.

**Q: You hit "Killed" errors on the cluster — what happened?**
A: DataLoader workers shared tensors through `/dev/shm`, which the Kubeflow container capped at ~64 MB.
Mid-epoch it overflowed and the kernel killed a worker. Switching PyTorch's sharing strategy to
`file_system` (disk-backed) fixed it without reducing workers.

**Q: Why timm instead of the official DINOv2?**
A: The official repo needs Python 3.10+ syntax; the cluster runs 3.8. `timm` ships the same pretrained
weights and imports cleanly on 3.8.

---

## 5. How to run this part
```bash
python code/00_install_check.py          # verify env
python code/01_cache_teacher.py          # cache DINOv2 features + logits for all subsets
python code/04_train_teacher_probe.py    # train + evaluate the linear probe
```
Config (image size, teacher name, feature dim, subsets) lives in `demo/config.yaml`.
