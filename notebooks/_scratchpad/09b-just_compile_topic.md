Text Classification with Fine-Tuned BERT
================


we’ll explore **fine-tuning pretrained Transformer models** for text classification using the **IMDB Movie Review** dataset. You’ll begin by training and evaluating a DistilBERT baseline, then conduct experiments to investigate how training configurations, dataset size, and model architecture affect predictive performance and computational cost.

You’ll complete three problems:

* **Problem 1:** Compare three context-length training configurations and their validation performance and training times. You’ll also compare your best configuration with the initial DistilBERT baseline.
* **Problem 2:** Measure how **training data size** affects both model performance and total training time.
* **Problem 3:** Compare **DistilBERT and BERT-base** to analyze the trade-offs between model size, accuracy, and training time.

In each problem, you’ll report experimental results, interpret the differences, and justify a choice based on validation performance and computational cost.


# Setup

```python
# ---- Backend BEFORE importing keras ----
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import time
import random
import numpy as np
import keras
import keras_hub as kh
import evaluate
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from sklearn.model_selection import train_test_split

from keras import mixed_precision                    # generally faster
mixed_precision.set_global_policy("mixed_float16")


# ------------ Global Configuration ------------

SEED = 42

# Model and training settings
MAX_LEN = 128
EPOCHS = 2
BATCH = 32
EVAL_BATCH = 64


# Development workflow
# Use "smoke" for debugging, "dev" for development,
# "baseline" for 25%, or "full" for 100%.

DATA_MODE = "baseline"

DATA_FRACTIONS = {
    "smoke":    0.01,   # 1% of training pool
    "dev":      0.10,   # 10% of training pool
    "baseline": 0.25,   # 25% of training pool
    "full":     1.00    # Entire training pool
}

assert DATA_MODE in DATA_FRACTIONS
TRAIN_FRAC = DATA_FRACTIONS[DATA_MODE]


keras.utils.set_random_seed(SEED)

# ------------ Load and Partition IMDB ------------

imdb = load_dataset("stanfordnlp/imdb")

# Preserve the original held-out test set
test_ds = imdb["test"]

# Create a fixed, stratified validation set
indices = np.arange(len(imdb["train"]))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=np.array(imdb["train"]["label"])
)

train_pool = imdb["train"].select(train_idx.tolist())
val_ds = imdb["train"].select(val_idx.tolist())

# ------------ Reproducible Nested Training Subsets ------------

# Shuffle each class independently, then interleave classes.
# This gives balanced, nested training subsets.
rng = np.random.default_rng(SEED)

labels = np.array(train_pool["label"])
neg_idx = np.flatnonzero(labels == 0)
pos_idx = np.flatnonzero(labels == 1)

rng.shuffle(neg_idx)
rng.shuffle(pos_idx)

train_order = np.column_stack((neg_idx, pos_idx)).ravel()

def get_training_subset(frac):
    """Return a balanced, nested fraction of the training pool."""
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1].")

    n_pairs = max(1, round(len(train_order) * frac / 2))
    selected = train_order[:2 * n_pairs]

    return train_pool.select(selected.tolist())

# Apply the development throttle
train_ds = get_training_subset(TRAIN_FRAC)

# ------------ NumPy Arrays for Keras ------------

X_tr = np.array(train_ds["text"], dtype=object)
y_tr = np.array(train_ds["label"], dtype="int32")

X_va = np.array(val_ds["text"], dtype=object)
y_va = np.array(val_ds["label"], dtype="int32")

X_te = np.array(test_ds["text"], dtype=object)
y_te = np.array(test_ds["label"], dtype="int32")

# ---- DistilBERT baseline: develop using the selected data throttle ----

print(f"Data mode: {DATA_MODE}")
print(f"Training examples: {len(X_tr):,}")
print(f"Validation examples: {len(X_va):,}")
print(f"Maximum sequence length: {MAX_LEN}")
print(f"Batch size: {BATCH}")

# Build preprocessor and classifier
keras.utils.set_random_seed(SEED)

preproc = kh.models.DistilBertTextClassifierPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=MAX_LEN
)

model = kh.models.DistilBertTextClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    preprocessor=preproc
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

# ---- Train with early stopping ----

cb = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

start = time.perf_counter()

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=cb,
    verbose=1
)

elapsed = time.perf_counter() - start

# ---- Evaluate the restored model on validation data ----

val_loss, val_acc = model.evaluate(
    X_va, y_va,
    batch_size=EVAL_BATCH,
    verbose=0
)

logits = model.predict(
    X_va,
    batch_size=EVAL_BATCH,
    verbose=0
)

y_pred = np.argmax(logits, axis=-1)

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

acc = acc_metric.compute(
    predictions=y_pred,
    references=y_va
)["accuracy"]

f1 = f1_metric.compute(
    predictions=y_pred,
    references=y_va
)["f1"]

# ---- Confusion matrix ----

def confusion_matrix_np(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ---- Summary ----

best_epoch = np.argmin(history.history["val_loss"]) + 1


```


# P1
## Mini Sweep: Context Length and Training Cost (3 Runs)

In this problem, we'll investigate how **context length** (`MAX_LEN`) affects the performance and computational cost of fine-tuning DistilBERT.

A longer context allows the model to use more of each text, potentially improving its predictions. However, longer sequences require more computation and memory, particularly because of the attention mechanism.

We'll compare three configurations, using different batch sizes and learning rates appropriate for each context length.

## Setup (keep these fixed)

* `SUBSET_FRAC = 0.25` — use 25% of the training dataset.
* `EPOCHS = 2` — maximum number of training epochs.
* **EarlyStopping** with `restore_best_weights=True`.
* Same random seed for all runs.
* Same training and validation examples for all runs (don't reshuffle or resplit between runs).

## Run these 3 configurations

| MAX_LEN | Learning rate | Batch size |
|---|---|---|
| 128 | `2e-5` | 32 |
| 256 | `1e-5` | 16 |
| 512 | `7.5e-6` | 8 |

These configurations represent progressively longer contexts, with smaller batch sizes to accommodate the increased memory requirements.

**Important:** Because learning rate and batch size also change, this experiment compares three practical training configurations. It does not isolate the effect of context length alone.

For each run, record:

* Best validation loss and the epoch at which it occurred.
* Validation accuracy at that epoch.
* Total training time.

Identify the configuration with the **lowest validation loss**. Problems 2 and 3 use their own fixed configurations to isolate the effects being studied.

### Q1a: Report the lowest validation loss

### Q1b: 
* How did validation accuracy change across the three context lengths (128 → 256 → 512)? Were the improvements equally large?
* How did training time change as context length increased?
* Compare the best Problem 1 configuration with the initial DistilBERT baseline. How much did validation accuracy improve, and what additional training time was required?
* Which configuration would you choose, considering both validation performance and computational cost? Explain your reasoning.

**Note:** Because learning rate and batch size also changed, you cannot attribute the differences in performance solely to context length.

> Answer in long text.


# P2

## How Much Data Is Enough? (3 Runs)

In this problem, you'll investigate how the amount of training data affects the performance and computational cost of fine-tuning DistilBERT.

More training data generally improves predictive performance, but also increases training time. We want to determine whether the additional improvement is worth the computational cost.

## Setup

* Use a fixed DistilBERT configuration: `MAX_LEN = 256`, `LR = 1e-5`, and `BATCH = 16`, independently of the selected configuration in Problem 1.
* Keep the validation dataset fixed across all runs.
* Use `EPOCHS = 2` and EarlyStopping with `restore_best_weights=True`.
* Use the same random seed for all runs.

## Run these 3 configurations

| SUBSET_FRAC | Training data |
|---|---|
| 0.25 | 25% of the training set |
| 0.50 | 50% of the training set |
| 1.00 | 100% of the training set |

For each run, record:

* Best validation loss and the epoch at which it occurred.
* Validation accuracy at that epoch.
* Total training time.

Select the configuration with the lowest validation loss.
### Q2a: Report the lowest validation loss
### Q2b: 
* How did validation accuracy change as the training dataset increased from 25% to 50% to 100%? Were the improvements equally large?
* How did training time change as the dataset increased?
* Which dataset size would you choose, considering both predictive performance and computational cost? Explain your reasoning using your results.

> Answer in long text.

# P3

## Model Swap: Speed vs. Accuracy (2 Runs)

In this problem, we'll compare two pretrained Transformer models with different capacities:

* **DistilBERT** — a smaller, computationally efficient model.
* **BERT-base** — a larger model with greater capacity, but higher computational cost.

Both models can be fine-tuned for the same text classification task. Our goal is to determine whether the larger model's improvement in predictive performance justifies its additional training time.

## Setup

Use the following settings for both models:

* `MAX_LEN = 256`
* `LR = 1e-5`
* `BATCH = 16`
* `SUBSET_FRAC = 0.50`
* `EPOCHS = 2`
* EarlyStopping with `restore_best_weights=True`
* Same random seed
* Same training and validation examples

Only the model architecture changes.

## Run these 2 configurations

| Model | Pretrained preset |
|---|---|
| DistilBERT | `distil_bert_base_en_uncased` |
| BERT-base | `bert_base_en_uncased` |

## How to switch models

**DistilBERT:**

```python
preproc = kh.models.DistilBertTextClassifierPreprocessor.from_preset(
    "distil_bert_base_en_uncased", sequence_length=MAX_LEN
)
model = kh.models.DistilBertTextClassifier.from_preset(
    "distil_bert_base_en_uncased", num_classes=2, preprocessor=preproc
)
```

**BERT-base:**

```python
preproc = kh.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en_uncased", sequence_length=MAX_LEN
)
model = kh.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2, preprocessor=preproc
)
```

## What to do

Train and evaluate both models using the same experimental settings.

For each model, record:

* Best validation loss and the epoch at which it occurred.
* Validation accuracy at that epoch.
* Total training time.

Compare the results, then answer the graded questions.

**Note:** This experiment compares models under the same training budget, not necessarily at their individually optimal hyperparameters.

### Q3a: Report the lowest validation loss for each model
### Q3b: 
* Which model achieved the higher validation accuracy? How large was the difference?
* How did their total training times compare?
* Given limited computational resources, which model would you choose? Explain whether the improvement in predictive performance justifies the additional computational cost.

> Answer