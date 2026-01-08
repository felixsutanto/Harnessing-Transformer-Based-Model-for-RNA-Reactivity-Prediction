# Harnessing Transformer-Based Model for RNA Reactivity Prediction

This repository contains the training code for a thesis project on **RNA reactivity prediction** using **Transformer-based sequence models**. The work is **heavily inspired by and adapted from the 10th place solution** by **Shlomoron** for the **Stanford Ribonanza RNA Folding Competition** on Kaggle.

> Note: At the moment, this repository only includes the **training script** (`model-training.ipynb`). Data preparation, inference, and ensembling are assumed to be done in an external environment (e.g., Kaggle notebooks).

***

## 1. Project Overview

Predicting **nucleotide-wise reactivity profiles** from RNA sequences is an important subproblem in RNA structure prediction and understanding RNA biology. The **Stanford Ribonanza RNA Folding Competition** provided a large-scale dataset of RNA sequences with measured reactivity under different chemical probing conditions (e.g., 2A3, DMS). The challenge was to learn a model that can **map raw sequences to per-position reactivity values**.

This thesis project:

- Reproduces and adapts a **sequence-only Transformer-based model** originally developed by Shlomoron (10th place solution).
- Focuses on **clean data ingestion from TFRecords**, **careful filtering**, and **Transformer encoder-based sequence modeling**.
- Adds a **K-Fold cross-validation training loop** to obtain multiple models that can be used for **ensembling** at inference time.

***

## 2. Origin and Acknowledgement

This implementation is directly inspired by:

**Shlomoron – 10th Place Solution (Stanford Ribonanza RNA Folding Competition)**
GitHub: https://github.com/shlomoron/Stanford-Ribonanza-RNA-Folding-10th-place-solution

Key aspects borrowed or adapted from Shlomoron’s solution include:

- The idea of a **sequence-only model** (no explicit 2D structure features).
- The use of deep **Transformer-like sequence encoders**.
- A **custom loss design** that:
    - Masks out unreliable targets (NaN).
    - Uses multiple **error-based metrics** (MAE-like, RMSE, percentage error, MAPE).
- A training strategy based on:
    - **Careful filtering** using read counts and signal-to-noise thresholds.
    - **Multiple models / folds** for potential ensembling.

While the implementation in this repository deviates in some details (e.g., focusing on pure Transformer encoder blocks and a particular K-Fold setup), the **core conceptual approach and inspiration belong to Shlomoron**. This should be clearly acknowledged in any academic or practical use of this code.

***

## 3. Methodology and Solution Approach

### 3.1 Data Pipeline (TFRecord → TensorFlow Dataset)

The training pipeline assumes that the competition data has been converted into **TFRecord files**, located at:

```python
tffiles_path = '/kaggle/input/srrf-tfrecords-ds/tfds'
tffiles = [f'{tffiles_path}/{x}.tfrecord' for x in range(164)]
```

Key steps:

1. **Decoding TFRecords**
Each TFRecord example contains (among others):
    - `seq`: encoded RNA sequence.
    - `reads_2A3`, `reads_DMS`: read counts.
    - `signal_to_noise_2A3`, `signal_to_noise_DMS`: S/N metrics.
    - `SN_filter_2A3`, `SN_filter_DMS`: quality flags.
    - `reactivity_2A3`, `reactivity_DMS`: target reactivity values.
    - `reactivity_error_*`: reactivity uncertainties (not all are used directly in this script).
2. **Filtering Examples**

Two filtering functions are defined:
    - `filter_function_1`: keeps examples where both `SN_filter_2A3 == 1` and `SN_filter_DMS == 1`.
    - `filter_function_2`: keeps examples where:
        - `(reads_2A3 > 100 and signal_to_noise_2A3 > 1)` **or**
        - `(reads_DMS > 100 and signal_to_noise_DMS > 1)`.

This removes examples with **low coverage or low signal-to-noise**, improving label quality.
3. **NaN Handling and Target Construction**
    - For positions that do not pass quality thresholds, reactivity values are converted to `NaN`.
    - Targets for both conditions (2A3 and DMS) are concatenated and clipped to `[0, 1]`:

```python
target = tf.clip_by_value(
    tf.concat(
        [reactivity_2A3[..., tf.newaxis],
         reactivity_DMS[..., tf.newaxis]],
        axis=1
    ),
    0, 1
)
```

    - The final dataset yields `(seq, target)` pairs with padding up to `X_max_len = 206`.
4. **Batched, Prefetched Dataset**

`get_tfrec_dataset(...)` constructs an efficient `tf.data.Dataset`:
    - Decoding, filtering, preprocessing.
    - `padded_batch` with:
        - Feature padding value `PAD_x = 0.0`.
        - Target padding value `PAD_y = np.nan`.
    - Optional shuffling, repeating, and `prefetch(tf.data.AUTOTUNE)`.

***

### 3.2 Model Architecture

The core model is a **pure Transformer encoder** applied to the tokenized RNA sequence.

1. **Input and Embedding**
```python
inputs = tf.keras.Input([max_len])          # [batch, 206]
x = tf.keras.layers.Embedding(
    input_dim=num_vocab,                   # A, C, G, U, N
    output_dim=hidden_dim,                 # e.g., 384
    mask_zero=True
)(inputs)
```

2. **Positional Encoding Layer**

A custom `positional_encoding_layer`:

- Implements a sinusoidal positional encoding similar to the original Transformer paper.
- Scales token embeddings by `sqrt(hidden_dim)` before adding positions.
- Supports variable-length sequences (up to a configured maximum length).

3. **Stacked Transformer Encoder Blocks**

A custom `TransformerEncoder` layer:

- **Multi-head self-attention** (`MultiHeadAttention` with 8 heads).
- Feed-forward sub-layer (`Dense` → ReLU → `Dense`).
- Two **residual connections** with **LayerNormalization**.
- Dropout applied to attention and feed-forward outputs.
- Respects the mask to avoid attending to padding tokens.

The model stacks multiple such encoder layers (e.g. 12 blocks):

```python
for _ in range(12):
    x = TransformerEncoder(
        embed_dim=hidden_dim,
        num_heads=8,
        feed_forward_dim=hidden_dim * 4
    )(x)
```

4. **Output Layer**

- Final `Dense(2)` layer to predict two reactivity channels (2A3 and DMS) per nucleotide position.

***

### 3.3 Loss Functions and Metrics

The training script defines several custom losses/metrics that:

- **Mask out positions where labels are NaN** (unreliable or padded).
- Compute errors only over valid positions to avoid bias.

Defined functions include:

- `loss_fn`:
Mean absolute error over non-NaN positions.
- `loss_fn_metric_rmse`:
Root Mean Square Error over non-NaN positions.
- `loss_fn_metric_pe`:
Percentage error `(label - target) / target * 100` with clipping.
- `loss_fn_metric_mape`:
Mean Absolute Percentage Error with clipping.
- `accuracy`:
Fraction of positions where rounded predictions equal rounded labels (over valid positions).

The model is compiled with **AdamW**:

```python
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.0005,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.98,
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        accuracy,
        loss_fn_metric_rmse,
        loss_fn_metric_pe,
        loss_fn_metric_mape,
    ],
)
```


***

### 3.4 Training Strategy

1. **Hardware Strategy**

- Attempts to connect to a **TPU** via `TPUStrategy`.
- Falls back to the default strategy (CPU/GPU) if TPU is unavailable.

2. **Learning Rate Schedule**

A cosine-like decay schedule with warmup:

- Total epochs: `N_EPOCHS = 80`
- Warmup epochs: `N_WARMUP_EPOCHS = 10`
- `LR_HIGH = 5e-4`, `LR_LOW = 5e-5`

3. **K-Fold Cross-Validation**

- Uses `KFold` from `sklearn.model_selection` with `n_splits = 5`.
- For each fold:
    - Constructs train/validation TFRecord subsets.
    - Trains a separate Transformer model on that fold.
    - Saves fold-specific weights.

4. **Model Checkpointing**

Two custom callbacks are used:

- `SaveModelPerFoldCallback(fold)`
Saves final weights for each fold under:

```text
/kaggle/working/weights/fold_{fold}/model.h5
```

- `SaveModelPerEpochCallback(fold)`
Optionally saves model weights at selected epochs (e.g. epoch 3, every 25 epochs).

This setup is suitable for **ensembling**, where predictions from multiple fold models can be averaged to improve robustness and accuracy.

***

## 4. Repository Structure

Current minimal structure:

```text
.
├── model-training.ipynb   # Main training notebook (this thesis implementation)
└── README.md              # Project documentation (this file)
```

Potential future extensions:

- `data/` – Scripts for data download and TFRecord creation.
- `inference/` – Inference and ensembling scripts for predictions / submissions.
- `configs/` – Configuration files for hyperparameters and paths.
- `results/` – Saved metrics, plots, and model checkpoints (normally git-ignored).

***

## 5. Getting Started

### 5.1 Requirements

The notebook assumes an environment similar to **Kaggle** with:

- Python 3.x
- TensorFlow 2.x (with TPU support if using TPUs)
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install locally (example):

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```


### 5.2 Data

You need access to the **Stanford Ribonanza RNA Folding** competition dataset, preprocessed into TFRecords with the fields used in the notebook.

Expected location in the notebook:

```python
tffiles_path = '/kaggle/input/srrf-tfrecords-ds/tfds'
tffiles = [f'{tffiles_path}/{x}.tfrecord' for x in range(164)]
```

If running outside Kaggle:

- Adjust `tffiles_path` to your local or cloud path.
- Ensure the TFRecord schema matches the `decode_tfrec` implementation.


### 5.3 Running Training

1. Open `model-training.ipynb` in Jupyter / Kaggle.
2. Adjust paths and configuration parameters if needed:
    - `tffiles_path`
    - `N_EPOCHS`, `batch_size`, `hidden_dim`, etc.
3. Run all cells to:
    - Build the datasets.
    - Construct the Transformer model.
    - Train using K-Fold cross-validation.
    - Save fold-specific weights under `/kaggle/working/weights`.

***

## 6. Limitations and Future Work

- **Scope of this repo**: Currently only includes the **training script**. Data preprocessing and inference/ensembling scripts are not yet included.
- **Hyperparameter exploration**: The model uses a fixed architecture (12 encoder layers, 384-dim embeddings, 8 heads). Future work could explore:
    - Different model depths and widths.
    - Alternative optimizers (e.g. Ranger as in the original solution).
    - Additional regularization (dropout, stochastic depth, etc.).
- **Uncertainty-aware loss**: This script primarily focuses on masking unreliable labels and using error-based metrics. Shlomoron’s original solution explores more advanced **“potential”-based losses** inspired by physical intuition (shell theorem). Integrating and analyzing those in more depth is a potential extension.
- **Structural information**: The current model is **sequence-only**. Incorporating secondary/tertiary structure features might further improve performance but is beyond the current scope.

***

## 7. Academic Use and Citation

If you use this repository for academic purposes (e.g., a thesis or publication), please:

1. **Cite and acknowledge Shlomoron’s competition solution**, as this implementation is derived from and inspired by his work.
2. Clearly state that this repository is an **adaptation and extension** of a **Kaggle competition solution**, not an entirely novel method from scratch.

Suggested acknowledgement:

> "This work is adapted from and strongly inspired by the 10th place solution by Shlomoron for the Stanford Ribonanza RNA Folding Competition, particularly in the use of Transformer-based sequence modeling, uncertainty-aware loss design, and multi-model training strategies."

