# Characterizing Pattern Matching and Its Limits on Compositional Task Structures

This repository contains code for **[“Characterizing Pattern Matching and Its Limits on Compositional Task Structures”](https://www.arxiv.org/abs/2505.20278?context=cs.AI)** The codes are built upon the [GrokkedTransformer repository](https://github.com/OSU-NLP-Group/GrokkedTransformer).

## Abstract

Despite impressive capabilities, LLMs' successes often rely on pattern-matching behaviors, yet these are also linked to OOD generalization failures in compositional tasks. However, behavioral studies commonly employ task setups that allow multiple generalization sources (e.g., algebraic invariances, structural repetition), obscuring a precise and testable account of how well LLMs perform generalization through pattern matching and their limitations. To address this ambiguity, we first formalize pattern matching as functional equivalence, i.e., identifying pairs of subsequences of inputs that consistently lead to identical results when the rest of the input is held constant. Then, we systematically study how decoder-only Transformer and Mamba behave in controlled tasks with compositional structures that isolate this mechanism. Our formalism yields predictive and quantitative insights: (1) Instance-wise success of pattern matching is well predicted by the number of contexts witnessing the relevant functional equivalence. (2) We prove a tight sample complexity bound of learning a two-hop structure by identifying the exponent of the data scaling law for perfect in-domain generalization. Our empirical results align with the theoretical prediction, under 20x parameter scaling and across architectures. (3) Path ambiguity is a structural barrier: when a variable influences the output via multiple paths, models fail to form unified intermediate state representations, impairing accuracy and interpretability. (4) Chain-of-Thought reduces data requirements yet does not resolve path ambiguity. Hence, we provide a predictive, falsifiable boundary for pattern matching and a foundational diagnostic for disentangling mixed generalization mechanisms.

## File Structure
```
coverage-principle/
├── dataset\_generation/: scripts for training/evaluation data generation
├── data/: cached training/evaluation data
├── main.py: main script for model training
├── determine\_coverage.py: coverage determination algorithm
└── circuit\_analysis/: cosine similarity and causal tracing analysis

````

## Environmental Setup
```bash
conda create -n coverage-principle python=3.10
conda activate coverage-principle

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 transformers==4.37.2 --extra-index-url https://download.pytorch.org/whl/cu116

cd simpletransformers
pip install -e .
cd ..
````

## Data Preparation

### Generate Synthetic Dataset

Use the dataset-generation scripts to create synthetic compositional tasks. For example, to generate a 2-hop compositional dataset:

```bash
cd dataset_generation
python twohop.py --num_tokens 50 --max_train_data_num 10000 --default_seen_ratio 0.7 --test_size_for_type 2000 --seed 42
```

**Key arguments**

| Argument               | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `--num_tokens`         | Size of the token vocabulary                          |
| `--max_train_data_num` | Maximum number of training examples                   |
| `--default_seen_ratio` | Fraction of each function’s domain marked as **seen** |
| `--test_size_for_type` | Number of test samples for each coverage type         |
| `--cot`                | Enable Chain-of-Thought supervision                   |

This creates a dataset in `data/twohop.50.10000.diff-f12.inf/` containing:

* `train.json` – training data
* `test.json` – test data with coverage-type annotations
* `atomic_facts_f1.json`, `atomic_facts_f2.json` – primitive-function mappings
* `vocab.json` – vocabulary tokens

## Coverage Determination

To analyze which test examples fall within the coverage of your training data:

```bash
python determine_coverage.py --data_dir data/twohop.50.10000.diff-f12.inf/ --min_evidence 1 --k_sweep
```

**Key arguments**

| Argument         | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `--data_dir`     | Path to dataset directory                                 |
| `--min_evidence` | Minimum evidence threshold *k* for functional equivalence |
| `--k_sweep`      | Run analysis for multiple *k* values                      |
| `--visualise`    | Generate graph visualization of coverage                  |
| `--ground_truth` | Use ground-truth functional equivalence (f1 only)         |

Outputs include:

* `k_sweep_results/` – coverage analysis results for different *k* values
* `test_annotated.json` – test data with coverage annotations
* Coverage visualization (if `--visualise` is used)

## Model Training

Train a GPT-2 model on the generated dataset:

```bash
bash script/train.sh twohop.50.10000.diff-f12.inf 0.1 8 12 0 42
```

**Script arguments**

1. Dataset name (e.g., `twohop.50.10000.diff-f12.inf`)
2. Weight decay (e.g., `0.1`)
3. Number of layers (e.g., `8`)
4. Number of attention heads (e.g., `12`)
5. GPU ID (e.g., `0`)
6. Random seed (e.g., `42`)

**Training configuration**

* Architecture: GPT-2 with specified layers/heads
* Learning rate: `8e-4`
* Batch size: `4096`
* Max steps: `62500`
* Distributed training on 4 GPUs

Trained models are saved in `CKPT_DIR/trained_checkpoints/`.

## Analysis

### Cosine Similarity Analysis

Analyze how models form clustered representations of functionally equivalent components:

```bash
cd circuit_analysis/hierarchical/2-hop
python collapse_analysis_2-hop.py \
    --ckpt CKPT_DIR/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42/final_checkpoint \
    --layer_pos_pairs "[(3,1)]" \
    --save_dir ./results/cosine_analysis \
    --atomic_idx 1 \
    --mode residual
```

### Causal Tracing Analysis

Identify which representations are causally important:

```bash
cd circuit_analysis/hierarchical/2-hop
python causal_tracing_2-hop.py \
    --model_dir CKPT_DIR/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list final_checkpoint \
    --data_dir ./data_fixed \
    --batch_size 1024 \
    --metric_type rank
```

## License

This project is licensed under the MIT License.
