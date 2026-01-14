# Predict Gene Expression – Submission Description

## 1. Overview

This submission includes trained models for predicting gene expression values for three genes. The `final.py` script loads the corresponding model for each gene, applies it to test data, and outputs predictions in TSV format.

The models were trained using feature scaling and optional feature selection. Any transformations applied during training (e.g., log2 scaling) are automatically reversed in the output.

---

## 2. Folder Structure

```
predict_expression/
│
├── models/                    # Saved models per gene
│   ├── model_gene1.joblib
│   ├── model_gene2.joblib
│   └── model_gene3.joblib
│
├── data/
│   └── raw/                   # Test data files
│       ├── gene_1_test.txt
│       ├── gene_2_test.txt
│       └── gene_3_test.txt
│
├── predictions/               # Output prediction files (created by final.py)
│   ├── pred_gene_1.tsv
│   ├── pred_gene_2.tsv
│   └── pred_gene_3.tsv
│
├── src/
│   └── final.py               # Final prediction script
│
└── requirements.txt           # Required Python packages
```

---

## 3. Installation

Create a Python environment and install the required packages:

```
# create environment
conda create -n predict_expression python=3.10
conda activate epi

# install required packages
pip install -r requirements.txt
```

**Minimum packages required to run `final.py`:**

* `numpy`
* `pandas`
* `scikit-learn`
* `joblib`


---

## 4. Usage

The `final.py` script predicts gene expression for a selected gene using a provided test file.

**Command line arguments:**

| Argument   | Description                                                                          |
| ---------- | ------------------------------------------------------------------------------------ |
| `--input`  | Path to test data TSV file (must include ID/Sample column, all other columns are features). |
| `--gene`   | Gene number (1, 2, or 3).                                                            |
| `--output` | Path to the output TSV file where predictions will be saved.                         |

**Command lines:**

```
python src/final.py --input data/raw/gene_1_test.txt --gene 1 --output predictions/pred_gene_1.tsv
python src/final.py --input data/raw/gene_2_test.txt --gene 2 --output predictions/pred_gene_2.tsv
python src/final.py --input data/raw/gene_3_test.txt --gene 3 --output predictions/pred_gene_3.tsv
```

**Output:**

```
ID    Expression
sample_1    2.34
sample_2    1.87
...
```

* `ID` column: sample identifiers (copied from test data).
* `Expression` column: predicted gene expression values.



## 5. Notes

1. Make sure the `models/` directory contains the trained `.joblib` files for all genes.
2. The script automatically reverses any scaling applied during training.
3. Predictions are saved per gene in the `predictions/` folder.
