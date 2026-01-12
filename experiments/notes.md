# Example_experiment

## Goal
- add goal/why you did this experiment if applicable, for example if another experiment gave rise to  this one

## Config highlights
- add most important steps here such as type of preprocessing, which model, etc.

## Results
- add RMSE and pearson correlation coefficient

## Conclusions
- add any conclusions or observations that you gathered from this experiment

# Baseline elastic net
Folders: 
- baseline_elasticnet_gene_3_809afa3d
- baseline_elasticnet_gene_2_e0bf0109
- baseline_elasticnet_gene_1_fe7930e4

## Goal
Establish the minimum metrics we are trying to outperform.

## Config highlights
reprocessing:
  feature_selection:
    apply: false
  save_preprocessed: true
  x_scaling:
  - params: {}
    scaler: standard
  y_scaling:
  - params: {}
    scaler: log2
  - params: {}
    scaler: standard
training:
  cv: 5
  estimator: elastic_net
  param_grid:
    alpha:
    - 0.1
    - 0.2
    - 0.3
    - 0.5
    - 1.0
    - 5
    - 10
    l1_ratio:
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9

## Results
See .csv file

## Conclusions
- Observation: gene_2 has very low variance in the gene expression. Evaluation must be carried out on many train-test-splits to gain an accurate estimate of the metrics


