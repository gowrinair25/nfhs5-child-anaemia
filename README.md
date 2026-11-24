# Child anaemia across Indian districts (NFHS-5)

**Question:** Which child, maternal and household factors are associated with anaemia in Indian children aged 6–59 months, how much of the variation lies between states, districts and villages, and can machine learning predict anaemia in districts and states it has never seen?

![District maps](results/figures/map_observed_predicted_residual.png)


## Repository

```
src/prepare.py            cohort, outcome, covariates, district folds
R/stats.R                 survey-weighted prevalence and odds ratios
R/multilevel.R            three-level model, VPC/MOR, out-of-fold predictions
src/models.py             logistic regression vs LightGBM vs multilevel
notebooks/figures.ipynb   figures, SHAP, maps, Moran's I, LISA
results/tables/           all aggregated results
results/figures/          all figures
```

