# Three-level logistic model of child anaemia, NFHS-5:
# children within clusters (villages/wards), within districts, within states.
#
# Reports how much of the variation in anaemia lies at each level (VPC, MOR),
# adjusted odds ratios, and out-of-fold predictions for unseen districts that
# src/models.py compares with the ML models.
#
# This model is unweighted and model-based; R/stats.R is design-based.
#
# Run from the project root (takes 20-40 minutes):
#   Rscript R/multilevel.R

suppressPackageStartupMessages({
  library(lme4)
  library(arrow)
})

tables <- "results/tables"
df <- as.data.frame(read_parquet("data/processed/analysis.parquet"))

df$age_band <- factor(as.character(df$age_band),
                      levels = c("48-59", "36-47", "24-35", "12-23", "6-11"))
df$wealth_q <- factor(df$wealth_q, levels = 5:1,
                      labels = c("richest", "richer", "middle", "poorer", "poorest"))
df$mother_edu <- factor(df$mother_edu, levels = 3:0,
                        labels = c("higher", "secondary", "primary", "none"))
df$birth_order_cat <- factor(pmin(df$birth_order, 4), levels = 1:4,
                             labels = c("1", "2", "3", "4+"))
df$ifa_pregnancy <- factor(df$ifa_pregnancy, levels = c("yes", "no", "not_asked"))
# IDs that are unique across India, so each cluster sits in one district.
df$state <- factor(df$state_code)
df$district <- factor(df$district_code)
df$cluster_id <- factor(paste(df$district_code, df$cluster, sep = "_"))

covariates <- c("age_band", "child_female", "birth_order_cat", "stunted",
                "diarrhoea_2wk", "fever_2wk", "mother_edu", "mother_anaemic",
                "wealth_q", "rural", "clean_fuel", "improved_sanitation",
                "ifa_pregnancy")
levels_re <- "(1 | state) + (1 | district) + (1 | cluster_id)"
ctrl <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)


fit <- function(rhs, data) {
  # nAGQ = 0 makes a model with ~29,000 clusters fast enough to fit on a laptop.
  glmer(as.formula(paste("anaemic ~", rhs, "+", levels_re)), data = data,
        family = binomial, nAGQ = 0, control = ctrl)
}


variance_table <- function(model, label) {
  # VPC: share of total latent variance at each level (pi^2/3 = child level).
  # MOR: median odds ratio between two children in different units of a level.
  v <- as.data.frame(VarCorr(model))
  v <- v[match(c("state", "district", "cluster_id"), v$grp), ]
  total <- sum(v$vcov) + pi^2 / 3
  data.frame(model = label, level = c("state", "district", "cluster"),
             variance = v$vcov, vpc = v$vcov / total,
             mor = exp(sqrt(2 * v$vcov) * qnorm(0.75)), row.names = NULL)
}


complete <- df[complete.cases(df[, covariates]), ]
cat(sprintf("Children used: %d (complete cases)\n", nrow(complete)))

# ---- 1. Null and full models ----------------------------------------------
cat("Fitting null model...\n")
null_model <- fit("1", complete)
cat("Fitting full model...\n")
full_model <- fit(paste(covariates, collapse = " + "), complete)

vp <- rbind(variance_table(null_model, "null"), variance_table(full_model, "full"))
vp$pcv <- NA
vp$pcv[vp$model == "full"] <- 1 - vp$variance[vp$model == "full"] /
  vp$variance[vp$model == "null"]
write.csv(vp, file.path(tables, "variance_partition.csv"), row.names = FALSE)

coefs <- summary(full_model)$coefficients[-1, ]
ors <- data.frame(term = rownames(coefs),
                  odds_ratio = exp(coefs[, "Estimate"]),
                  ci_low = exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"]),
                  ci_high = exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"]),
                  row.names = NULL)
write.csv(ors, file.path(tables, "multilevel_odds_ratios.csv"), row.names = FALSE)

cat("\nVariance partition (pcv = share of null variance explained by covariates):\n")
print(format(vp, digits = 3), row.names = FALSE)
cat("\nMultilevel adjusted odds ratios:\n")
print(format(ors, digits = 3), row.names = FALSE)

# ---- 2. Out-of-fold predictions for unseen districts ----------------------
# Refit without each fold's districts, then predict those children using the
# fixed effects and their state's effect only (their district is unseen).
oof <- data.frame(row_id = complete$row_id, p_glmer = NA_real_)
for (k in sort(unique(complete$fold))) {
  cat(sprintf("Out-of-fold model, fold %d...\n", k))
  test <- complete$fold == k
  m <- fit(paste(covariates, collapse = " + "), complete[!test, ])
  oof$p_glmer[test] <- predict(m, newdata = complete[test, ], type = "response",
                               re.form = ~(1 | state), allow.new.levels = TRUE)
}
write.csv(oof, "data/processed/oof_glmer.csv", row.names = FALSE)
cat("Saved data/processed/oof_glmer.csv\n")