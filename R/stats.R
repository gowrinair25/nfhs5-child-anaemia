# Survey-weighted (design-based) analysis of child anaemia, NFHS-5.
#
# Uses the NFHS-5 sample weights, clusters (PSUs) and strata so that
# prevalence and odds ratios are valid for all Indian children aged 6-59 months.
#
# Run from the project root:
#   Rscript R/stats.R

suppressPackageStartupMessages({
  library(survey)
  library(arrow)
})
options(survey.lonely.psu = "adjust")

tables <- "results/tables"
dir.create(tables, showWarnings = FALSE, recursive = TRUE)

df <- as.data.frame(read_parquet("data/processed/analysis.parquet"))

# Categorical versions with an easy-to-read reference level (listed first).
df$age_band <- factor(as.character(df$age_band),
                      levels = c("48-59", "36-47", "24-35", "12-23", "6-11"))
df$wealth_q <- factor(df$wealth_q, levels = 5:1,
                      labels = c("richest", "richer", "middle", "poorer", "poorest"))
df$mother_edu <- factor(df$mother_edu, levels = 3:0,
                        labels = c("higher", "secondary", "primary", "none"))
df$birth_order_cat <- factor(pmin(df$birth_order, 4), levels = 1:4,
                             labels = c("1", "2", "3", "4+"))
df$ifa_pregnancy <- factor(df$ifa_pregnancy, levels = c("yes", "no", "not_asked"))

des <- svydesign(ids = ~cluster, strata = ~strata, weights = ~weight,
                 data = df, nest = TRUE)

covariates <- c("age_band", "child_female", "birth_order_cat", "stunted",
                "diarrhoea_2wk", "fever_2wk", "mother_edu", "mother_anaemic",
                "wealth_q", "rural", "clean_fuel", "improved_sanitation",
                "ifa_pregnancy")


prevalence_by <- function(var, design) {
  # Weighted anaemia prevalence with 95% CI for each level of `var`.
  est <- svyby(~anaemic, as.formula(paste0("~", var)), design, svymean,
               na.rm = TRUE)
  n <- table(design$variables[[var]])
  data.frame(variable = var,
             level = as.character(est[[var]]),
             n_children = as.integer(n[as.character(est[[var]])]),
             prevalence = est$anaemic,
             ci_low = est$anaemic - 1.96 * SE(est),
             ci_high = est$anaemic + 1.96 * SE(est),
             row.names = NULL)
}


odds_ratios <- function(model, label) {
  # Odds ratios with 95% CI from a survey-weighted logistic model.
  ci <- exp(cbind(coef(model), confint(model)))[-1, , drop = FALSE]
  data.frame(model = label, term = rownames(ci), odds_ratio = ci[, 1],
             ci_low = ci[, 2], ci_high = ci[, 3], row.names = NULL)
}


# ---- 1. Prevalence ---------------------------------------------------------
national <- svymean(~anaemic, des)
national_ci <- confint(national)
cat(sprintf("National prevalence: %.1f%% (95%% CI %.1f-%.1f)\n",
            100 * coef(national), 100 * national_ci[1], 100 * national_ci[2]))

severity <- svymean(~factor(anaemia_level), des)
write.csv(data.frame(level = c("severe", "moderate", "mild", "not anaemic"),
                     share = as.numeric(coef(severity)),
                     se = as.numeric(SE(severity))),
          file.path(tables, "prevalence_national.csv"), row.names = FALSE)

by_covariate <- do.call(rbind, lapply(covariates, prevalence_by, design = des))
write.csv(by_covariate, file.path(tables, "prevalence_by_covariate.csv"),
          row.names = FALSE)

by_state <- prevalence_by("state_name", des)
write.csv(by_state[order(-by_state$prevalence), ],
          file.path(tables, "prevalence_by_state.csv"), row.names = FALSE)

by_district <- prevalence_by("district_name", des)
by_district$imprecise <- by_district$n_children < 50
write.csv(by_district[order(-by_district$prevalence), ],
          file.path(tables, "prevalence_by_district.csv"), row.names = FALSE)
cat(sprintf("Districts with < 50 children (flagged imprecise): %d\n",
            sum(by_district$imprecise)))

# ---- 2. Survey-weighted logistic regression --------------------------------
crude <- do.call(rbind, lapply(covariates, function(v) {
  m <- svyglm(as.formula(paste("anaemic ~", v)), design = des,
              family = quasibinomial())
  odds_ratios(m, "crude")
}))

# Adjusted model: complete cases on all covariates.
complete <- complete.cases(df[, covariates])
cat(sprintf("Adjusted model uses %d of %d children (complete cases)\n",
            sum(complete), nrow(df)))
adjusted_model <- svyglm(as.formula(paste("anaemic ~",
                                          paste(covariates, collapse = " + "))),
                         design = subset(des, complete),
                         family = quasibinomial())
adjusted <- odds_ratios(adjusted_model, "adjusted")

ors <- merge(crude, adjusted, by = "term", suffixes = c("_crude", "_adjusted"),
             sort = FALSE)
ors <- ors[, c("term", "odds_ratio_crude", "ci_low_crude", "ci_high_crude",
               "odds_ratio_adjusted", "ci_low_adjusted", "ci_high_adjusted")]
write.csv(ors, file.path(tables, "weighted_odds_ratios.csv"), row.names = FALSE)

cat("\nTop states by prevalence:\n")
print(head(by_state[order(-by_state$prevalence),
                    c("level", "n_children", "prevalence")], 5), row.names = FALSE)
cat("\nAdjusted odds ratios:\n")
print(format(adjusted[, -1], digits = 3), row.names = FALSE)