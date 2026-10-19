# Run one equivalence test or Bayes factor with TOSTER / BayesFactor and write the result as JSON.
#
# Run by `helpers/equivalence.py`:  Rscript equivalence.R job.json
#
# The job is a JSON file with `op`, `output` (JSON path) and the arguments of that op:
#   tost_t        x, y, low, high, var_equal, alpha          two-sample TOST on raw bounds (t_TOST)
#   tost_wilcox   x, y, low, high, alpha                     the same with the Wilcoxon-Mann-Whitney test (wilcox_TOST)
#   bf_ttest      x, y, rscale, one_sided (null, "greater" or "less")      JZS Bayes factor, BF10
#   bf_contingency table (rows x columns), fixed_margin                    BF10 of association vs independence
#   bf_summary    estimate, se, df (null for the normal), prior_sd     BF10 of a coefficient from its estimate and SE (bayesplay)
#   bf_term       data (CSV path), factors, formula_full, formula_null, random, rscale_fixed, rscale_cont, iterations, seed
#                 BF10 of the term the full formula has and the null formula lacks (lmBF; random = random factors)

suppressPackageStartupMessages({
  library(TOSTER)
  library(BayesFactor)
  library(bayesplay)
  library(jsonlite)
})

job <- fromJSON(commandArgs(trailingOnly = TRUE)[1])

tost_summary <- function(fit, estimate_row) {
  # `fit$effsize` has the point estimate and its (1 - 2 alpha) interval; `fit$TOST` one row per one-sided test
  effect <- fit$effsize[estimate_row, ]
  list(
    estimate = effect[["estimate"]], ci_low = effect[["lower.ci"]], ci_high = effect[["upper.ci"]],
    p_lower = fit$TOST["TOST Lower", "p.value"], p_upper = fit$TOST["TOST Upper", "p.value"]
  )
}

bayes_factor_summary <- function(bf, row = 1) {
  extracted <- extractBF(bf)
  list(bf10 = extracted$bf[row], error = extracted$error[row])
}

run_job <- function(job) {
  switch(
    job$op,
    tost_t = {
      fit <- t_TOST(
        x = job$x, y = job$y, eqb = c(job$low, job$high), eqbound_type = "raw",
        var.equal = isTRUE(job$var_equal), alpha = job$alpha
      )
      c(tost_summary(fit, "Raw"), list(df = fit$TOST["TOST Lower", "df"]))
    },
    tost_wilcox = {
      fit <- wilcox_TOST(x = job$x, y = job$y, eqb = c(job$low, job$high), eqbound_type = "raw", alpha = job$alpha)
      tost_summary(fit, "Median of Differences")
    },
    bf_ttest = {
      interval <- switch(
        if (is.null(job$one_sided)) "none" else job$one_sided,
        greater = c(0, Inf), less = c(-Inf, 0), none = NULL
      )
      bf <- ttestBF(x = job$x, y = job$y, rscale = job$rscale, nullInterval = interval)
      bayes_factor_summary(bf)       # with an interval, the first row is the directional hypothesis against the point null
    },
    bf_contingency = {
      bf <- contingencyTableBF(as.matrix(job$table), sampleType = "indepMulti", fixedMargin = job$fixed_margin)
      bayes_factor_summary(bf)
    },
    bf_summary = {
      observed <- if (is.null(job$df)) {
        likelihood(family = "normal", mean = job$estimate, sd = job$se)
      } else {
        likelihood(family = "student_t", mean = job$estimate, sd = job$se, df = job$df)
      }
      alternative <- prior(family = "normal", mean = 0, sd = job$prior_sd)
      null <- prior(family = "point", point = 0)
      list(bf10 = as.numeric(integral(observed * alternative) / integral(observed * null)), error = NULL)
    },
    bf_term = {
      data <- read.csv(job$data, stringsAsFactors = FALSE)
      for (column in intersect(as.character(job$factors), names(data))) data[[column]] <- factor(data[[column]])
      random <- as.character(unlist(job$random))
      set.seed(job$seed)
      fit <- function(formula) lmBF(
        as.formula(formula), data = data, whichRandom = random, rscaleFixed = job$rscale_fixed, rscaleCont = job$rscale_cont,
        iterations = job$iterations, progress = FALSE
      )
      bayes_factor_summary(fit(job$formula_full) / fit(job$formula_null))
    },
    stop("unknown op: ", job$op)
  )
}

write_json(run_job(job), job$output, auto_unbox = TRUE, digits = NA, na = "null")
