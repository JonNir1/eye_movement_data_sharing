# Fit one (generalized) linear mixed model with lme4 and write the results as JSON.
#
# Run by `helpers/r_models.py`:  Rscript fit_glmm.R job.json
#
# The job is a JSON file with: data (CSV path), formula, family ("gaussian", "poisson" or "nb"),
# factors (columns to treat as factors), output (JSON path), and optionally newdata (CSV path) for
# fixed-effects-only predictions with standard errors on the link scale, and theta (family "nb" only)
# to fix the negative-binomial dispersion instead of estimating it, which is much faster.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

job <- fromJSON(commandArgs(trailingOnly = TRUE)[1])

read_data <- function(path, factors) {
  data <- read.csv(path, stringsAsFactors = FALSE)
  for (column in intersect(factors, names(data))) data[[column]] <- factor(data[[column]])
  data
}

# the default optimizer stops short of lme4's gradient tolerance on these models; bobyqa converges
count_control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

fit_negative_binomial <- function(formula, data, theta) {
  if (is.null(theta)) return(glmer.nb(formula, data = data, control = count_control))
  glmer(formula, data = data, family = MASS::negative.binomial(theta), control = count_control)
}

fit_model <- function(formula, data, family, theta) {
  switch(
    family,
    gaussian = lmerTest::lmer(formula, data = data, REML = FALSE),
    poisson = glmer(formula, data = data, family = poisson, control = count_control),
    nb = fit_negative_binomial(formula, data, theta),
    stop("unknown family: ", family)
  )
}

fixed_effects <- function(fit) {
  table <- coef(summary(fit))
  p_column <- ncol(table)
  data.frame(
    term = rownames(table),
    estimate = table[, "Estimate"],
    se = table[, "Std. Error"],
    statistic = table[, p_column - 1],
    p = table[, p_column],
    row.names = NULL
  )
}

variance_components <- function(fit) {
  components <- as.data.frame(VarCorr(fit))
  data.frame(
    group = components$grp, term = components$var1, term2 = components$var2, variance = components$vcov, sd = components$sdcor
  )
}

predictions <- function(fit, newdata) {
  terms_fixed <- delete.response(terms(fit, fixed.only = TRUE))
  frame <- model.frame(terms_fixed, newdata, xlev = .getXlevels(terms_fixed, model.frame(fit)))
  design <- model.matrix(terms_fixed, frame, contrasts.arg = attr(getME(fit, "X"), "contrasts"))
  link <- as.vector(design %*% fixef(fit))
  link_se <- sqrt(rowSums((design %*% as.matrix(vcov(fit))) * design))
  data.frame(link = link, link_se = link_se)
}

data <- read_data(job$data, job$factors)
messages <- character()
fit <- withCallingHandlers(
  fit_model(as.formula(job$formula), data, job$family, job$theta),
  warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
# lme4 records its convergence problems in optinfo; other warnings (glmer.nb emits a harmless
# recycling warning) are kept in `messages` but do not mean the fit failed
convergence_problems <- unlist(fit@optinfo$conv$lme4$messages)
messages <- unique(c(messages, convergence_problems))

# Nakagawa's marginal (fixed effects) and conditional (fixed + random effects) R^2, with random slopes handled by `performance`;
# approximate for the count families. NA when `performance` is missing or cannot compute it
nakagawa_r2 <- function(fit) {
  r2 <- tryCatch(suppressMessages(suppressWarnings(performance::r2_nakagawa(fit))), error = function(e) NULL)
  if (is.null(r2)) return(list(marginal = NA, conditional = NA))
  list(marginal = unname(r2$R2_marginal), conditional = unname(r2$R2_conditional))
}

is_count <- job$family != "gaussian"
r2 <- nakagawa_r2(fit)
result <- list(
  fit = list(
    n_obs = nobs(fit),
    log_lik = as.numeric(logLik(fit)),
    # AIC is only comparable between fits of the same family and outcome (a Gaussian density and a count
    # pmf, or two Gaussians on different outcomes, are not), so callers decide what to compare
    aic = AIC(fit),
    r2_marginal = r2$marginal,
    r2_conditional = r2$conditional,
    singular = isSingular(fit),
    converged = length(convergence_problems) == 0,
    messages = messages,
    overdispersion = if (is_count) sum(residuals(fit, type = "pearson")^2) / df.residual(fit) else NA,
    theta = if (job$family != "nb") NA else if (!is.null(job$theta)) job$theta else getME(fit, "glmer.nb.theta")
  ),
  fixed_effects = fixed_effects(fit),
  variance_components = variance_components(fit)
)
if (!is.null(job$newdata)) result$predictions <- predictions(fit, read.csv(job$newdata, stringsAsFactors = FALSE))

write_json(result, job$output, auto_unbox = TRUE, na = "null", digits = NA)
