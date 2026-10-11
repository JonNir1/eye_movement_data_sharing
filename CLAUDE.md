# CLAUDE.md

Guidance for working in this repo. `README.md` covers the scientific background; this file
covers the conventions and invariants the analysis code depends on.

## What this is

Analysis code for a solo-author Brief Report submitted to *Behavior Research Methods*:
"Data Sharing and Citation Impact in Visual Search Research". It extends the open-science
audit of Godwin et al. (2025, *BRM* 57(9):235) by joining their curated corpus with
bibliometric indicators from OpenAlex.

The manuscript itself lives one directory up from the repo (`brief report.docx` is the
current in-progress revision; `brief report.pdf` is the version that was submitted).

## Repository layout

```
data/       acquisition only - OpenAlex/CrossRef fetching, and _api_secrets.py
analysis/   nine notebooks, one per research question, plus helpers/
output/     exported figures
data_store/ source data AND the derived parquet frames (gitignored)
```

**`data_store/` is not safe to delete.** It holds the two irreplaceable source files -
`Godwin_2025_dataset.xlsx` and the frozen `Godwin_2025_metadata.csv` - alongside the
regenerable `*.parquet` frames. To force a rebuild, delete only the parquet files, or call
`load_or_build(rebuild=True)`.

**The parquet cache is a pure function of `(Godwin_2025_dataset.xlsx, Godwin_2025_metadata.csv)`**
- `build_merged()` / `build_analytic_sample()` read only those two files, with no other input and
no dependency on wall-clock time. So it never needs its own backup: given any past copy of the
metadata CSV (the xlsx never changes), `load_or_build(rebuild=True)` reproduces the exact parquet
cache that CSV would have produced.

**Re-fetching the metadata CSV is a deliberate, rare, and consequential action** - it moves the
one thing in this repo that genuinely cannot be reconstructed, and changes every citation-based
number the manuscript reports (N=234, all citation counts, FWCI, ...). Before overwriting the
live `Godwin_2025_metadata.csv`, copy the outgoing file to `Godwin_2025_metadata.backup.csv`
(overwriting *that* is fine - it only ever needs to hold the immediately-prior snapshot, not a
history). After the CSV is replaced, rebuild the parquet cache with `load_or_build(rebuild=True)`
and re-check the "Sample construction" and "Result invariants" numbers below - they are pinned to
whichever snapshot was frozen when they were last verified, and a re-fetch is exactly the kind of
change that can move them.

The notebooks are numbered but independent — each calls `helpers.dataset.load_or_build()`
and runs from a cold kernel. The parquet cache is an optimization, never a dependency
between notebooks.

| Notebook | Question |
|---|---|
| `01_dataset_construction` | What is the corpus, how was it built, and does it match Godwin et al.? |
| `02_feature_descriptives` | What do the article and impact features look like, overall and by sharing status? |
| `03_feature_preprocessing` | How are the continuous features transformed, and does the transformation fix their normality and skewness? |
| `04_feature_correlations` | How do the article features correlate with each other, and is there collinearity among the regression predictors? |
| `05_sharing_and_article_features` | Is sharing associated with other article characteristics? |
| `06_citation_descriptives` | What do the citation and impact outcomes look like, overall and by sharing status, before any modelling? |
| `07_citation_counts` | Do sharing articles accrue more citations, univariately and after adjusting for the other predictors (OLS and mixed-effects models)? |
| `08_citation_dynamics` | When does the advantage appear, and does it persist? |
| `09_fwci` | Do sharing articles score higher on field-weighted impact? |

`analysis/helpers/` holds only code that two or more notebooks import (`config`, `dataset`,
`stats`, `plotting`, `r_models`). Single-notebook helpers stay inline on purpose. There is no
`__init__.py` anywhere — these are namespace packages, and helper modules use absolute
imports (`from helpers.config import ...`).

## Revision plan (v2)

The submitted version is under major revision (BR-NCR-26-005, resubmit by 09-Feb-2027). The
reviewer's substantive asks: include other known citation predictors rather than testing
sharing alone, stop describing near-significant effects as noteworthy, and add Bayesian
analyses. The notebooks are the work-in-progress answer to that.

Results order in the revised manuscript:

1. feature descriptives and appendix figures (nb 02, with the transformation checks in nb 03)
2. sharing status against the other article features (nb 05)
3. multivariable regression on the shared-year citation count: age and venue dominate, sharing is null (nb 07; four models, see below)
4. citation dynamics, year by year post-publication (nb 08)
5. FWCI (nb 09)

Decisions already taken, each of which needs re-arguing before it is changed:

- **No fixed k-year window, and no cutoff sensitivity analyses.** An earlier plan added a second
  citation DV: cumulative citations over the first k = 3 post-publication years, k being the
  largest value for which every article has k complete calendar years, with robustness checks
  sweeping k down to 1 and 2. That DV is dropped, and so are the sweeps.

  It answered the same question as nb 07, which already adjusts for exposure with
  `log_weeks_since_pub` rather than equalizing it by construction, so the two were near functions
  of the same counts. Sweeping k, or re-running the shared-year count at an earlier cutoff, only
  discards citations; the temporal question either would gesture at is answered directly by the
  year-by-year model in nb 08. Report the one determined cutoff; do not sweep it.
- **Citation dynamics stays, as a multivariable model.** The reviewer never asked for the
  year-by-year analysis to go, and it is the only thing in the paper that answers *when* an
  advantage appears, which the aggregate DVs cannot. What has to change is how it is fitted: his
  two substantive results-section comments both land on it (include other citation predictors, and
  stop presenting near-significant effects as noteworthy), and the Years 5-7 "marginal" result is
  the clearest instance of what he objected to.
- **No two-stage residualization.** Fit one multivariable model with the citation count as the
  DV. Regressing on venue and then t-testing the residuals gives the right point estimate and
  the wrong standard errors.
- **Venue is modelled four ways in nb 07**, all on `log(1 + citations through the shared year)`:
  OLS with no venue term and standard errors clustered by venue (Model 0, the total association),
  OLS with venue impact as a covariate (Model 1), a mixed-effects model with venue as a random
  intercept (Model 2), and a mixed-effects model with both (Model 3). Model 3 is the most
  conservative; it is reported beside Model 0, and the difference between them is read as mechanism
  (see the FWCI decision below: venue is a plausible mediator). Model 0 handles the within-venue
  correlation in the variance (CR1, t on G - 1 df: 22 for the 23 venues, 19 for the sharers-only
  granularity models), since a random intercept would also condition on venue. Convergence, R^2 and
  venue ICC are shown for each, and the sharing effect is compared across all four (AIC/BIC are
  deliberately not used). The granularity comparison repeats the four models on the 80 sharers only
  (baseline PARTICIPANT), with one omnibus Wald test per model.
- **Citation dynamics is one mixed-effects model of yearly citation flow (nb 08)**, fitted in R with
  lme4 (`helpers.r_models.fit_glmm`): `citations ~ is_sharing_data * k + has_us_author +
  is_open_access + has_preprint + log_number_of_authors + venue_impact + factor(publication_year) +
  (1 | venue_name/article)`. Articles are nested in venues; publication year is a categorical fixed
  effect, and `log_weeks_since_pub` is left out because cohort and k already carry age. Gaussian,
  Poisson and negative-binomial fits are compared, on AIC for Poisson vs. NB only (a Gaussian density
  is not comparable to a count pmf). The sharing main effect is the group difference at k = 0, the
  excluded publication year, so it is not interpreted: the sharing-by-time interaction is. Time is
  tried as `k`, `log(k)` and a quadratic `k + k^2`, and the simplest is kept unless another lowers the
  AIC by more than 2. The quadratic has the same curvature in both groups, i.e. `is_sharing_data * k_z
  + I(k_z^2)` with no sharing x k^2 interaction (not supported, LRT p = 0.31), so the sharing
  difference stays linear in time and one coefficient summarizes it. It is fitted on standardized k
  (`k_z`, the same model): raw k and k^2 are nearly collinear and lme4 does not converge on them. Count fits use the bobyqa
  optimizer. `glmer.nb` estimates theta in an outer loop and takes about two minutes per fit, so the
  `log(k)` and quadratic refits hold theta at the value estimated for `k` (`fit_glmm(theta=...)`, which
  reproduces the `glmer.nb` AIC to two decimals); their AICs are therefore conditional on that theta.
- **FWCI is the declared primary endpoint**, as it was in v1. The other two citation DVs are
  secondary operationalizations. No multiplicity correction across the three: they are near
  functions of the same counts, so the effective number of tests is close to one. BH still
  applies within the nb 05 covariate table, where the tests really are distinct.
- **FWCI excludes age from its model**, since OpenAlex normalizes it by field, year, and
  document type. Venue is *not* baked in, and is a plausible mediator rather than a confounder
  (sharers publish in higher-impact venues), so report both the unadjusted and the
  venue-adjusted model and read the difference as mechanism, not as nuisance. In nb 09 the
  unadjusted one is Model 0: OLS with no venue term and standard errors clustered by venue (CR1, t on
  G - 1 = 22 df, since there are only 23 venues). A venue-free model cannot absorb the within-venue
  correlation (ICC about 0.07, about 10 articles per venue, design effect about 1.6) through a random
  effect, because a random intercept also conditions on venue and so is not venue-free.
- **Superseded v1 analyses stay in the notebooks**, clearly marked as superseded, so the
  published results remain reproducible. They do not appear in the revised manuscript.

Still open:

- **Bayes factors are not implemented anywhere.** This is the one reviewer request the current
  plan does not touch, and it is what would let the null sharing effect be reported as evidence
  for the null rather than hedged.
- **Citation dynamics by sharing granularity (nb 08).** The sharing vs. non-sharing model is
  implemented (see "Decisions already taken"); the same model with `sharing_class * k` on the 80
  sharers is not yet. Also worth one sentence in the manuscript on why time is binned annually, since
  the reviewer's page-6 comment suggests expressing things in weeks: OpenAlex serves
  `counts_by_year`, so annual is the finest resolution the source supports.

## Environment

Use the project venv — the system Python does not have the required packages:

```bash
./.venv/Scripts/python.exe
```

Python 3.14. Dependencies in `requirements.txt`.

**R** (4.5.2) is used only for the mixed models `statsmodels` cannot fit (Poisson and negative-binomial
LMEs in nb 08), through `lme4`. The package library is `C:\Users\nirjo\R_library\4.5`, and `R_LIBS_USER`
must point at it (`helpers.config.R_LIBRARY_DIR` does, unless the variable is already set). It holds
`lme4` 2.0-1, `lmerTest`, `glmmTMB`, `broom.mixed`, `emmeans`, `performance` and `jsonlite`; `DHARMa` and
`car` are **not** installed. `Rscript` is `C:\Program Files\R\R-4.5.2\bin\Rscript.exe`. Models are fitted by
`helpers.r_models.fit_glmm()`, which runs `helpers/fit_glmm.R` as a subprocess and reads the results back as
JSON: there is no `rpy2` in the venv, and none should be added.

**Run notebooks with `analysis/` as the working directory.** The Jupyter kernel puts the
notebook's own directory first on `sys.path`, which is the whole reason `from helpers
import ...` resolves with no path setup in the notebooks. Note this puts `analysis/` on
the path, not the project root, so `from analysis.helpers import ...` does *not* work.

That is not a style preference, it is the only arrangement that works. `helpers/` imports itself
absolutely (`from helpers.config import ...`), so `analysis/` has to be on the path; adding the
project root as well is what the `analysis.`-prefixed form would require, and depending on both
means the notebooks only run where something injects both. They had drifted to that form and
failed on the first cell from a plain kernel, in PyCharm as well as headless. Keep every notebook
import relative to `analysis/`.

To execute notebooks in batch, run this from `analysis/` with the project venv's interpreter
(spell out the absolute path to it; `analysis/` sits at a different depth in a worktree than in
the main checkout, so a relative path is not portable between them):

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 02_feature_descriptives.ipynb
```

Order does not matter, since each notebook runs standalone from a cold kernel. One caveat:
notebooks 07, 08 and 09 call `fig.show()` with the renderer set to `"browser"`, which opens a tab
per figure. For an unattended run, put a `sitecustomize.py` that no-ops `webbrowser.open` on
`PYTHONPATH` rather than editing the notebooks.

Figure exports are individually gated behind `if False:` blocks calling `save_figure()`; flip the one figure you want to re-export.

**`data_store/Godwin_2025_metadata.csv` is a frozen OpenAlex snapshot and is gitignored.** Every
reported number depends on it. The current snapshot was queried on **2026-09-02** (the manuscript
should state this date, since a later query can return different counts).
`prepare_data._load_or_fetch_metadata()` will silently re-query
OpenAlex and write a *new* snapshot if the file is missing, so `helpers.dataset` refuses to run
without it rather than regenerating it. If it is ever lost, restore it from a backup or another
checkout — do not let it rebuild. (`Godwin_2025_metadata.backup.csv` sits in the same directory,
so a wholesale `data_store/` delete would take the backup with it.)

Analysis code does not need API credentials: `helpers.dataset` imports the acquisition layer
lazily, so a warm cache runs with no `_api_secrets.py` present at all.

That invariant holds for the whole repo, including notebook 01 and `tests/test_corpus.py`, which
both reach `prepare_data` and through it `fetch_metadata`. What keeps it true is that
`fetch_metadata` does **not** import `_api_secrets` at module scope. Credentials are loaded, and
`pyalex` configured, by `_configure_api()` on first network use. It is called from the three entry
points (`fetch_all_metadata`, `fetch_single_metadata`, `fetch_covariates_by_id`), which is
sufficient because every other network helper in that module is reachable only through them.

Adding a module-scope `from _api_secrets import ...` back, or a new network entry point that does
not call `_configure_api()` first, breaks this. In the second case `pyalex` would run unconfigured
rather than failing loudly, so a new entry point must call it.

## Sample construction

Check any pipeline change against these counts:

- 251 records after dropping those without a valid title or link
- 11 OpenAlex queries fail, 6 more return duplicate DOIs
- **N = 234** analytic sample
- Groups: `NONE` = 154, `FIX` = 31, `TRIAL` = 26, `PPT` = 23 (sharing total = 80)

Articles that shared at more than one level are classified at the **finest** granularity
available (fixation > trial > participant), so the groups never double-count.

**The "6" above is not the full duplicate-DOI count - do not read it as "6 of 8 duplicates make
the sample."** `duplicate_records_diagnostic.find_duplicate_doi_records()` finds 8 duplicate-DOI
pairs across the full 525-record corpus (matching `prepare_data._SHARING_CORRECTIONS`'s 8 keys
exactly), and **all 8** have their surviving copy in the N=234 sample, confirmed for both the
June 30 and the current snapshot. The exclusion cascade's own dedup step only catches 6 of them,
because it runs *after* the topic/year/retraction filters: for 6 pairs both copies pass those
filters and reach the dedup step, which drops one; for the other 2 pairs
(`10.1177/1747021820919351`, `10.3758/s13423-021-01920-1`) only one copy ever reaches that step,
its sibling having already failed topic/year/retraction for unrelated reasons - so the cascade
never "sees" a duplicate there, even though the wider diagnostic correctly does.

## Conventions that should not change silently

Each of these is a deliberate choice defended in the manuscript. Changing one means
changing the text too.

- **Article age** is weeks since the OpenAlex `publication_date`, not Godwin et al.'s
  categorical "Year Published" field — online-first appearance precedes issue publication.
  Ten articles that OpenAlex dates to 2016 are **deliberately retained** for consistency
  with the original curation.
- **FWCI** is the primary impact metric, preferred over the h-index (which is confounded by
  total output and career length). The five articles with FWCI = 0 are imputed per
  Dengler (2024) — half the minimum non-zero FWCI in the same year — via
  `helpers.dataset.fwci_with_zeros_imputed()`, then log-transformed. That imputation is applied
  at analysis time, on already-loaded data, and deliberately never baked into the parquet cache.
- **Transformations**: `log(citations + 1)`, `log(weeks)`, `log(n_authors)` (skew 1.39 → 0.14).
  Venue two-year mean citedness stays on its native scale; logging over-corrects it
  (skew 0.38 → -0.97).
- **Journal impact** uses OpenAlex "two-year mean citedness" as a JIF proxy, since the
  official JIF is proprietary to Clarivate. OpenAlex serves no historical journal metrics,
  so these are *current* values, not values as of each article's publication — that caveat
  belongs in the text whenever this feature is used.
- **Preprint detection** is a three-stage cascade: OpenAlex venues (28) → CrossRef
  (+11 not already found) → repository search (+8) = 47 total (20.1%). A candidate is
  accepted only when title similarity is high, first *and* last author match exactly, and
  the preprint year is at or before the publication year.
- **Statistics**: alpha = 0.05. Tests of the sharing hypothesis are **one-sided**; the covariate
  comparisons in nb 05 are two-sided. Multiplicity is handled with Benjamini-Hochberg FDR.
- **Feature matrix**: `build_feature_matrix()` also carries `Venue Name` (a string), the grouping
  factor for the mixed-effects models in nb 07. It is not a predictor, so any loop or correlation over
  the matrix's columns must exclude it, along with `Sharing Class` where relevant.
- **Collinearity** is currently clean and should stay so: max pairwise r = 0.29,
  max VIF = 1.2 across the seven predictors.

## Result invariants

If a refactor moves any of these, it is a regression until proven otherwise.

- **FWCI**: sharing > non-sharing, one-sided Mann-Whitney p = 0.04, medians 1.20 vs 0.92.
  Effect is small (r_rb = 0.14, CLES = 0.57). The Mann-Whitney test ignores venue clustering, and the
  multivariable models on `log(imputed FWCI)` (nb 09) are all null: the sharing coefficient is
  b = 0.172 (cluster-robust p = 0.19) with no venue term (Model 0), 0.092 (p = 0.54) with venue impact
  (Model 1), 0.126 (p = 0.41) with a venue random intercept (Model 2), and 0.072 (p = 0.64) with both
  (Model 3). Adjusting for venue about halves the estimate, in line with venue as a mediator.
- **Multivariable citation models (nb 07)**: the sharing coefficient is null in all four: b = 0.137
  (cluster-robust p = 0.16) with no venue term (Model 0), then 0.075 (p = 0.56), 0.092 (p = 0.48) and
  0.044 (p = 0.73) for Models 1-3, so adjusting for venue cuts the estimate by about two thirds. In
  Model 3, venue impact is significant (b = 0.170, p = 0.012) and log age dominates (b = 1.12,
  p < 0.001). The univariate
  one-sided t-test on `log(1 + citations)` is also null (t = 0.40, p = 0.35, d = 0.06).
- **Age-adjusted residuals** (v1, superseded, nb 07): sharers accrue ~17.8% more citations, but only marginally
  (t(232) = 1.38, p = 0.085, d = 0.19).
- **Citation dynamics, model (nb 08)**: negative binomial beats Poisson on AIC (5574.3 vs 5636.7;
  Poisson overdispersion ratio 1.05, NB theta 11.5). With linear k, the `is_sharing_data:k` rate ratio
  is 0.939 (95% CI 0.902 to 0.978, p = 0.0025): sharing articles' yearly citations fall about 6% per
  year relative to non-sharing ones. Poisson gives 0.944 (p = 0.0007) and Gaussian b = -0.21
  citations/year (p = 0.0056). Time term: log(k) does not improve on k (AIC -0.9), but the shared
  quadratic does (AIC -6.5, so it is the one used). In it the sharing-by-time slope is b = -0.142 per SD
  of k (about 2 years; rate ratio 0.87, p = 0.0009), the common curvature is negative (p = 0.003), and
  the sharing main effect at the mean k is null (p = 0.97). Predicted citations for a typical article:
  sharing 2.5, 2.6, 2.5 down to 1.6 and non-sharing 2.1 rising to 2.5 at k = 5 then falling to 2.2,
  over k = 1 to 8.
- **Citation dynamics** (v1, superseded, nb 08): significant cumulative advantage in Years 1-4, marginal in 5-7,
  null by Year 8. Citation *velocity* differs only in Year 1 — read as early visibility
  rather than a sustained rate difference. Attrition caveat: N = 234 at Year 1 falls to
  84 by Year 7, with only 19 sharers left.
- **Granularity never matters**: null for FWCI (H = 3.08, p = 0.21), for residuals
  (H = 2.00, p = 0.37), in every post-publication year, and in the nb 07 multivariable models
  (omnibus Wald p = 0.38, 0.41, 0.52, 0.37 for Models 0-3).
- **Covariate comparison**: sharers publish in higher-impact venues, are 4.30x more likely
  to have a preprint, and are more recent. No difference in author count, U.S. authorship,
  or open-access status.
- Covariate prevalence: preprint 47 (20.1%), U.S. author 99 (42.3%), open access 199 (85.0%).
