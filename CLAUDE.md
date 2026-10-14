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
analysis/   ten notebooks, one per research question, plus helpers/
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
| `08_fwci` | Do sharing articles score higher on field-weighted impact? |
| `09_results_figures` | The manuscript figures that summarize nb 07 and nb 08 together: sharing vs. non-sharing violins and the supplementary granularity bars |
| `10_citation_dynamics` | When does the advantage appear, and does it persist? |

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
4. FWCI (nb 08), with the main figure for items 3 and 4 and the supplementary granularity figure (nb 09)
5. citation dynamics, year by year post-publication (nb 10)

Decisions already taken, each of which needs re-arguing before it is changed:

- **No fixed k-year window, and no cutoff sensitivity analyses.** An earlier plan added a second
  citation DV: cumulative citations over the first k = 3 post-publication years, k being the
  largest value for which every article has k complete calendar years, with robustness checks
  sweeping k down to 1 and 2. That DV is dropped, and so are the sweeps.

  It answered the same question as nb 07, which already adjusts for exposure with
  `log_weeks_since_pub` rather than equalizing it by construction, so the two were near functions
  of the same counts. Sweeping k, or re-running the shared-year count at an earlier cutoff, only
  discards citations; the temporal question either would gesture at is answered directly by the
  year-by-year model in nb 10. Report the one determined cutoff; do not sweep it.
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
  OLS with no venue term and standard errors clustered by venue (Model 0), OLS with venue impact as
  a covariate (Model 1), a mixed-effects model with venue as a random intercept (Model 2), and a
  mixed-effects model with both (Model 3). Model 3 is the most conservative.

  **Models 0 and 3 bracket the sharing estimate; neither supersedes the other, and the gap between
  them is not read as a mediated effect.** Venue's causal role is not identified by this corpus. If
  sharing leads to publication in higher-impact venues, venue is a mediator, Model 0 is the total
  association and Model 3 the direct one. If a journal's data policy instead drives whether its
  authors share, venue is a confounder, Model 0 is confounded upward and Model 3 is the estimate of
  interest. Venue sharing rates running from 0% (*Visual Cognition*, *Journal of Eye Movement
  Research*) to 82% (*PLoS ONE*) look more like editorial policy than author choice, so the
  confounder reading is live. Report Model 0 as a bound, not as the preferred estimate.

  The other covariates stay in Model 0 even though they also correlate with sharing, because they
  are not comparable cases: publication age is a common cause of both sharing and citations (and is
  this DV's exposure term), so dropping it would confound rather than decompose, and preprint
  availability is not caused by sharing but reflects the same open-science disposition, placing it
  beside the sharing predictor rather than downstream of it.

  Model 0 handles the within-venue
  correlation in the variance (CR1, t on G - 1 df: 22 for the 23 venues, 19 for the sharers-only
  granularity models), since a random intercept would also condition on venue. The mixed models
  (2 and 3, in nb 07 and nb 08) are fitted by REML, which gives less biased variance components (venue
  variance, ICC, R^2) with only 23 venues; ML would only be needed to compare fixed-effects structures
  by AIC or likelihood ratio, which these notebooks do not do (nb 10's R fits are ML, because its
  time-term comparison changes the fixed effects). Convergence, R^2 and
  venue ICC are shown for each, and the sharing effect is compared across all four (AIC/BIC are
  deliberately not used); nb 07 and nb 08 share `helpers.stats.model_diagnostics` and `sharing_effect`
  for this, `pairwise_contrasts` for the three granularity contrasts and `granularity_wald_test` for the omnibus test. The granularity
  comparison repeats the four models on the 80 sharers only (baseline PARTICIPANT), with the pairwise contrasts (two-sided, BH across
  the three within each model) shown before one omnibus Wald test per model.

  **The granularity analyses report Model 1, not Model 3.** The 80 sharers span 20 venues of which 9
  contribute a single article, which is too thin to estimate between-venue variance: a singleton
  venue carries no information toward separating venue variance from residual. The fit shows it,
  with the venue variance collapsing to 0.0002 against a residual of 1.03 for FWCI (statsmodels
  flags the MLE as on the boundary) and reaching only ICC 0.026 for citations. Calling that a
  mixed-effects model overstates what was fitted. One rule for both notebooks, so the two
  granularity analyses match each other; it does break the match with the main analyses, which
  report Model 3, so the Methods or the table note needs a sentence saying why. Nothing hinges on
  it: the omnibus is null under either specification, differing in the third decimal.

  **All three pairwise contrasts are examined, and the omnibus is what gets reported.** With
  PARTICIPANT as baseline the model gives only FIXATION and TRIAL against it, which hides the
  largest gap: for FWCI, TRIAL sits essentially on PARTICIPANT (0.98x) while FIXATION - TRIAL is
  1.64x (p = 0.097), the biggest of the three and still null after FDR (0.186). For citations all
  three are flat (1.18x to 1.40x, p = 0.19 to 0.52). Report the omnibus in the main text and state
  that all three pairwise contrasts were examined with none significant, so the search is visible
  and the reader is not left asking why only two of three appeared; the contrast table itself goes
  to Supplementary S6 with its FDR column. Chasing pairwise contrasts after a null omnibus is
  unprotected testing, and a lone p = 0.097 reported as noteworthy is exactly what the reviewer
  objected to.
- **Citation dynamics is one mixed-effects model of yearly citation flow (nb 10)**, fitted in R with
  lme4 (`helpers.r_models.fit_glmm`): `citations ~ is_sharing_data * k + has_us_author +
  is_open_access + has_preprint + log_number_of_authors + venue_impact + factor(publication_year) +
  (1 | venue_name/article)`. Articles are nested in venues; publication year is a categorical fixed
  effect, and `log_weeks_since_pub` is left out because cohort and k already carry age. Four fits go
  through the same R bridge: **the primary is a Gaussian on `log(1 + citations)`, chosen a priori and
  not by AIC**, with the raw-count Gaussian, Poisson and negative binomial as reference and robustness
  fits. Thelwall & Wilson (2014) recommend adding one, taking the log and using the general linear
  model for citation data, and their simulations cover this regime (the yearly flow fits a discrete
  lognormal, mu = 0.63, sigma = 1.19): there `log(x + 1)` keeps the nominal 5% false-positive rate and
  is the most powerful trustworthy option, while negative binomial regression falsely flags factors
  about 3x too often at log-SD 1 and about half the time at log-SD 2, worsening with sample size, so
  its conclusions cannot be trusted. The discrete lognormal they prefer has no mixed-model
  implementation, so `log(1 + y)` is the stand-in. The `+ 1` is not driving the result (the notebook's
  offset-sensitivity table: dropping the zeros gives b = -0.042, the same as with `+ 1`). AIC compares
  only Poisson with NB (a Gaussian density is not comparable to a count pmf, nor to a Gaussian on
  another outcome); it never picks the primary. The sharing main effect is the group difference at k = 0, the
  excluded publication year, so it is not interpreted: the sharing-by-time interaction is. Time is
  tried as `k`, `log(k)`, `k + log(k)`, `k^2`, a quadratic `k + k^2` and two `(main only)` variants, within the
  primary family (where the ML AIC is comparable), and the simplest is kept unless another lowers the AIC by
  more than 2. The two `(main only)` variants have the sharing-by-time interaction with linear k but no main
  effect of k, so their interaction coefficient is the sharing group's whole slope rather than a difference
  in slopes; only their AIC is comparable. The non-linear term in `k + log(k)` and `k + k^2` is shared by
  both groups (no sharing x log(k) or sharing x k^2 interaction; for the quadratic this was rejected on the
  NB fit, LRT p = 0.31, and not re-tested on the Gaussian), so the sharing difference stays linear in time
  and one coefficient summarizes it. The quadratic is fitted on standardized k (`k_z`, the same model as raw
  k): raw k and k^2 are nearly collinear and lme4 does not converge on them. The model is a random
  intercept per article and venue only (no random slope for k, no serial correlation), and cohort fixed
  effects plus k absorb any linear calendar-year trend into the k slope; both are stated in the notebook
  as limitations. Count fits use the bobyqa optimizer. `glmer.nb` estimates theta in an outer loop and takes
  about two minutes per fit (the only slow fit now); `fit_glmm(theta=...)` holds it fixed, which
  reproduces the `glmer.nb` AIC to two decimals.
- **FWCI is the declared primary endpoint**, as it was in v1. The other two citation DVs are
  secondary operationalizations. No multiplicity correction across the three: they are near
  functions of the same counts, so the effective number of tests is close to one. BH still
  applies within the nb 05 covariate table, where the tests really are distinct.
- **FWCI excludes age from its model**, since OpenAlex normalizes it by field, year, and
  document type. Venue is *not* baked in, so report both the unadjusted and the venue-adjusted
  model. Whether venue is a mediator or a confounder is not identified here, so the pair brackets
  the estimate and the difference between them is **not** read as mechanism (see the nb 07 venue
  decision above for the full argument). In nb 08 the
  unadjusted one is Model 0: OLS with no venue term and standard errors clustered by venue (CR1, t on
  G - 1 = 22 df, since there are only 23 venues). A venue-free model cannot absorb the within-venue
  correlation (ICC about 0.08, about 10 articles per venue, design effect about 1.7) through a random
  effect, because a random intercept also conditions on venue and so is not venue-free.
- **Superseded v1 analyses stay in the notebooks**, clearly marked as superseded, so the
  published results remain reproducible. They do not appear in the revised manuscript.

Still open:

- **Bayes factors are not implemented anywhere.** This is the one reviewer request the current
  plan does not touch, and it is what would let the null sharing effect be reported as evidence
  for the null rather than hedged.
- **Citation dynamics by sharing granularity (nb 10).** The sharing vs. non-sharing model is
  implemented (see "Decisions already taken"); the same model with `sharing_class * k` on the 80
  sharers is not yet. Also worth one sentence in the manuscript on why time is binned annually, since
  the reviewer's page-6 comment suggests expressing things in weeks: OpenAlex serves
  `counts_by_year`, so annual is the finest resolution the source supports.

## For the discussion

Points that belong in the Discussion rather than the Results, parked here so they are not
rediscovered or lost. Nothing here is written yet.

- **The venue mechanism, and the venue-free estimates.** Results report only Model 3, the most
  conservative specification, which keeps that section short and avoids specifying four models in
  the Methods. The venue-free estimates belong here instead, in a paragraph on what venue is doing:
  `b = 0.14, p = 0.16` for citation counts (nb 07) and `b = 0.17, p = 0.19` for FWCI (nb 08), against
  `0.04` and `0.07` once venue is adjusted for. Adjusting for venue cuts the estimate by roughly 70%
  and 60% respectively, and both ends of that range are null.

  The paragraph's job is to lay out the two readings without picking one, since this corpus cannot
  settle the direction (see the nb 07 venue decision above). Either higher-impact journals *require*
  sharing through their data policies, making venue a confounder, or sharing *leads to* publication
  in higher-impact venues, making it a mediator. The 0% to 82% spread in venue sharing rates favours
  the policy reading, but it is suggestive, not decisive. Say that the conclusion holds either way.

- **Each group against the field average, which only FWCI supports.** `log(FWCI) = 0` is exactly the
  field average, and nb 08 uses plain `log`, not `log1p`, so that reference survives the transform.
  It is the one absolute benchmark in the paper, and the contrast between groups throws it away.
  Unadjusted, non-sharers sit *below* field average at a geometric mean of 0.827 (95% CI 0.703 to
  0.973, p = 0.022) and sharers sit *at* it, 1.047 (0.825 to 1.330, p = 0.70). Adjusted (Model 3
  marginal means at covariate means) neither differs from 1: 0.870 (p = 0.22) and 0.930 (p = 0.61).
  That asymmetry is the same story the rest of the paper tells, with the apparent FWCI advantage
  being covariate structure rather than sharing.

  Two caveats, both of which have to be stated if this is used. The five imputed FWCI zeros sit at
  log -2.3 to -3.0 against a median log-FWCI of exactly 0, so each is worth about three articles'
  deviation, and they pull a test against a fixed reference in a way they do not pull a
  between-group contrast, where they partly cancel: dropping them moves non-sharers from 0.827 to
  0.869 and p from 0.022 to 0.077. And "field average" is OpenAlex's field assignment, which is
  coarse, so sitting below it may reflect which journals the corpus samples. It is a descriptive
  fact about this corpus, not an inference about sharing, and should be worded that way.

  By granularity nothing differs from 1, before or after adjustment (unadjusted 1.128 / 0.753 /
  1.308 for participant / trial / fixation, adjusted 0.875 / 0.862 / 1.409, nothing surviving FDR,
  fixation closest at p = 0.077 adjusted). Trial's apparent dip is largely its two imputed zeros:
  excluding them moves it from 0.753 to 0.915. At N = 23 to 31 the intervals span roughly 0.5 to 2.0
  on the FWCI scale, so the honest statement is that the data cannot locate any class relative to
  the field, not that the classes match it.

- **Robustness, with a pointer to the code.** One sentence that the findings hold across model
  specifications (with and without venue, as fixed effect, as random intercept, or both) and across
  distribution families, referencing the GitHub repository. The specification range is what makes
  this defensible, so it is worth stating even though the Results report a single model.

- **Why time is binned annually (nb 10).** The reviewer's page-6 comment asks whether things could
  be expressed in weeks. OpenAlex serves `counts_by_year`, so annual is the finest resolution the
  source supports. One sentence, or the comment reads as unaddressed.

- **Journal impact is a current value applied retrospectively.** OpenAlex serves no historical
  journal metrics, so two-year mean citedness is today's value attached to articles published up to
  nine years earlier. This is already in "Conventions" as a repo note; it needs to reach the
  manuscript text wherever the feature is used.

## Environment

Use the project venv — the system Python does not have the required packages:

```bash
./.venv/Scripts/python.exe
```

Python 3.14. Dependencies in `requirements.txt`.

**R** (4.5.2) is used only for the mixed models `statsmodels` cannot fit (Poisson and negative-binomial
LMEs in nb 10, and the Gaussian one they are compared with), through `lme4`. The package library is `C:\Users\nirjo\R_library\4.5`, and `R_LIBS_USER`
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
notebooks 07, 08 and 10 call `fig.show()` with the renderer set to `"browser"`, which opens a tab
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
- **Figure annotations** (nb 09): the sharing coefficient is `b` (unstandardized, on the log scale), not `beta`,
  from Model 3, written `b=0.04, p=0.77` (leading zero, deliberately not APA). Both outcomes are plotted on
  the scale the models are fitted on, `log(1 + citations)` and `log(imputed FWCI)`, with tick labels in the
  original units; colors come from `CLASS_COLORS`. The one exception is the supplementary granularity
  figure, whose bars show raw means +/- SEM (citations through 2025, and FWCI with a line at 1, the field
  average): its caption has to say the omnibus tests were fitted on the log scale.
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
  multivariable models on `log(imputed FWCI)` (nb 08) are all null: the sharing coefficient is
  b = 0.172 (cluster-robust p = 0.19) with no venue term (Model 0), 0.092 (p = 0.54) with venue impact
  (Model 1), 0.119 (p = 0.45) with a venue random intercept (Model 2), and 0.067 (p = 0.67) with both
  (Model 3). Adjusting for venue about halves the estimate; that is consistent with venue as a
  mediator but equally with venue as a confounder, so it is reported as a bracket and not as a
  decomposition. The models explain little: marginal R^2 0.06 (Model 0) to 0.09 (Model 3), venue
  ICC about 0.08.
- **Multivariable citation models (nb 07)**: the sharing coefficient is null in all four: b = 0.137
  (cluster-robust p = 0.16) with no venue term (Model 0), then 0.075 (p = 0.55), 0.086 (p = 0.52) and
  0.038 (p = 0.77) for Models 1-3, so adjusting for venue cuts the estimate by about 70%. In
  Model 3, venue impact is significant (b = 0.180, p = 0.013) and log age dominates (b = 1.12,
  p < 0.001). The univariate
  one-sided t-test on `log(1 + citations)` is also null (t = 0.40, p = 0.35, d = 0.06).
- **Age-adjusted residuals** (v1, superseded, nb 07): sharers accrue ~17.8% more citations, but only marginally
  (t(232) = 1.38, p = 0.085, d = 0.19).
- **Citation dynamics, primary model (nb 10)**: Gaussian on `log(1 + citations)`, converged and not
  singular (venue variance 0.025, article-within-venue 0.283, residual 0.239). With linear k, the
  `is_sharing_data:k` coefficient is b = -0.0416 (SE 0.0160, p = 0.0094): sharing articles' yearly
  citations fall about 4% per year relative to non-sharing ones (`exp(b) - 1`, on `1 + citations`).
  Offset sensitivity (coefficients are on different scales, so compare |t|): `log(y + 0.5)` b = -0.051
  (|t| 2.48), `+ 0.25` b = -0.061 (2.34), `+ 0.1` b = -0.073 (2.15), zeros dropped (N = 1117) b = -0.042
  (2.10), against 2.60 for `+ 1`. Time term, within this family (AIC vs. linear k): log(k) +0.5, k^2
  -0.1, `k + k^2` -6.3, `k^2 (main only)` -6.5, `log(k) (main only)` -3.4 and `k + log(k)` -16.2, which is
  the one used. In it the sharing-by-time slope is b = -0.047 per year (SE 0.016, p = 0.0029), against
  b = -0.042 (p = 0.0094) with linear k and b = -0.093 per SD of k (SE 0.033, p = 0.0043) in the
  quadratic: the sign and significance do not depend on the time term. Predicted citations
  (geometric-mean scale) for a typical article over k = 1 to 8: sharing 2.3, 2.6, 2.5, 2.4, 2.1, 1.9, 1.6,
  1.4 and non-sharing 2.0 rising to 2.4 at k = 3 to 4 then falling to 2.0.
- **Citation dynamics, robustness fits (nb 10)**: negative binomial beats Poisson on AIC (5574.3 vs
  5636.7; Poisson overdispersion ratio 1.05, NB theta 11.5). With linear k, the NB `is_sharing_data:k`
  rate ratio is 0.939 (95% CI 0.902 to 0.978, p = 0.0025): about 6% fewer yearly citations per year.
  Poisson gives 0.944 (p = 0.0007) and the raw-count Gaussian b = -0.21 citations/year (p = 0.0056).
  On the NB fit (an earlier run; the notebook now compares time terms on the primary family only) the
  shared quadratic also won (AIC -6.5) with a sharing-by-slope of b = -0.142 per SD
  of k (rate ratio 0.87, p = 0.0009), a negative common curvature (p = 0.003) and a null sharing main
  effect at the mean k (p = 0.97). All four families agree on the sign and significance.
- **Citation dynamics** (v1, superseded, nb 10): significant cumulative advantage in Years 1-4, marginal in 5-7,
  null by Year 8. Citation *velocity* differs only in Year 1 — read as early visibility
  rather than a sustained rate difference. Attrition caveat: N = 234 at Year 1 falls to
  84 by Year 7, with only 19 sharers left.
- **Granularity never matters**: null for FWCI (H = 3.08, p = 0.21), for residuals
  (H = 2.00, p = 0.37), in every post-publication year, in the nb 07 multivariable models
  (omnibus Wald p = 0.38, 0.41, 0.56, 0.44 for Models 0-3), and in the nb 08 FWCI models on the 80
  sharers (omnibus Wald p = 0.36, 0.16, 0.29, 0.16 for Models 0-3). In nb 08 the venue variance of
  Model 3 still collapses to about zero on the sharers (ICC 0.0002), so it is effectively Model 1; in
  nb 07 it no longer does under REML (ICC 0.03).
- **Covariate comparison**: sharers publish in higher-impact venues, are 4.30x more likely
  to have a preprint, and are more recent. No difference in author count, U.S. authorship,
  or open-access status.
- Covariate prevalence: preprint 47 (20.1%), U.S. author 99 (42.3%), open access 199 (85.0%).
