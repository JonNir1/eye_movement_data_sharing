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
analysis/   five notebooks, one per research question, plus helpers/
output/     exported figures
data_store/ source data AND the derived parquet frames (gitignored)
```

**`data_store/` is not safe to delete.** It holds the two irreplaceable source files -
`Godwin_2025_dataset.xlsx` and the frozen `Godwin_2025_metadata.csv` - alongside the
regenerable `*.parquet` frames. To force a rebuild, delete only the parquet files, or call
`load_or_build(rebuild=True)`.

The notebooks are numbered but independent — each calls `helpers.dataset.load_or_build()`
and runs from a cold kernel. The parquet cache is an optimization, never a dependency
between notebooks.

| Notebook | Question |
|---|---|
| `01_dataset_and_descriptives` | What is the corpus, and does it match Godwin et al.? |
| `02_sharing_and_article_features` | Is sharing associated with other article characteristics? |
| `03_citation_counts` | Do sharing articles accrue more raw citations? |
| `04_citation_dynamics` | When does the advantage appear, and does it persist? |
| `05_fwci` | Do sharing articles score higher on field-weighted impact? |

`analysis/helpers/` holds only code that two or more notebooks import (`config`, `dataset`,
`stats`, `plotting`). Single-notebook helpers stay inline on purpose. There is no
`__init__.py` anywhere — these are namespace packages, and helper modules use absolute
imports (`from helpers.config import ...`).

## Revision plan (v2)

The submitted version is under major revision (BR-NCR-26-005, resubmit by 09-Feb-2027). The
reviewer's substantive asks: include other known citation predictors rather than testing
sharing alone, stop describing near-significant effects as noteworthy, and add Bayesian
analyses. The notebooks are the work-in-progress answer to that.

Results order in the revised manuscript:

1. feature descriptives and appendix figures (nb 01)
2. sharing status against the other article features (nb 02)
3. multivariable regression on total citations: age and venue dominate, sharing is null (nb 03)
4. 3-year cumulative citations on the same covariates, normalizing for age (nb 03/04)
5. citation dynamics (nb 04)
6. FWCI (nb 05)

Decisions already taken, each of which needs re-arguing before it is changed:

- **The 3-year window is determined, not chosen.** It is the largest k for which every article
  has k complete post-publication calendar years (latest publication 2022, and 2025 is closed).
  It is therefore not a sample restriction: all 232 articles are retained, which also respects
  the reviewer's two separate requests not to shrink the sample. Sensitivity analyses run
  downward (1, 2 years); 5 years would drop articles.
- **No two-stage residualization.** Fit one multivariable model with the windowed count as the
  DV. Regressing on venue and then t-testing the residuals gives the right point estimate and
  the wrong standard errors.
- **FWCI is the declared primary endpoint**, as it was in v1. The other two citation DVs are
  secondary operationalizations. No multiplicity correction across the three: they are near
  functions of the same counts, so the effective number of tests is close to one. BH still
  applies within the nb 02 covariate table, where the tests really are distinct.
- **FWCI excludes age from its model**, since OpenAlex normalizes it by field, year, and
  document type. Venue is *not* baked in, and is a plausible mediator rather than a confounder
  (sharers publish in higher-impact venues), so report both the unadjusted and the
  venue-adjusted model and read the difference as mechanism, not as nuisance.
- **Superseded v1 analyses stay in the notebooks**, clearly marked as superseded, so the
  published results remain reproducible. They do not appear in the revised manuscript.

Still open:

- **Bayes factors are not implemented anywhere.** This is the one reviewer request the current
  plan does not touch, and it is what would let the null sharing effect be reported as evidence
  for the null rather than hedged.
- **Whether citation dynamics survives as inference.** Year-by-year tests with N falling from
  232 to 83 are the pattern the reviewer objected to. The intended replacement is a single
  longitudinal model (sharing x year interaction, random intercept per article), with the
  year-by-year plot kept as description. See the TODO at the bottom of nb 04.

## Environment

Use the project venv — the system Python does not have the required packages:

```bash
./.venv/Scripts/python.exe
```

Python 3.14. Dependencies in `requirements.txt`.

**Run notebooks with `analysis/` as the working directory.** The Jupyter kernel puts the
notebook's own directory first on `sys.path`, which is the whole reason `from helpers
import ...` resolves with no path setup in the notebooks. Note this puts `analysis/` on
the path, not the project root, so `from analysis.helpers import ...` does *not* work.

Figure exports are individually gated behind `if False:` blocks calling `save_figure()`; flip the one figure you want to re-export.

**`data_store/Godwin_2025_metadata.csv` is a frozen OpenAlex snapshot and is gitignored.** Every
reported number depends on it. `prepare_data._load_or_fetch_metadata()` will silently re-query
OpenAlex and write a *new* snapshot if the file is missing, so `helpers.dataset` refuses to run
without it rather than regenerating it. If it is ever lost, restore it from a backup or another
checkout — do not let it rebuild. (`Godwin_2025_metadata.backup.csv` sits in the same directory,
so a wholesale `data_store/` delete would take the backup with it.)

Analysis code does not need API credentials: `helpers.dataset` imports the acquisition layer
lazily, so a warm cache runs with no `_api_secrets.py` present at all.

## Sample construction

Check any pipeline change against these counts:

- 251 records after dropping those without a valid title or link
- 13 OpenAlex queries fail, 6 more return duplicate DOIs
- **N = 232** analytic sample
- Groups: `NONE` = 153, `FIX` = 31, `TRIAL` = 25, `PPT` = 23 (sharing total = 79)

Articles that shared at more than one level are classified at the **finest** granularity
available (fixation > trial > participant), so the groups never double-count.

## Conventions that should not change silently

Each of these is a deliberate choice defended in the manuscript. Changing one means
changing the text too.

- **Article age** is weeks since the OpenAlex `publication_date`, not Godwin et al.'s
  categorical "Year Published" field — online-first appearance precedes issue publication.
  Ten articles that OpenAlex dates to 2016 are **deliberately retained** for consistency
  with the original curation.
- **FWCI** is the primary impact metric, preferred over the h-index (which is confounded by
  total output and career length). The five articles with FWCI = 0 are imputed per
  Dengler (2024) — half the minimum non-zero FWCI in the same year — and then log-transformed.
- **Transformations**: `log(citations + 1)`, `log(weeks)`, `log(n_authors)` (skew 1.41 → 0.14).
  Venue two-year mean citedness stays on its native scale; logging over-corrects it
  (skew 0.39 → -1.01).
- **Journal impact** uses OpenAlex "two-year mean citedness" as a JIF proxy, since the
  official JIF is proprietary to Clarivate. OpenAlex serves no historical journal metrics,
  so these are *current* values, not values as of each article's publication — that caveat
  belongs in the text whenever this feature is used.
- **Preprint detection** is a three-stage cascade: OpenAlex venues (28) → CrossRef
  (+11 not already found) → repository search (+8) = 47 total (20.3%). A candidate is
  accepted only when title similarity is high, first *and* last author match exactly, and
  the preprint year is at or before the publication year.
- **Statistics**: alpha = 0.05. Tests of the sharing hypothesis are **one-sided**;
  multiplicity is handled with Benjamini-Hochberg FDR.
- **Collinearity** is currently clean and should stay so: max pairwise r = 0.29,
  max VIF = 1.2 across the seven predictors.

## Result invariants

If a refactor moves any of these, it is a regression until proven otherwise.

- **FWCI**: sharing > non-sharing, one-sided Mann-Whitney p = 0.03, medians 1.51 vs 1.13.
  Effect is small (r_rb = 0.15, CLES = 0.56).
- **Age-adjusted residuals**: sharers accrue ~18.3% more citations, but only marginally
  (t(230) = 1.44, p = 0.076, d = 0.2).
- **Citation dynamics**: significant cumulative advantage in Years 1-4, marginal in 5-7,
  null by Year 8. Citation *velocity* differs only in Year 1 — read as early visibility
  rather than a sustained rate difference. Attrition caveat: N = 232 at Year 1 falls to
  83 by Year 7, with only 19 sharers left.
- **Granularity never matters**: null for FWCI (H = 2.91, p = 0.23), for residuals
  (H = 1.73, p = 0.42), and in every post-publication year.
- **Covariate comparison**: sharers publish in higher-impact venues, are 4.35x more likely
  to have a preprint, and are more recent. No difference in author count, U.S. authorship,
  or open-access status.
- Covariate prevalence: preprint 47 (20.3%), U.S. author 98 (42.2%), open access 193 (83.2%).
