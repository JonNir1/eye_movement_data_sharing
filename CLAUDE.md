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

## Environment

Use the project venv — the system Python does not have the required packages:

```bash
./.venv/Scripts/python.exe
```

Python 3.14. Dependencies in `requirements.txt`.

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
