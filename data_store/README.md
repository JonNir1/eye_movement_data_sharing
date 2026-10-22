# Eye-Movement Data-Sharing Corpus

A dataset of **234 visual-search eye-tracking articles** published between 2017 and 2022, each coded for what
research data its authors shared, and joined to bibliometric indicators from
[OpenAlex](https://openalex.org).

It exists to answer one question: do articles that share their data get cited more? The sharing coding comes
from the open-science audit of

> Godwin, H. J., et al. (2025). *Behavior Research Methods*, 57(9), 235.
> <https://doi.org/10.3758/s13428-025-02759-3>

and the citation indicators, the preprint detection and the analytic sample were added for the accompanying
report, *Data Sharing and Citation Impact in Visual Search Research* (under review; preprint at
<https://osf.io/preprints/psyarxiv/q8jrd>). If you use this file, please cite both: Godwin et al. for the
sharing coding, and the report for the joined dataset.

## How it was curated

Godwin et al. screened the visual-search eye-tracking literature and hand-coded, for every article, whether
the authors *claimed* to share data and whether data could *actually* be found, down to the granularity of
what was shared (by fixation, by trial, or by participant). Their spreadsheet is the starting point. Each
surviving article was then looked up in OpenAlex and its metadata attached.

The exclusion cascade, in order, with the number of articles remaining after each step:

| Step | Remaining |
|---|---|
| Godwin et al.'s corpus, after dropping records with neither a title nor a link, de-duplicating on title and link, and removing unpublished entries | 525 |
| Primary human research **and** visual search **and** eye tracking (all three coded `YES`) | 256 |
| Published 2017-2022 (by Godwin et al.'s `YEAR_PUBLISHED`) | 251 |
| OpenAlex lookup succeeded | 240 |
| Not retracted | 240 |
| Unique DOI (one copy kept per duplicate) | **234** |

Two consequences of that cascade are worth knowing. The year filter uses Godwin et al.'s `YEAR_PUBLISHED`,
while OpenAlex's `PublicationDate` reflects online-first appearance, so ten articles carry an OpenAlex date in
2016; they are **deliberately retained**, so that the sample matches the original curation. And because the
retraction and type filters have already been applied, `IsRetracted` is `False` and `PublicationType` is
`article` for every row.

Articles that shared at more than one granularity are classified at the **finest** level available
(fixation > trial > participant), so the groups never double-count:

| `data_sharing_class` | Articles |
|---|---|
| `NONE` | 154 |
| `FIXATION` | 31 |
| `TRIAL` | 26 |
| `PARTICIPANT` | 23 |
| **Total sharing** | **80** |

### Size and snapshot

**234 rows x 81 columns.** Citation counts are a frozen OpenAlex snapshot, queried article by article in late
August and early September 2026; each row's own census timestamp is in `LastUpdate` (these run from 15 August
to 1 September 2026). Citation counts move over time, so any number you compute will differ from a fresh query
against OpenAlex today. That is why the snapshot is frozen rather than re-fetched.

## Where each column comes from

Three naming conventions, each marking a different origin:

| Convention | Origin | Count | Examples |
|---|---|---|---|
| `ALL_CAPS` | Godwin et al.'s hand-coded spreadsheet | 41 | `BY_FIXATION`, `CLAIMED_TO_SHARE`, `YEAR_PUBLISHED` |
| `CamelCase` | OpenAlex, fetched for this study | 37 | `TotalCitations`, `VenueName`, `PublicationDate` |
| `snake_case` | Derived here from the columns above | 3 | `data_sharing_class`, `is_sharing_data` |

**One exception to the rule: `DOI` is `ALL_CAPS` but comes from OpenAlex**, not from Godwin et al. Their
spreadsheet identifies articles by `PAPER_LINK` instead. Every other `ALL_CAPS` column is theirs.

The two sources also disagree in coverage in one place worth noting: `PAPER_TITLE` and `AUTHORS` (Godwin et
al.) are missing for 14 articles, whereas OpenAlex's `Authors` is complete for all 234. Prefer the OpenAlex
columns for anything that has to be populated for every row.

## Main columns

The full 81 are mostly Godwin et al.'s fine-grained sharing codes (data formats, open versus closed formats,
search attempts). These are the ones most analyses need.

| Column | Source | Type | Description |
|---|---|---|---|
| `DOI` | OpenAlex | str | Resolvable DOI URL. **The recommended key** for joining to other sources |
| `OpenAlexID` | OpenAlex | str | OpenAlex work ID, e.g. `W2982222209` |
| `PAPER_LINK` | Godwin | str | The bare DOI or URL as recorded in the original sheet |
| `PAPER_TITLE` | Godwin | str | Article title (missing for 14 rows) |
| `AUTHORS` / `Authors` | Godwin / OpenAlex | str | Semicolon-separated author list; the OpenAlex one is complete |
| `NumAuthors` | OpenAlex | float | Number of authors |
| `HasUSAuthor` | OpenAlex | bool | Any author with a U.S. affiliation |
| `YEAR_PUBLISHED` | Godwin | float | Publication year as coded by Godwin et al.; **the year the sample was filtered on** |
| `PublicationDate` | OpenAlex | datetime (UTC) | Publication date, online-first |
| `PublicationYear` | OpenAlex | float | Year of `PublicationDate`; 2016 for ten online-first articles |
| `TotalCitations` | OpenAlex | float | Citations as of that row's `LastUpdate` |
| `Citations2016` ... `Citations2026` | OpenAlex | float | Citations received in each calendar year. `Citations2013`-`Citations2015` exist but are empty, and `Citations2026` is a partial year |
| `FieldWeightedCitationIndex` | OpenAlex | float | FWCI: citations relative to the field and year average, where 1.0 is average. Exactly 0 for five articles |
| `LastUpdate` | OpenAlex | datetime (UTC) | When this row was fetched |
| `Pub2UpdateTime` | OpenAlex | timedelta | `LastUpdate` minus `PublicationDate`, i.e. exposure time |
| `IsOpenAccess` | OpenAlex | bool | OpenAlex's `is_oa`: true if **any** free-to-read version exists, including a repository copy |
| `HasPreprint` | OpenAlex + CrossRef | bool | A matched preprint, found by a three-stage search (OpenAlex, CrossRef, repository). 47 articles (20.1%) |
| `PreprintSources` | OpenAlex + CrossRef | str | Which stage found the preprint, and where |
| `VenueName` / `VenueID` | OpenAlex | str | Journal, 23 distinct venues |
| `Venue2yrMeanCitedness` | OpenAlex | float | Two-year mean citedness, used as a journal-impact proxy. **A current value, not the value at publication** |
| `VenueHIndex` / `VenueI10Index` | OpenAlex | float | Other journal-level indicators |
| `Topics` | OpenAlex | str | OpenAlex topic IDs with scores, as a stringified dict |
| `CLAIMED_TO_SHARE` | Godwin | str | `YES` / `NO`: did the article say data were available |
| `ACTUALLY_SHARED` | Godwin | str | `YES` / `NO`: could the data be found |
| `BY_FIXATION` / `BY_TRIAL` / `BY_PPT` | Godwin | str | `YES` / `NO` per granularity of shared data |
| `CODE` / `MATERIALS` / `EXPERIMENT` / `CODEBOOK_GUIDE` | Godwin | str | `YES` / `NO` for other shared research outputs |
| `SHARING_LINK` / `SHARING_LOCATION` / `OSF_ID` | Godwin | str | Where the data were found (OSF, GitHub, ...) |
| `data_sharing_class` | derived | str | `NONE`, `PARTICIPANT`, `TRIAL` or `FIXATION`, at the **finest** granularity shared |
| `is_sharing_data` | derived | bool | `data_sharing_class != "NONE"`: shared eye-movement data at some granularity |
| `is_sharing_anything` | derived | bool | Shared **any** research output, including code or materials but no data |

Note that `is_sharing_data` and `is_sharing_anything` ask different questions: an article that released only
analysis code is `is_sharing_anything = True` but `is_sharing_data = False`.

## Loading it

The file is Parquet, so you need `pandas` with `pyarrow` (`pip install pandas pyarrow`).

```python
import pandas as pd

articles = pd.read_parquet("combined.parquet")
print(articles.shape)                       # (234, 81)

# the index is a positional id from the source spreadsheet and is not contiguous;
# use the DOI as the key instead
articles = articles.set_index("DOI")

# citations by sharing granularity
print(
    articles
    .groupby("data_sharing_class")["TotalCitations"]
    .agg(["count", "median", "mean"])
)

# citations per calendar year, long format, one row per observed article-year.
# dropna() matters: the year columns are a rectangle, so an article that did not
# exist in a given year has a NaN there rather than a zero
yearly = (
    articles
    .filter(regex=r"^Citations\d{4}$")
    .rename(columns=lambda column: int(column.removeprefix("Citations")))
    .stack()
    .rename("citations")
    .rename_axis(["DOI", "year"])
    .dropna()
    .reset_index()
)
print(yearly.shape)                         # (1387, 3)
```

Four things that will otherwise surprise you. The index is the row's position in the source spreadsheet, is
not contiguous, and means nothing: key on `DOI`. Godwin et al.'s `YES` / `NO` columns are **strings**, not
booleans, and are empty where the question did not apply, so compare with `== "YES"` rather than relying on
truthiness. `Error` is an empty column left over from the fetch, which recorded OpenAlex lookup failures;
those rows were excluded, so it is null throughout. And the per-year columns do not quite sum to
`TotalCitations`: eight articles are short by exactly one citation (5,119 against 5,127 across the corpus),
because OpenAlex's own `cited_by_count` and its year-by-year breakdown are compiled separately. Use
`TotalCitations` for a total up to the `LastUpdate` date, and the year columns for anything about timing, but do not expect them to
reconcile exactly.
