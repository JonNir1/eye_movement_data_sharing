# Eye-Movement Data-Sharing Corpus
A dataset of **234 visual-search eye-tracking articles** published between 2017 and 2022, each coded for what
research data its authors shared, and joined to bibliometric indicators from
[OpenAlex](https://openalex.org).

It exists to answer one question: do articles that share their data get cited more? The sharing coding comes
from the open-science audit of

> Godwin, H. J., et al. (2025). *Behavior Research Methods*, 57(9), 235.
> <https://doi.org/10.3758/s13428-025-02759-3>

and the citation indicators, the preprint detection and the analytic sample were added for the accompanying
report, *Data Sharing and Citation Impact in Visual Search Research* (Behavior Research Methods *under review*;
preprint at <https://osf.io/preprints/psyarxiv/q8jrd>).<br>
If you use this file, please cite both: Godwin et al. for the sharing coding, and the report for the joined dataset.

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

## Data Integrity
Two known problems in the data, recorded here so that anyone reusing the corpus can see what was
removed and what was lost.

### Duplicate Records
Eight articles appear **twice** in Godwin et al.'s spreadsheet, once keyed by a bare DOI with a title, and
once by a publisher URL with the title missing. Because both the link and the title differ between the two
copies, de-duplicating on `(PAPER_LINK, PAPER_TITLE)` does not catch them; they collapse only once OpenAlex
resolves both forms to the same DOI, which is why the exclusion cascade reports them as "duplicate DOIs".

**Five of the eight pairs carry conflicting sharing codes**, one copy recording a granularity and the other
recording none. A naive keep-first de-duplication would therefore change an article's classification
depending on row order, so `prepare_data._SHARING_CORRECTIONS` pins the class for all eight by DOI.

Row numbers are 0-based positions in the source sheet; add 2 for the Excel row number.

| Kept Row | Duplicate Row | DOI                        | Title                                                                                                                                           | Authors                                                              | Sharing codes             |
|----------|---------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|---------------------------|
| 10       | 86            | 10.3758/s13414-018-1579-7  | Lack of free choice reveals the cost of multiple-target search within and across feature dimensions                                             | Ort, E; Fahrenfort, JJ; Olivers, CNL                                 | FIXATION / FIXATION       |
| 21       | 82            | 10.3758/s13423-021-01944-7 | Understanding the collinear masking effect in visual search through eye tracking                                                                | Hsiao, JH; Chan, AB; An, J; Yeh, S; Jingling, L                      | PARTICIPANT / PARTICIPANT |
| 75       | 84            | 10.1038/s41598-018-37548-w | Searchers adjust their eye-movement dynamics to target characteristics in natural scenes                                                        | Rothkegel, LOM; Schütt, HH; Trukenbrod, HA; Wichmann, FA; Engbert, R | FIXATION / FIXATION       |
| 83       | 109           | 10.3758/s13414-017-1328-3  | Comparing visual search and eye movements in bilinguals and monolinguals                                                                        | Ratiu, I; Hout, MC; Walenchok, SC; Azuma, T; Goldinger, SD           | FIXATION / NONE           |
| 88       | 216           | 10.3758/s13414-017-1354-1  | Categorical templates are more useful when features are consistent: Evidence from eye movements during search for societally important vehicles | Hout, MC; Robbins, A; Godwin, HJ; Fitzsimmons, G; Scarince, C        | FIXATION / NONE           |
| 98       | 358           | 10.1177/1747021820919351   | Dynamics of attentional and oculomotor orienting in visual foraging tasks                                                                       | Tagu, J; Kristjánsson, Á                                             | FIXATION / NONE           |
| 99       | 295           | 10.3758/s13423-021-01920-1 | Eye-movements reveal semantic interference effects during the encoding of naturalistic scenes in long-term memory                               | Mikhailova, A; Raposo, A; Della Sala, S; Coco, MI                    | TRIAL / NONE              |
| 104      | 217           | 10.3758/s13414-021-02336-8 | Comparing imagery and perception: Using eye movements to dissociate mechanisms in search                                                        | Cochrane, BA; Wang, C; Pratt, J; Milliken, B; Sun, H                 | TRIAL / NONE              |

All eight surviving copies are in the N = 234 sample. My article states *6 articles* were duplicates, because two of
the eight were already filtered out based on year or topic, but the table above shows all eight.

### Incorrect Records
Eleven records passed the topic and year filters but could not be matched to OpenAlex automatically, because Godwin
et al.'s sheet gave **no title and no authors, only a broken link** (leading to a 404), so they were initially dropped
from analysis.<br>
I manually recovered the DOIs and titles for these eleven, but they remain excluded from the $N=234$ sample, and this
table exists so that the loss is visible and the records are recoverable.

| Row | Broken Link                                     | Correct DOI                     | Title                                                                                                                          | Godwin class | Why the lookup failed                                                                     |
|-----|-------------------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------|-------------------------------------------------------------------------------------------|
| 1   | jov.arvojournals.org/…articleid=2770293         | 10.1167/jov.20.7.8              | How spatial frequencies and color drive object search in real-world scenes: A new eye-movement corpus                          | FIXATION     | publisher blocked the URL (jov.arvojournals.org)                                          |
| 79  | jov.arvojournals.org/…articleid=2657481         | 10.1167/17.12.8                 | To search or to like: Mapping fixations to differentiate two forms of incidental scene memory                                  | TRIAL        | publisher blocked the URL (jov.arvojournals.org)                                          |
| 85  | jov.arvojournals.org/…articleid=2718934         | 10.1167/18.13.11                | Anchoring visual search in scenes: Assessing the role of anchor objects on eye movements during visual search                  | FIXATION     | publisher blocked the URL (jov.arvojournals.org)                                          |
| 89  | 10.1037/xap0000235.                             | 10.1037/xap0000235              | Correction to Muhl-Richardson et al. (2018)                                                                                    | FIXATION     | OpenAlex 404: the sheet's DOI carries a **trailing period**, which `DOI_PATTERN` captures |
| 91  | mdpi.com/2076-3425/11/3/283                     | 10.3390/brainsci11030283        | Saliency-Aware Subtle Augmentation Improves Human Visual Search Performance in VR                                              | FIXATION     | publisher blocked the URL (mdpi.com)                                                      |
| 92  | psycnet.apa.org/record/2019-10303-001           | 10.1037/xhp0000618              | Expectations and perceptual priming in a visual search task: Evidence from eye movements and behavior                          | FIXATION     | publisher blocked the URL (psycnet.apa.org)                                               |
| 93  | sciencedirect.com/…S2451958821000750            | 10.1016/j.chbr.2021.100127      | Assessing how visual search entropy and engagement predict performance in a multiple-objects tracking air traffic control task | TRIAL        | publisher blocked the URL (sciencedirect.com)                                             |
| 94  | sciencedirect.com/…S0010945221000149            | 10.1016/j.cortex.2020.12.019    | Spatial attention and spatial short term memory in PSP and Parkinson's disease                                                 | NONE         | publisher blocked the URL (sciencedirect.com)                                             |
| 100 | sciencedirect.com/…S0010027721003589            | 10.1016/j.cognition.2021.104935 | The selection balance: Contrasting value, proximity and priming in a multitarget foraging task                                 | FIXATION     | publisher blocked the URL (sciencedirect.com)                                             |
| 101 | sciencedirect.com/…S0010027719303233            | 10.1016/j.cognition.2019.104149 | Active sampling in visual search is coupled to the cardiac cycle                                                               | FIXATION     | publisher blocked the URL (sciencedirect.com)                                             |
| 102 | biorxiv.org/content/10.1101/2021.02.05.429946v1 | 10.1101/2021.02.05.429946       | Pupil Size Reflects Exploration and Exploitation in Visual Search (and It's Like Object-Based Attention)                       | FIXATION     | OpenAlex 404: the link carries the **version suffix** `v1`                                |

Two of the eleven need more than a corrected DOI:

- **10.1037/xap0000235 is a correction notice, not primary research.** Its sharing codes describe the study
  being corrected, **10.1037/xap0000155** (*Individual differences in search and monitoring for color targets
  in dynamic visual displays*), which is itself absent from Godwin et al.'s corpus. If this record is ever
  recovered, **the correction notice should be replaced by its underlying study**.
- **10.1101/2021.02.05.429946 is a bioRxiv preprint**, not a journal article, and its `YEAR_PUBLISHED` (2019)
  conflicts with the date encoded in its DOI (2021-02-05). It is **excluded from further analyses**.

The loss is not neutral with respect to data sharing. Godwin et al. audited all eleven and found an OSF link
for every one: **ten shared eye-movement data** (eight at fixation level, two at trial level), and the
eleventh shared materials, an experiment and a codebook but no data. Recovering the nine that are genuine
journal articles, i.e. excluding the correction notice and the preprint, would take the sample from 234 to
243 articles, the sharing group from 80 to 88, and the fixation class from 31 to 37.


## Where each column comes from
Three naming conventions, each marking a different origin:

| Convention   | Origin                                 | Count | Examples                                            |
|--------------|----------------------------------------|-------|-----------------------------------------------------|
| `ALL_CAPS`   | Godwin et al.'s hand-coded spreadsheet | 41    | `BY_FIXATION`, `CLAIMED_TO_SHARE`, `YEAR_PUBLISHED` |
| `CamelCase`  | OpenAlex, fetched for this study       | 37    | `TotalCitations`, `VenueName`, `PublicationDate`    |
| `snake_case` | Derived here from the columns above    | 3     | `data_sharing_class`, `is_sharing_data`             |

**One exception to the rule: `DOI` is `ALL_CAPS` but comes from OpenAlex**, not from Godwin et al. Their
spreadsheet identifies articles by `PAPER_LINK` instead. Every other `ALL_CAPS` column is theirs.

The two sources also disagree in coverage in one place worth noting: `PAPER_TITLE` and `AUTHORS` (Godwin et
al.) are missing for 14 articles, whereas OpenAlex's `Authors` is complete for all 234. Prefer the OpenAlex
columns for anything that has to be populated for every row.

## Main columns
The full 81 are mostly Godwin et al.'s fine-grained sharing codes (data formats, open versus closed formats,
search attempts). These are the ones most analyses need.

| Column                                                 | Source              | Type           | Description                                                                                                                          |
|--------------------------------------------------------|---------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `DOI`                                                  | OpenAlex            | str            | Resolvable DOI URL. **The recommended key** for joining to other sources                                                             |
| `OpenAlexID`                                           | OpenAlex            | str            | OpenAlex work ID, e.g. `W2982222209`                                                                                                 |
| `PAPER_LINK`                                           | Godwin              | str            | The bare DOI or URL as recorded in the original sheet                                                                                |
| `PAPER_TITLE`                                          | Godwin              | str            | Article title (missing for 14 rows)                                                                                                  |
| `AUTHORS` / `Authors`                                  | Godwin / OpenAlex   | str            | Semicolon-separated author list; the OpenAlex one is complete                                                                        |
| `NumAuthors`                                           | OpenAlex            | float          | Number of authors                                                                                                                    |
| `HasUSAuthor`                                          | OpenAlex            | bool           | Any author with a U.S. affiliation                                                                                                   |
| `YEAR_PUBLISHED`                                       | Godwin              | float          | Publication year as coded by Godwin et al.; **the year the sample was filtered on**                                                  |
| `PublicationDate`                                      | OpenAlex            | datetime (UTC) | Publication date, online-first                                                                                                       |
| `PublicationYear`                                      | OpenAlex            | float          | Year of `PublicationDate`; 2016 for ten online-first articles                                                                        |
| `TotalCitations`                                       | OpenAlex            | float          | Citations as of that row's `LastUpdate`                                                                                              |
| `Citations2016` ... `Citations2026`                    | OpenAlex            | float          | Citations received in each calendar year. `Citations2013`-`Citations2015` exist but are empty, and `Citations2026` is a partial year |
| `FieldWeightedCitationIndex`                           | OpenAlex            | float          | FWCI: citations relative to the field and year average, where 1.0 is average. Exactly 0 for five articles                            |
| `LastUpdate`                                           | OpenAlex            | datetime (UTC) | When this row was fetched                                                                                                            |
| `Pub2UpdateTime`                                       | OpenAlex            | timedelta      | `LastUpdate` minus `PublicationDate`, i.e. exposure time                                                                             |
| `IsOpenAccess`                                         | OpenAlex            | bool           | OpenAlex's `is_oa`: true if **any** free-to-read version exists, including a repository copy                                         |
| `HasPreprint`                                          | OpenAlex + CrossRef | bool           | A matched preprint, found by a three-stage search (OpenAlex, CrossRef, repository). 47 articles (20.1%)                              |
| `PreprintSources`                                      | OpenAlex + CrossRef | str            | Which stage found the preprint, and where                                                                                            |
| `VenueName` / `VenueID`                                | OpenAlex            | str            | Journal, 23 distinct venues                                                                                                          |
| `Venue2yrMeanCitedness`                                | OpenAlex            | float          | Two-year mean citedness, used as a journal-impact proxy. **A current value, not the value at publication**                           |
| `VenueHIndex` / `VenueI10Index`                        | OpenAlex            | float          | Other journal-level indicators                                                                                                       |
| `Topics`                                               | OpenAlex            | str            | OpenAlex topic IDs with scores, as a stringified dict                                                                                |
| `CLAIMED_TO_SHARE`                                     | Godwin              | str            | `YES` / `NO`: did the article say data were available                                                                                |
| `ACTUALLY_SHARED`                                      | Godwin              | str            | `YES` / `NO`: could the data be found                                                                                                |
| `BY_FIXATION` / `BY_TRIAL` / `BY_PPT`                  | Godwin              | str            | `YES` / `NO` per granularity of shared data                                                                                          |
| `CODE` / `MATERIALS` / `EXPERIMENT` / `CODEBOOK_GUIDE` | Godwin              | str            | `YES` / `NO` for other shared research outputs                                                                                       |
| `SHARING_LINK` / `SHARING_LOCATION` / `OSF_ID`         | Godwin              | str            | Where the data were found (OSF, GitHub, ...)                                                                                         |
| `data_sharing_class`                                   | derived             | str            | `NONE`, `PARTICIPANT`, `TRIAL` or `FIXATION`, at the **finest** granularity shared                                                   |
| `is_sharing_data`                                      | derived             | bool           | `data_sharing_class != "NONE"`: shared eye-movement data at some granularity                                                         |
| `is_sharing_anything`                                  | derived             | bool           | Shared **any** research output, including code or materials but no data                                                              |

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
