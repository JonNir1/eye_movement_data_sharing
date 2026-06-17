import re
import time
from typing import Optional

import requests
import pyalex
import pandas as pd
from pyalex.api import QueryError
from tqdm import tqdm

from _api_secrets import *


pyalex.config.email = EMAIL
pyalex.config.api_key = OPENALEX_API_KEY
DOI_PATTERN = r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)'


def fetch_all_metadata(
        articles: pd.DataFrame,
        link_column: str = "PAPER_LINK",
        title_column: str = "PAPER_TITLE",
        sleep_period=0.01,
        verbose=True,
) -> pd.DataFrame:
    if sleep_period < 0:
        raise ValueError("sleep_period must be non-negative.")
    results = []
    venue_cache = {}        # reuse Source-level summary_stats across articles from the same venue
    for _, article_row in tqdm(list(articles.iterrows()), disable=not verbose):
        link = __coerce_string(article_row, link_column)
        title = __coerce_string(article_row, title_column)
        metadata = fetch_single_metadata(link, title, article_row.name, venue_cache=venue_cache, verbose=verbose)
        results.append(metadata)
        time.sleep(sleep_period)     # to respect rate limits of 100 requests per second
    results = (
        pd.DataFrame(results)
        .assign(
            LastUpdate=lambda df: pd.to_datetime(df["LastUpdate"], utc=True),
            PublicationDate=lambda df: pd.to_datetime(df["PublicationDate"], utc=True),
            Pub2UpdateTime=lambda df: df["LastUpdate"] - df["PublicationDate"],
            # MeanNormalizedCitationScore=_calculate_mncs,    # bad calculation; see function docstring
        )
        .astype({"idx": articles.index.dtype})
        .set_index("idx")
    )
    results.index.name = articles.index.name
    return results


def fetch_single_metadata(link: str, title: str, idx, venue_cache: Optional[dict] = None, verbose=True) -> dict:
    venue_cache = venue_cache if venue_cache is not None else {}
    result = dict()
    try:
        work = _fetch_work_by_doi_unsafe(link)
        if not work:
            work = _fetch_work_by_url_unsafe(link)
        if not work:
            work = _fetch_work_by_title_unsafe(title)
        if not work:
            return _return_on_error(link, idx, None, verbose)
        result["idx"] = idx
        result["DOI"] = work.get("doi")
        result["OpenAlexID"] = work.get("id").split("/")[-1]
        result["LastUpdate"] = work.get("updated_date")
        result["PublicationType"] = work.get("type")
        result["PublicationYear"] = work.get("publication_year")
        result["PublicationDate"] = work.get("publication_date")
        result["Topics"] = {
            topic["id"].split("/")[-1]: topic["score"] for topic in work.get("topics", [])
        }
        result["FieldWeightedCitationIndex"] = work.get("fwci", pd.NA)
        result["IsRetracted"] = work.get("is_retracted")
        result["TotalCitations"] = work.get("cited_by_count")
        result.update(_extract_covariates(work, venue_cache))
        for yr in work.get("counts_by_year", []):
            result[f"Citations{yr['year']}"] = yr["cited_by_count"]
        return result
    except Exception as e:
        return _return_on_error(link, idx, e, verbose)


_COVARIATE_COLUMNS = ["NumAuthors", "HasUSAuthor", "IsOpenAccess", "HasPreprint", "PreprintSources",
                      "VenueID", "VenueName", "Venue2yrMeanCitedness", "VenueHIndex", "VenueI10Index"]

# Substrings (matched against a repository source's display_name) that mark a genuine preprint server.
# Deliberately excludes green-OA repositories (PubMed Central, institutional repos, DOAJ, Zenodo, Figshare).
# "rxiv" covers arXiv / bioRxiv / medRxiv / PsyArXiv / SocArXiv / ChemRxiv / TechRxiv / EngrXiv.
_PREPRINT_SERVER_KEYWORDS = ("rxiv", "preprint", "research square", "ssrn")


def fetch_covariates_by_id(
        openalex_ids: pd.Series, sleep_period: float = 0.01, verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch author/open-access/venue covariates for already-known OpenAlex work IDs, returned as a
    DataFrame indexed like ``openalex_ids``. Keying off the cached id (rather than re-matching by
    link/title) guarantees the covariates describe the same work as previously-fetched citation data.
    """
    venue_cache = {}
    records = []
    for openalex_id in tqdm(openalex_ids, disable=not verbose):
        if pd.isna(openalex_id):
            records.append({})
            continue
        try:
            work = pyalex.Works()[openalex_id]
            records.append(_extract_covariates(work, venue_cache))
        except (requests.exceptions.RequestException, QueryError) as e:
            if verbose:
                print(f"Error fetching covariates for {openalex_id}: {e}")
            records.append({})
        time.sleep(sleep_period)     # to respect rate limits of 100 requests per second
    return pd.DataFrame(records, index=openalex_ids.index).reindex(columns=_COVARIATE_COLUMNS)


def _extract_covariates(work: dict, venue_cache: dict) -> dict:
    return {
        "NumAuthors": len(work.get("authorships", [])),
        "HasUSAuthor": _has_us_author(work),
        "IsOpenAccess": (work.get("open_access") or {}).get("is_oa"),
        **_detect_preprint(work),
        **_fetch_venue_summary_stats(work, venue_cache),
    }


def _detect_preprint(work: dict) -> dict:
    """
    Flag whether a work was ever preprinted, unioning two independent (and individually incomplete)
    signals: an OpenAlex preprint-server location and a Crossref ``has-preprint`` relation. The two
    barely overlap, so the union materially reduces the undercount. PreprintSources records which
    signal(s) fired, for auditing.
    """
    openalex_server = _openalex_preprint_server(work)
    crossref_preprint = _crossref_has_preprint(work.get("doi"))
    sources = []
    if openalex_server:
        sources.append(f"openalex:{openalex_server}")
    if crossref_preprint:
        sources.append("crossref")
    return {
        "HasPreprint": bool(openalex_server) or crossref_preprint,
        "PreprintSources": ";".join(sources) if sources else pd.NA,
    }


def _openalex_preprint_server(work: dict) -> Optional[str]:
    """Return the display name of a recognised preprint server among the work's repository
    locations (excluding green-OA repositories), or None."""
    for location in work.get("locations", []) or []:
        source = location.get("source") or {}
        if source.get("type") != "repository":
            continue
        name = source.get("display_name") or ""
        if any(keyword in name.lower() for keyword in _PREPRINT_SERVER_KEYWORDS):
            return name
    return None


def _crossref_has_preprint(doi: Optional[str]) -> bool:
    """Whether Crossref records a ``has-preprint`` relation for this DOI (publisher-deposited)."""
    if not doi:
        return False
    doi_match = re.search(DOI_PATTERN, doi)
    if not doi_match:
        return False
    try:
        resp = requests.get(f"https://api.crossref.org/works/{doi_match.group(1)}",
                            params={"mailto": EMAIL}, timeout=20)
        if resp.status_code != 200:
            return False
        relation = resp.json().get("message", {}).get("relation", {}) or {}
        return bool(relation.get("has-preprint"))
    except (requests.exceptions.RequestException, ValueError):
        return False


def _has_us_author(work: dict) -> bool:
    for authorship in work.get("authorships", []):
        if any(inst.get("country_code") == "US" for inst in authorship.get("institutions", [])):
            return True
        if "US" in (authorship.get("countries") or []):     # fallback when institutions are unresolved
            return True
    return False


def _fetch_venue_summary_stats(work: dict, venue_cache: dict) -> dict:
    empty = {"VenueID": pd.NA, "VenueName": pd.NA,
             "Venue2yrMeanCitedness": pd.NA, "VenueHIndex": pd.NA, "VenueI10Index": pd.NA}
    source = (work.get("primary_location") or {}).get("source") or {}
    venue_url = source.get("id")
    if not venue_url:
        return empty
    venue_id = venue_url.split("/")[-1]
    if venue_id not in venue_cache:
        # the work object carries the source id/name but not its citation summary; fetch it once per venue
        try:
            stats = pyalex.Sources()[venue_id].get("summary_stats") or {}
        except (requests.exceptions.RequestException, QueryError, KeyError):
            stats = {}
        venue_cache[venue_id] = {
            "VenueID": venue_id,
            "VenueName": source.get("display_name"),
            "Venue2yrMeanCitedness": stats.get("2yr_mean_citedness", pd.NA),
            "VenueHIndex": stats.get("h_index", pd.NA),
            "VenueI10Index": stats.get("i10_index", pd.NA),
        }
    return venue_cache[venue_id]


def _fetch_work_by_doi_unsafe(text_with_doi: str) -> Optional[dict]:
    doi_match = re.search(DOI_PATTERN, text_with_doi)
    if doi_match:
        found_doi = doi_match.group(1)
        work = pyalex.Works()[f"doi:{found_doi}"]
        return work
    return None


def _fetch_work_by_url_unsafe(article_link: str) -> Optional[dict]:
    found_works = pyalex.Works().filter(locations={"landing_page_url": article_link}).get()
    if found_works:
        # OpanAlex .get() returns a list when using filters
        return found_works[0]
    # attempt to fetch URL's content and extract DOI
    if not article_link.startswith("http"):
        raise ValueError("The provided link is not a valid URL.")
    resp = requests.get(article_link, timeout=10)
    if not resp.ok:
        raise ConnectionError(f"Failed to fetch URL content: {article_link}")
    return _fetch_work_by_doi_unsafe(resp.text)


def _fetch_work_by_title_unsafe(title: str) -> Optional[dict]:
    title = title.strip().lower()
    found_works = pyalex.Works().search_filter(title=title).get()
    found_works = [work for work in found_works if work.get("title", "").strip().lower() == title]
    if len(found_works) > 1:
        raise RuntimeError(f"Multiple works found with the exact same title: {title}")
    if found_works:
        return found_works[0]
    return None


def _return_on_error(
        link: str,
        idx,
        error: Optional[Exception] = None,
        verbose: bool = True
) -> dict:
    error = error or "Unknown Error"
    if verbose:
        print(f"Error fetching metadata for link {link}: {error}")
    return {"idx": idx, "Error": str(error)}


def _calculate_mncs(publications: pd.DataFrame) -> pd.Series:
    """
    NOTE: to correctly calculate MNCS, field normalization is also required. This implementation only normalizes
    within the small dataset provided, which may lead to misleading results if the dataset is not representative
    of the broader field.
    Hence, we decided to keep this function here for demonstration purposes, but not use it in the main analysis.

    The Mean Normalized Citation Score (MNCS) is the industry-standard metric for citation impact after normalizing for
    field and publication time (see https://open.leidenranking.com/information/indicators). It is calculated as the
    ratio between the actual number of citations received by a publication and the expected number of citations for
    publications of the same field and publication year.

    :param publications: pd.DataFrame with at least the following columns:
        - 'TotalCitations': int; total number of citations received by the publication
        - 'PublicationDate': int; UTC date of publication, in ISO8601 format (https://en.wikipedia.org/wiki/ISO_8601)
        - 'IsRetracted': bool; whether the publication is retracted
    :return: pd.Series[float] with the MNCS values for each publication, NaN for retracted publications
    """
    non_retracted = publications[publications["IsRetracted"] == False]   # hard equality to ignore NaNs
    non_retracted.loc[:, "PublicationYear"] = pd.to_datetime(non_retracted["PublicationDate"]).dt.year
    year_mean_citations = non_retracted.groupby(["PublicationYear"])["TotalCitations"].transform('mean')
    mncs = (non_retracted["TotalCitations"] / year_mean_citations).rename("MeanNormalizedCitationScore")
    mncs = mncs.reindex(publications.index, fill_value=pd.NA)
    return mncs


def __coerce_string(row: pd.Series, col_name: str) -> str:
    val = row.get(col_name, "")
    if pd.isna(val):
        return ""
    return str(val).strip()
