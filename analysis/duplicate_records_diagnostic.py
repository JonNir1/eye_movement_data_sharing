"""
Diagnostic for the duplicate-records issue raised in review (Reviewer 1, Method comment).

Background: the draft reports that, of the queried records, a handful "returned duplicate DOIs".
This script identifies those duplicates and explains *why* they are not caught by the dedup in
``load_data.load_godwin2025`` (which drops duplicates on the ``(PAPER_LINK, PAPER_TITLE)`` tuple).

Finding: each duplicated DOI corresponds to the *same* article entered twice in the source
dataset - once keyed by a bare DOI (with a ``PAPER_TITLE``) and once by a publisher URL (with
``PAPER_TITLE`` missing). Because both the link and the title differ between the two copies, the
tuple-based dedup does not remove them; they only collapse after OpenAlex canonicalises both forms
to the same DOI. Some duplicate pairs additionally carry conflicting ``data_sharing_class`` codes,
so a naive "keep-first" dedup can change an article's sharing classification.

Usage:
    python duplicate_records_diagnostic.py
(expects ``Godwin_2025_dataset.xlsx`` and ``Godwin_2025_metadata.csv`` in the working directory).
"""
import os
from typing import Optional

import pandas as pd

from prepare_data import load_godwin2025


def find_duplicate_doi_records(
        dataset_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the source rows that share a DOI with at least one other row, annotated with the
    matched DOI/OpenAlexID and the source link/title/sharing-class, so the duplication mechanism
    is visible. Detection is post-resolution: duplicates are defined by a shared (normalised) DOI
    in the fetched metadata, not by the source ``(PAPER_LINK, PAPER_TITLE)`` tuple.
    """
    if not metadata_path:
        metadata_path = os.path.join(os.getcwd(), "Godwin_2025_metadata.csv")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata cache not found: {metadata_path}")

    dataset = load_godwin2025(dataset_path)
    metadata = (
        pd.read_csv(metadata_path)
        .rename(columns={"Unnamed: 0": "idx"})
        .set_index("idx")
    )
    metadata = metadata.loc[metadata.index.intersection(dataset.index)]

    matched = metadata.loc[metadata["DOI"].notna()].copy()
    matched["doi_norm"] = matched["DOI"].str.lower().str.strip()
    duplicated = matched.loc[matched["doi_norm"].duplicated(keep=False)]

    report = (
        duplicated
        .join(dataset[["PAPER_LINK", "PAPER_TITLE", "data_sharing_class"]])
        .assign(link_is_bare_doi=lambda df: ~df["PAPER_LINK"].astype(str).str.startswith("http"),
                title_missing=lambda df: df["PAPER_TITLE"].isna())
        .sort_values(["doi_norm"])
        [["doi_norm", "OpenAlexID", "PAPER_LINK", "PAPER_TITLE",
          "data_sharing_class", "link_is_bare_doi", "title_missing"]]
    )
    return report


def _print_summary(report: pd.DataFrame) -> None:
    n_doi = report["doi_norm"].nunique()
    n_rows = len(report)
    conflicting = (
        report.groupby("doi_norm")["data_sharing_class"].nunique().gt(1).sum()
    )
    print(f"Duplicated DOIs: {n_doi}  (spanning {n_rows} source rows; "
          f"{n_rows - n_doi} rows are removable duplicates)")
    print(f"Duplicate pairs with conflicting data_sharing_class: {conflicting}\n")
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 200)
    for doi, group in report.groupby("doi_norm"):
        print(f"DOI: {doi}  ->  OpenAlexID(s): {list(group['OpenAlexID'].unique())}")
        for idx, row in group.iterrows():
            print(f"   idx={idx} | class={row.data_sharing_class} | "
                  f"bare_doi_link={row.link_is_bare_doi} title_missing={row.title_missing} | "
                  f"title={str(row.PAPER_TITLE)[:60]!r}")
        print("-" * 100)


if __name__ == "__main__":
    _print_summary(find_duplicate_doi_records())
