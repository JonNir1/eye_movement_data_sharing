import os
from typing import Optional

import re
import pandas as pd

from fetch_metadata import fetch_all_metadata, DOI_PATTERN

_SHARING_CORRECTIONS = {
    # articles with duplicate records in the original Godwin dataset are re-classified manually
    "10.1038/s41598-018-37548-w": "FIXATION",
    "10.3758/s13414-018-1579-7":  "FIXATION",
    "10.3758/s13414-017-1328-3": "FIXATION",
    "10.3758/s13414-017-1354-1": "FIXATION",
    "10.1177/1747021820919351": "FIXATION",
    "10.3758/s13414-021-02336-8": "TRIAL",
    "10.3758/s13423-021-01920-1": "TRIAL",
    "10.3758/s13423-021-01944-7": "PARTICIPANT",
}


def prepare_analytical_dataset(
        godwin_path: str, metadata_path: str, apply_corrections: bool = True
) -> pd.DataFrame:
    godwin_dataset = load_godwin2025(godwin_path)
    metadata = _load_or_fetch_metadata(metadata_path, godwin_dataset)
    merged = (
        pd.concat([metadata, godwin_dataset], axis=1)
        .assign(
            is_sharing_data=lambda df: df["data_sharing_class"] != "NONE",
            is_sharing_anything=lambda df: (df[[
                "EXPERIMENT", "MATERIALS", "CODE", "CODEBOOK_GUIDE", "BY_FIXATION", "BY_TRIAL", "BY_PPT"
            ]] == "YES").any(axis=1)
        )
    )
    if apply_corrections:
        merged = _correct_sharing_class(merged)
    return merged


def load_godwin2025(path: str = None) -> pd.DataFrame:
    if not path:
        path = os.path.join(os.getcwd(), "Godwin_2025_dataset.xlsx")
    if not (path.endswith(".xlsx") or path.endswith(".xls")):
        raise ValueError("The database file must be an Excel file with .xlsx or .xls extension.")
    if not (os.path.exists(path) or os.path.isfile(path)):
        raise FileNotFoundError(f"The specified database file does not exist: {path}")
    dataset = (
        pd.read_excel(path)
        .assign(missing_link_and_title=lambda df: df["PAPER_LINK"].isna() & df["PAPER_TITLE"].isna())
        .loc[lambda df: ~df["missing_link_and_title"]]
        .drop(columns=["missing_link_and_title"])
        .drop_duplicates(subset=["PAPER_LINK", "PAPER_TITLE"])
        .assign(data_sharing_class=lambda df: _assign_eyemovement_sharing_class(df))
    )
    dataset = dataset[dataset["PAPER_LINK"] != "UNPUBLISHED"]
    return dataset


def _assign_eyemovement_sharing_class(df: pd.DataFrame) -> pd.Series:
    sharing_class = pd.Series("NONE", index=df.index, dtype="string").rename("data_sharing_class")
    # specify data from coarse to finest to overwrite when article shares more fine-grained data
    sharing_class[df["BY_PPT"] == "YES"] = "PARTICIPANT"
    sharing_class[df["BY_TRIAL"] == "YES"] = "TRIAL"
    sharing_class[df["BY_FIXATION"] == "YES"] = "FIXATION"
    return sharing_class


def _load_or_fetch_metadata(path: str, godwin_dataset: pd.DataFrame) -> pd.DataFrame:
    try:
        metadata = pd.read_csv(os.path.join(path), index_col=0)
        print("Loaded OpenAlex metadata from CSV.")
    except FileNotFoundError:
        print("Fetching metadata from OpenAlex. This may take a few minutes...")
        metadata = fetch_all_metadata(godwin_dataset, sleep_period=0.01, verbose=True)
        metadata.to_csv(path, index=True)
    metadata["Pub2UpdateTime"] = pd.to_timedelta(metadata["Pub2UpdateTime"])    # validate casting
    metadata["LastUpdate"] = pd.to_datetime(metadata["LastUpdate"], utc=True)   # validate casting
    return metadata


def _correct_sharing_class(df: pd.DataFrame) -> pd.DataFrame:
    def bare_doi(doi: Optional[str]) -> Optional[str]:
        if pd.isnull(doi):
            return doi
        match = re.search(DOI_PATTERN, doi)
        return match.group(1).lower() if match else None

    sanitized_dois = df["DOI"].map(bare_doi)
    missing = set(_SHARING_CORRECTIONS) - set(sanitized_dois.dropna())
    if missing:
        raise ValueError(f"Correction DOIs not found in the dataset: {sorted(missing)}")
    hand_picked_class = sanitized_dois.map(_SHARING_CORRECTIONS)
    # apply the re-classification:
    new_df = df.copy()      # avoid SettingWithCopy warnings
    new_df["data_sharing_class"] = hand_picked_class.fillna(df["data_sharing_class"])
    return new_df
