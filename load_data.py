import os

import pandas as pd

from fetch_metadata import fetch_all_metadata


def prepare_analytical_dataset(godwin_path: str, metadata_path: str) -> pd.DataFrame:
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
    sharing_class[df["BY_PPT"] == "YES"] = "PPT"
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
        metadata.to_csv(os.path.join(os.getcwd(), "Godwin_2025_metadata.csv"), index=True)
    metadata["Pub2UpdateTime"] = pd.to_timedelta(metadata["Pub2UpdateTime"])    # validate casting
    return metadata
