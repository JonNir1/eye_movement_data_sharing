import os

import pandas as pd


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
