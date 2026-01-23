import os

import pyalex
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from load_data import *
from fetch_metadata import fetch_all_metadata

# %%
# Prepare Data

godwin = load_godwin2025()
godwin_subset = (
    godwin
    # exclusion criteria from Godwin et al. 2025:
    .loc[(godwin["YEAR_PUBLISHED"] >= 2017) & (godwin["YEAR_PUBLISHED"] <= 2022)]
    .loc[godwin["IS_PRIMARY_RESEARCH_HUMAN"] == "YES"]
    .loc[godwin["IS_VISUAL_SEARCH"] == "YES"]
    .loc[godwin["IS_EYE_TRACKING"] == "YES"]
    .drop(columns=[
        col for col in godwin.columns if col not in [
            "PAPER_LINK",
            "CLAIMED_TO_SHARE", "SHARING_LINK", "ACTUALLY_SHARED",
            "EXPERIMENT", "MATERIALS", "CODE", "CODEBOOK_GUIDE",
            "BY_FIXATION", "BY_TRIAL", "BY_PPT",
            "data_sharing_class"
        ]
    ])
    .assign(shared_any=lambda df: (df[[
        "EXPERIMENT", "MATERIALS", "CODE", "CODEBOOK_GUIDE",
        "BY_FIXATION", "BY_TRIAL", "BY_PPT"
    ]] == "YES").any(axis=1))
)

# %%
# Fetch Metadata from OpenAlex

metadata = fetch_all_metadata(godwin_subset["PAPER_LINK"].tolist(), sleep_period=0.01, verbose=True)

# %%
# Merge Datasets

combined = (
    metadata
    .dropna(subset=["DOI"])     # drop entries with unsuccessful metadata fetch
    .merge(godwin_subset, on="PAPER_LINK")
    .drop_duplicates(subset="DOI")
    .loc[lambda df: df["IsRetracted"] == False]   # drop retracted articles
)

errors = (
    metadata
    .merge(godwin, on="PAPER_LINK")
    .loc[lambda df: df["Error"].notna()]
)


# %%
# Topic Analysis

topics = dict()
for topic_dict in metadata["Topics"].dropna():
    for topic_id, score in topic_dict.items():
        if topic_id not in topics:
            topics[topic_id] = dict(
                name=pyalex.Topics()[topic_id]["display_name"],
                count=0,
                score=0.0,
            )
        topics[topic_id]["count"] += 1
        topics[topic_id]["score"] += score
del topic_id, score, topic_dict
topics_df = (
    pd.DataFrame(topics).T
    .astype({"count": int, "score": float})
    .sort_values(by="count", ascending=False)
    .reset_index(drop=False)
    .rename(columns={"index": "topic_id", "score": "total_score"})
    .assign(average_score=lambda df: df["total_score"] / df["count"])
)


# %%
# TODO:
## Find additional articles not in their dataset by:
##  1) run the title+abstract query from Godwin et al. 2025 on OpenAlex
##  2) run a Topic-based search of Works, from the topics identified above
##  both searches should be filtered from 2017 onward to match Godwin et al (they included up to 2022)

# TODO:
## - Use these articles to calculate the actual MCNS/FWCI distributions for eye-tracking articles
## - Evaluate the Godwin et al. dataset against these distributions
