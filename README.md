# Citation Benefits in Eye-Movement Data Sharing

## Background
Godwin et al. (2025) conducted a comprehensive survey of articles published between 2015 and 2024 that reported
eye-movement data. Their analysis revealed that researchers often share processed data that enables verification
of statistical results (_analytic reproducibility_), they rarely share the raw eye-movement data or experimental
materials required for _direct replication_ and _secondary analysis_.<br>
This highlights a significant gap in data sharing practices within the field of visual search and eye movements.

## Citation Benefits
The Godwin et al. study did not examine whether sharing eye-movement data leads to increased citations. Since sharing
data should promote reproduction and data-reuse for extended analyses, it is plausible that sharing eye-movement data
could increase articles' impact. However, data sharing is not without costs, such as the time and effort required to
prepare the data and complementary materials for sharing. Therefore, understanding the citation benefits of data sharing
is crucial for researchers considering whether to share their eye-movement data.

## Current Analyses
To investigate the citation benefits of sharing eye-movement data, we are conducting analyses on the dataset compiled
by Godwin et al. (2025). We extract citation counts from the [OpenAlex](https://openalex.org/) database, using their
Python `pyalex` package.
### Analysis #1: Impact Metrics Comparisons
We compare impact metrics (FWCI, MNCS) between articles that share eye-movement data and those
that do not; and also between different granularities of data-sharing (based on Godwin et al.'s coding scheme).
### Analysis #2: Regression Analysis
We also attempt to quantify the benefits of sharing data by regressing citation counts on the time since publication,
and comparing the residuals between articles that share data and those that do not.

### References
```
[1] Godwin, H. J., Dewis, H., Darch, P. T., Hout, M. C., Ernst, D., Broadbent, P.,
Papesh, M., & Wolfe, J. M. (2025). A sharing practices review of the visual search and
eye movements literature reveals recommendations for our field and others.
Behavior Research Methods, 57(9), 235.

[2] Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works,
authors, venues, institutions, and concepts. ArXiv. https://arxiv.org/abs/2205.01833
```
