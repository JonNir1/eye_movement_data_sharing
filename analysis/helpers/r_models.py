"""Mixed models fitted in R (`lme4`), for what `statsmodels` cannot fit: Poisson and negative-binomial LMEs.

`fit_glmm()` writes the data to a temp CSV, runs `helpers/fit_glmm.R` through `Rscript` as a subprocess,
and reads the results back from JSON. There is deliberately no `rpy2`: it is not in the venv and is
fiddly to install on Windows.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from helpers.config import R_LIBRARY_DIR

Family = Literal["gaussian", "poisson", "nb"]

R_SCRIPT = Path(__file__).with_name("fit_glmm.R")
_RSCRIPT_FALLBACK = Path(r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe")
_Z_975 = 1.959964


@dataclass(frozen=True)
class GlmmResult:
    """What `fit_glmm()` returns.

    `fixed_effects` is indexed by term with `estimate`, `se`, `statistic` and `p`; the count families
    (Poisson, NB) add `rate_ratio` and its Wald 95% CI, since `exp(b)` is the multiplicative effect.
    `predictions`, when requested, repeats the new data with `fit`, the 95% CI (`ci_low`, `ci_high`) and
    the +/- 1 SE interval (`sem_low`, `sem_high`) on the response scale (fixed effects only, i.e. the
    population-level prediction). The intervals are symmetric on the link scale, so not on the response scale.
    """
    family: Family
    formula: str
    fixed_effects: pd.DataFrame
    fit: dict[str, Any]
    variance_components: pd.DataFrame
    predictions: Optional[pd.DataFrame]


def find_rscript() -> str:
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path
    if _RSCRIPT_FALLBACK.exists():
        return str(_RSCRIPT_FALLBACK)
    raise FileNotFoundError("Rscript not found: put it on PATH or install R 4.5")


def _to_response_scale(link: pd.Series, family: Family) -> pd.Series:
    if family == "gaussian":
        return link
    return np.exp(link)


def _parse_predictions(raw: list[dict], newdata: pd.DataFrame, family: Family) -> pd.DataFrame:
    link = pd.DataFrame(raw)
    out = newdata.reset_index(drop=True).copy()
    out["fit"] = _to_response_scale(link["link"], family)
    out["ci_low"] = _to_response_scale(link["link"] - _Z_975 * link["link_se"], family)
    out["ci_high"] = _to_response_scale(link["link"] + _Z_975 * link["link_se"], family)
    out["sem_low"] = _to_response_scale(link["link"] - link["link_se"], family)
    out["sem_high"] = _to_response_scale(link["link"] + link["link_se"], family)
    return out


def _parse_fixed_effects(raw: list[dict], family: Family) -> pd.DataFrame:
    table = pd.DataFrame(raw).set_index("term")
    if family == "gaussian":
        return table
    table["rate_ratio"] = np.exp(table["estimate"])
    table["rr_ci_low"] = np.exp(table["estimate"] - _Z_975 * table["se"])
    table["rr_ci_high"] = np.exp(table["estimate"] + _Z_975 * table["se"])
    return table


def fit_glmm(
        df: pd.DataFrame,
        formula: str,
        family: Family,
        factors: tuple[str, ...] = ("venue_name", "article"),
        newdata: Optional[pd.DataFrame] = None,
        theta: Optional[float] = None,
        timeout: int = 900,
) -> GlmmResult:
    """Fit `formula` (lme4 syntax, e.g. `y ~ x + (1 | group/subject)`) to `df` with the given family.

    :param factors: columns of `df` to treat as factors in R (the grouping variables).
    :param newdata: optional frame to predict on, fixed effects only; it needs every column the fixed
        part of `formula` uses.
    :param theta: negative-binomial dispersion to hold fixed (family `nb` only). Estimating it with
        `glmer.nb` takes minutes; a fixed theta, e.g. from an earlier fit, takes seconds and gives the same
        log-likelihood up to the theta that was fixed, so AICs of models sharing one theta are comparable.
    :raises ValueError: when `theta` is given for a family other than `nb`.
    :raises RuntimeError: when R fails, with R's stderr in the message.
    """
    if theta is not None and family != "nb":
        raise ValueError(f"theta only applies to the negative-binomial family, not {family!r}")
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        job = {
            "data": str(work / "data.csv"), "formula": formula, "family": family,
            "factors": list(factors), "output": str(work / "result.json"),
        }
        df.to_csv(job["data"], index=False)
        if theta is not None:
            job["theta"] = theta
        if newdata is not None:
            job["newdata"] = str(work / "newdata.csv")
            newdata.to_csv(job["newdata"], index=False)
        (work / "job.json").write_text(json.dumps(job), encoding="utf8")

        completed = subprocess.run(
            [find_rscript(), str(R_SCRIPT), str(work / "job.json")],
            capture_output=True, text=True, encoding="utf8", timeout=timeout,
            env={**os.environ, "R_LIBS_USER": str(R_LIBRARY_DIR)},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"R failed fitting `{formula}` ({family}):\n{completed.stderr.strip()}")
        raw = json.loads(Path(job["output"]).read_text(encoding="utf8"))

    predictions = None
    if newdata is not None:
        predictions = _parse_predictions(raw["predictions"], newdata, family)
    return GlmmResult(
        family=family, formula=formula,
        fixed_effects=_parse_fixed_effects(raw["fixed_effects"], family),
        fit={key: (np.nan if value is None else value) for key, value in raw["fit"].items()},
        variance_components=pd.DataFrame(raw["variance_components"]),
        predictions=predictions,
    )
