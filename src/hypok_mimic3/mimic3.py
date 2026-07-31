from __future__ import annotations

import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .labels import PotassiumLabeler
from .utils import write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_: object):
        return iterable


_SUBJECT_RE = re.compile(r"(?:^|/)p(?P<subject>\d{6})(?:/|-)")


def _find_table(root: Path, stem: str) -> Path:
    names = (stem, stem.lower(), stem.upper())
    candidates: list[Path] = []
    for name in names:
        for suffix in (".parquet", ".csv.gz", ".csv"):
            candidates.extend((root / f"{name}{suffix}", root / "hosp" / f"{name}{suffix}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find MIMIC-III table {stem} under {root}")


def _duckdb_reader_sql(path: Path) -> str:
    safe = str(path).replace("'", "''")
    if path.suffix == ".parquet":
        return f"read_parquet('{safe}')"
    return (
        f"read_csv_auto('{safe}', header=true, sample_size=100000, "
        "ignore_errors=false, union_by_name=true)"
    )


def _record_list_path(root: Path) -> Path:
    for candidate in (root / "RECORDS-waveforms", root / "matched" / "RECORDS-waveforms"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing RECORDS-waveforms under {root}. Download the 806 KB index from "
        "https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms"
    )


def _subject_from_record_path(record_path: str) -> int:
    match = _SUBJECT_RE.search(record_path)
    if not match:
        raise ValueError(f"Cannot parse MIMIC-III subject_id from {record_path}")
    return int(match.group("subject"))


def _remote_record_locator(record_path: str, pn_dir: str) -> tuple[str, str]:
    """Return the WFDB record name and PhysioNet directory for a nested path."""
    path = Path(record_path)
    parent = path.parent.as_posix()
    remote_dir = pn_dir.rstrip("/")
    if parent not in ("", "."):
        remote_dir = f"{remote_dir}/{parent}"
    return path.name, remote_dir


def _read_waveform_header(
    record_path: str,
    waveform_root: Path,
    pn_dir: str | None,
) -> dict:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wfdb is required to index MIMIC-III waveforms") from exc

    local_record = waveform_root / record_path
    if local_record.with_suffix(".hea").exists():
        header = wfdb.rdheader(str(local_record))
        source = "local"
    elif pn_dir:
        remote_name, remote_dir = _remote_record_locator(record_path, pn_dir)
        header = wfdb.rdheader(remote_name, pn_dir=remote_dir)
        source = "physionet"
    else:
        raise FileNotFoundError(f"Missing local header: {local_record}.hea")

    if header.base_date is None or header.base_time is None:
        raise ValueError("master header has no base_date/base_time")
    start = datetime.combine(header.base_date, header.base_time)
    fs = float(header.fs)
    samples = int(header.sig_len)
    end = start + timedelta(seconds=samples / fs)
    return {
        "subject_id": _subject_from_record_path(record_path),
        "waveform_record": record_path,
        "record_start_time": start.isoformat(sep=" "),
        "record_end_time": end.isoformat(sep=" "),
        "sampling_rate": fs,
        "signal_length": samples,
        "duration_seconds": samples / fs,
        "n_sig": int(header.n_sig),
        "lead_names": "|".join(header.sig_name or []),
        "header_source": source,
        "index_error": "",
    }


def build_waveform_index(
    waveform_root: str | Path,
    output_csv: str | Path,
    pn_dir: str | None = "mimic3wdb-matched/1.0",
    workers: int = 16,
    limit: int | None = None,
) -> pd.DataFrame:
    """Index the 22,317 matched MIMIC-III master waveform records.

    Only WFDB headers are read. Signal files are not downloaded at this stage.
    """
    root = Path(waveform_root).expanduser().resolve()
    record_list = _record_list_path(root)
    records = [
        line.strip().removesuffix(".hea")
        for line in record_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if limit is not None:
        records = records[:limit]

    def safe_read(record_path: str) -> dict:
        try:
            return _read_waveform_header(record_path, root, pn_dir)
        except Exception as exc:
            subject_id = -1
            try:
                subject_id = _subject_from_record_path(record_path)
            except ValueError:
                pass
            return {
                "subject_id": subject_id,
                "waveform_record": record_path,
                "record_start_time": "",
                "record_end_time": "",
                "sampling_rate": math.nan,
                "signal_length": -1,
                "duration_seconds": math.nan,
                "n_sig": -1,
                "lead_names": "",
                "header_source": "",
                "index_error": f"{type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(
            tqdm(
                pool.map(safe_read, records),
                total=len(records),
                desc="Indexing MIMIC-III waveform headers",
            )
        )
    frame = pd.DataFrame(rows)
    target = Path(output_csv).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    write_json(
        target.with_suffix(".summary.json"),
        {
            "records_listed": len(records),
            "records_indexed": int((frame["index_error"] == "").sum()),
            "index_errors": int((frame["index_error"] != "").sum()),
            "subjects": int(frame.loc[frame["subject_id"] >= 0, "subject_id"].nunique()),
            "remote_header_directory": pn_dir,
        },
    )
    return frame


def _selected_itemids(config: dict) -> list[int]:
    ids = [int(x) for x in config["data"]["potassium_itemids"]["serum"]]
    if config["data"].get("include_whole_blood", False):
        ids.extend(int(x) for x in config["data"]["potassium_itemids"]["whole_blood"])
    return sorted(set(ids))


def validate_potassium_items(clinical_root: str | Path, itemids: Iterable[int]) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required to query MIMIC-III Clinical") from exc

    root = Path(clinical_root).expanduser().resolve()
    table = _find_table(root, "D_LABITEMS")
    ids = ",".join(str(int(x)) for x in sorted(set(itemids)))
    frame = duckdb.sql(
        f"""
        SELECT
            CAST(itemid AS INTEGER) AS itemid,
            CAST(label AS VARCHAR) AS label,
            CAST(fluid AS VARCHAR) AS fluid,
            CAST(category AS VARCHAR) AS category
        FROM {_duckdb_reader_sql(table)}
        WHERE CAST(itemid AS INTEGER) IN ({ids})
        ORDER BY itemid
        """
    ).df()
    missing = set(int(x) for x in itemids) - set(frame["itemid"].astype(int))
    if missing:
        raise ValueError(f"Configured potassium item IDs absent from D_LABITEMS: {missing}")
    invalid = frame[~frame["label"].str.contains("potassium", case=False, na=False)]
    if not invalid.empty:
        raise ValueError(f"Configured item IDs are not potassium items:\n{invalid}")
    return frame


def _drop_near_duplicate_windows(frame: pd.DataFrame, separation_seconds: float) -> pd.DataFrame:
    if frame.empty or separation_seconds <= 0:
        return frame
    kept: list[int] = []
    for _, part in frame.sort_values(["waveform_record", "ecg_anchor_time", "labevent_id"]).groupby(
        "waveform_record", sort=False
    ):
        last_time: pd.Timestamp | None = None
        for idx, row in part.iterrows():
            current = pd.Timestamp(row["ecg_anchor_time"])
            if last_time is None or (current - last_time).total_seconds() >= separation_seconds:
                kept.append(idx)
                last_time = current
    return (
        frame.loc[kept]
        .sort_values(["subject_id", "ecg_anchor_time", "labevent_id"])
        .reset_index(drop=True)
    )


def build_potassium_cohort(config: dict) -> tuple[pd.DataFrame, dict]:
    """Match each potassium result to the nearest covered bedside ECG time."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required to build the MIMIC-III cohort") from exc

    data = config["data"]
    index_path = Path(data["waveform_index_csv"]).expanduser().resolve()
    clinical_root = Path(data["clinical_root"]).expanduser().resolve()
    target = Path(data["cohort_manifest_csv"]).expanduser().resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Build the waveform index first: {index_path}")

    labevents = _find_table(clinical_root, "LABEVENTS")
    itemids = _selected_itemids(config)
    dictionary = validate_potassium_items(clinical_root, itemids)
    item_sql = ",".join(str(value) for value in itemids)
    window = int(data["lab_window_minutes"])
    min_k = float(data["min_potassium"])
    max_k = float(data["max_potassium"])
    duration = float(config["preprocess"]["duration_seconds"])

    cohort = duckdb.sql(
        f"""
        WITH records AS (
            SELECT
                CAST(subject_id AS BIGINT) AS subject_id,
                waveform_record,
                CAST(record_start_time AS TIMESTAMP) AS record_start_time,
                CAST(record_end_time AS TIMESTAMP) AS record_end_time,
                CAST(sampling_rate AS DOUBLE) AS sampling_rate,
                CAST(signal_length AS BIGINT) AS signal_length,
                lead_names
            FROM {_duckdb_reader_sql(index_path)}
            WHERE COALESCE(CAST(index_error AS VARCHAR), '') = ''
        ),
        labs AS (
            SELECT
                CAST(subject_id AS BIGINT) AS subject_id,
                CAST(row_id AS BIGINT) AS labevent_id,
                CAST(hadm_id AS BIGINT) AS hadm_id,
                CAST(itemid AS INTEGER) AS potassium_itemid,
                CAST(charttime AS TIMESTAMP) AS potassium_time,
                CAST(valuenum AS DOUBLE) AS potassium,
                CAST(valueuom AS VARCHAR) AS potassium_unit,
                CAST(flag AS VARCHAR) AS potassium_flag
            FROM {_duckdb_reader_sql(labevents)}
            WHERE CAST(itemid AS INTEGER) IN ({item_sql})
              AND CAST(valuenum AS DOUBLE) BETWEEN {min_k} AND {max_k}
              AND charttime IS NOT NULL
        ),
        candidates AS (
            SELECT
                l.*,
                r.waveform_record,
                r.record_start_time,
                r.record_end_time,
                r.sampling_rate,
                r.signal_length,
                r.lead_names,
                CASE
                    WHEN l.potassium_time < r.record_start_time THEN r.record_start_time
                    WHEN l.potassium_time > r.record_end_time THEN r.record_end_time
                    ELSE l.potassium_time
                END AS ecg_anchor_time,
                CASE
                    WHEN l.potassium_time < r.record_start_time
                        THEN date_diff('second', l.potassium_time, r.record_start_time) / 60.0
                    WHEN l.potassium_time > r.record_end_time
                        THEN date_diff('second', r.record_end_time, l.potassium_time) / 60.0
                    ELSE 0.0
                END AS abs_delta_minutes,
                CASE
                    WHEN l.potassium_time BETWEEN r.record_start_time AND r.record_end_time
                        THEN 0 ELSE 1
                END AS outside_record
            FROM labs l
            INNER JOIN records r USING (subject_id)
            WHERE l.potassium_time BETWEEN
                r.record_start_time - INTERVAL '{window} minutes'
                AND r.record_end_time + INTERVAL '{window} minutes'
        )
        SELECT *
        FROM candidates
        WHERE abs_delta_minutes <= {window}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY labevent_id
            ORDER BY outside_record, abs_delta_minutes, waveform_record
        ) = 1
        ORDER BY subject_id, ecg_anchor_time, labevent_id
        """
    ).df()

    if not cohort.empty:
        offset_seconds = (
            pd.to_datetime(cohort["ecg_anchor_time"]) - pd.to_datetime(cohort["record_start_time"])
        ).dt.total_seconds()
        half = duration / 2.0
        max_start = cohort["signal_length"] / cohort["sampling_rate"] - duration
        start_seconds = np.minimum(np.maximum(offset_seconds - half, 0.0), max_start)
        cohort["sample_start"] = np.floor(start_seconds * cohort["sampling_rate"]).astype("int64")
        cohort["sample_end"] = np.minimum(
            cohort["sample_start"] + np.ceil(duration * cohort["sampling_rate"]).astype("int64"),
            cohort["signal_length"].astype("int64"),
        )
        cohort = cohort[cohort["sample_end"] > cohort["sample_start"]].copy()

    cohort = _drop_near_duplicate_windows(
        cohort,
        float(data.get("minimum_window_separation_seconds", duration)),
    )
    labeler = PotassiumLabeler.from_config(config)
    cohort["label_id"] = labeler.transform(cohort["potassium"].to_numpy())
    cohort["label"] = labeler.label_names(cohort["label_id"].to_numpy())
    cohort["study_id"] = cohort["labevent_id"].astype("int64")

    if cohort["study_id"].duplicated().any():
        raise AssertionError("Each MIMIC-III labevent must map to at most one ECG window")
    target.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(target, index=False)
    counts = cohort["label"].value_counts().reindex(labeler.names, fill_value=0)
    summary = {
        "mimic_waveform_version": data["mimic_waveform_version"],
        "mimic_clinical_version": data["mimic_clinical_version"],
        "lab_window_minutes": window,
        "potassium_itemids": itemids,
        "potassium_item_dictionary": dictionary.to_dict("records"),
        "candidate_windows": int(len(cohort)),
        "records": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_subject_counts": {
            name: int(cohort.loc[cohort["label"] == name, "subject_id"].nunique())
            for name in labeler.names
        },
        "within_record_fraction": (
            float((cohort["outside_record"] == 0).mean()) if len(cohort) else None
        ),
        "median_abs_time_delta_minutes": (
            float(cohort["abs_delta_minutes"].median()) if len(cohort) else None
        ),
    }
    write_json(target.with_suffix(".summary.json"), summary)
    return cohort, summary


def _normalize_lead(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _select_leads(signal: np.ndarray, names: list[str], requested: list[str]) -> np.ndarray:
    available = {_normalize_lead(name): idx for idx, name in enumerate(names)}
    aliases = {
        "II": ("II", "MLII", "ECGII"),
        "I": ("I", "ECGI"),
        "III": ("III", "ECGIII"),
        "V": ("V", "V1", "MCL", "MCL1"),
    }
    selected = []
    for lead in requested:
        choices = aliases.get(_normalize_lead(lead), (_normalize_lead(lead),))
        index = next((available[key] for key in choices if key in available), None)
        if index is None:
            raise ValueError(f"Missing requested lead {lead}; available={names}")
        selected.append(signal[:, index])
    return np.column_stack(selected)


def _cache_name(row: pd.Series) -> str:
    digest = hashlib.sha1(
        f"{row['waveform_record']}:{int(row['sample_start'])}:{int(row['sample_end'])}".encode()
    ).hexdigest()[:16]
    return f"p{int(row['subject_id']):06d}_{int(row['study_id'])}_{digest}.npz"


def materialize_waveform_windows(
    config: dict,
    workers: int = 8,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch only selected 10-second windows and cache them as compressed NPZ files."""
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wfdb is required to materialize ECG windows") from exc

    data = config["data"]
    source = Path(data["cohort_manifest_csv"]).expanduser().resolve()
    target = Path(data["materialized_cohort_csv"]).expanduser().resolve()
    waveform_root = Path(data["waveform_root"]).expanduser().resolve()
    cache_root = Path(data["waveform_cache_dir"]).expanduser().resolve()
    pn_dir = data.get("physionet_waveform_dir")
    requested_leads = list(data["lead_order"])
    cache_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if limit is not None:
        frame = frame.head(limit).copy()

    def materialize(item: tuple[int, pd.Series]) -> dict:
        _, row = item
        output = row.to_dict()
        cache_path = cache_root / _cache_name(row)
        output["waveform_cache_path"] = str(cache_path)
        output["materialize_error"] = ""
        if cache_path.exists():
            return output
        try:
            local = waveform_root / str(row["waveform_record"])
            kwargs = {
                "sampfrom": int(row["sample_start"]),
                "sampto": int(row["sample_end"]),
                "physical": True,
                "m2s": True,
            }
            if local.with_suffix(".hea").exists():
                record = wfdb.rdrecord(str(local), **kwargs)
            elif pn_dir:
                remote_name, remote_dir = _remote_record_locator(
                    str(row["waveform_record"]), pn_dir
                )
                record = wfdb.rdrecord(remote_name, pn_dir=remote_dir, **kwargs)
            else:
                raise FileNotFoundError(f"No local waveform and no physionet_waveform_dir: {local}")
            if record.p_signal is None:
                raise ValueError("WFDB record has no calibrated physical signal")
            signal = _select_leads(record.p_signal, list(record.sig_name), requested_leads)
            if not np.isfinite(signal).all():
                raise ValueError("selected ECG contains NaN/Inf, likely a signal gap")
            np.savez_compressed(
                cache_path,
                signal=signal.astype(np.float32),
                sampling_rate=np.float32(record.fs),
                lead_names=np.asarray(requested_leads),
            )
        except Exception as exc:
            output["materialize_error"] = f"{type(exc).__name__}: {exc}"
        return output

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(
            tqdm(
                pool.map(materialize, frame.iterrows()),
                total=len(frame),
                desc="Materializing selective ECG windows",
            )
        )
    result = pd.DataFrame(rows)
    success = result[result["materialize_error"] == ""].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    success.to_csv(target, index=False)
    errors = result[result["materialize_error"] != ""]
    errors.to_csv(target.with_name(f"{target.stem}_errors.csv"), index=False)
    summary = {
        "requested": int(len(result)),
        "materialized": int(len(success)),
        "records": int(len(success)),
        "failed": int(len(errors)),
        "subjects": int(success["subject_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in success["label"].value_counts().items()},
        "lead_order": requested_leads,
        "potassium_itemids": _selected_itemids(config),
        "lab_window_minutes": int(data["lab_window_minutes"]),
        "cache_directory": str(cache_root),
    }
    write_json(target.with_suffix(".summary.json"), summary)
    return success, summary
