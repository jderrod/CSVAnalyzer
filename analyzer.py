from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


@dataclass
class AnalysisResult:
    machine_without_operator: List[datetime]
    operator_without_machine: List[Tuple[datetime, datetime]]
    total_machine_events: int
    total_operator_intervals: int


class AnalysisError(RuntimeError):
    """Raised when the analysis cannot be completed because of missing data."""


def _load_table(path: Path) -> pd.DataFrame:
    lower_suffix = path.suffix.lower()
    if lower_suffix == ".csv":
        df = pd.read_csv(path)
    elif lower_suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise AnalysisError(
            f"Unsupported file format for {path.name}. Expected CSV or Excel files."
        )
    if df.empty:
        raise AnalysisError(f"{path.name} is empty.")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _ensure_datetime(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    return converted.dropna()


def _find_matching_column(columns: Sequence[str], *candidates: str) -> str:
    normalized = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized:
            return normalized[key]
    for col in columns:
        lower = col.lower().strip()
        if any(key in lower for key in candidates):
            return col
    raise AnalysisError(
        "Could not find a required column. Tried to match one of: "
        + ", ".join(candidates)
    )


def _extract_machine_times(df: pd.DataFrame) -> List[datetime]:
    column = _find_matching_column(df.columns, "date", "timestamp", "time")
    timestamps = _ensure_datetime(df[column])
    if timestamps.empty:
        raise AnalysisError(
            "No valid timestamps found in machine data column " f"'{column}'."
        )
    return sorted(timestamps.dt.to_pydatetime().tolist())


def _extract_operator_intervals(df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
    start_col = _find_matching_column(df.columns, "start date", "start", "begin")
    end_col = _find_matching_column(df.columns, "end date", "end", "finish")
    start_times = _ensure_datetime(df[start_col])
    end_times = _ensure_datetime(df[end_col])
    if len(start_times) != len(end_times):
        min_len = min(len(start_times), len(end_times))
        start_times = start_times.iloc[:min_len]
        end_times = end_times.iloc[:min_len]
    valid = start_times.notna() & end_times.notna()
    intervals = []
    for start, end in zip(start_times[valid], end_times[valid]):
        if end < start:
            continue
        intervals.append((start.to_pydatetime(), end.to_pydatetime()))
    if not intervals:
        raise AnalysisError("No valid operator intervals found.")
    intervals.sort(key=lambda pair: pair[0])
    return intervals


def locate_source_files(base_directory: Path) -> Tuple[Path, Path]:
    machine_candidates = sorted(base_directory.glob("*MachineData*"))
    operator_candidates = sorted(base_directory.glob("*OperatorData*"))
    if not machine_candidates:
        raise AnalysisError(
            f"No file containing 'MachineData' found in {base_directory.resolve()}"
        )
    if not operator_candidates:
        raise AnalysisError(
            f"No file containing 'OperatorData' found in {base_directory.resolve()}"
        )
    return machine_candidates[0], operator_candidates[0]


def _bisect_left(times: Sequence[datetime], target: datetime) -> int:
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(times: Sequence[datetime], target: datetime) -> int:
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if target < times[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _machine_without_operator(
    machine_times: Sequence[datetime],
    operator_intervals: Sequence[Tuple[datetime, datetime]],
) -> List[datetime]:
    if not operator_intervals:
        return list(machine_times)
    starts = [interval[0] for interval in operator_intervals]
    ends = [interval[1] for interval in operator_intervals]
    output: List[datetime] = []
    for timestamp in machine_times:
        idx = _bisect_right(starts, timestamp) - 1
        if idx < 0 or ends[idx] < timestamp:
            output.append(timestamp)
    return output


def _operator_without_machine(
    machine_times: Sequence[datetime],
    operator_intervals: Sequence[Tuple[datetime, datetime]],
) -> List[Tuple[datetime, datetime]]:
    output: List[Tuple[datetime, datetime]] = []
    if not machine_times:
        return list(operator_intervals)
    for start, end in operator_intervals:
        left = _bisect_left(machine_times, start)
        right = _bisect_right(machine_times, end)
        if right - left == 0:
            output.append((start, end))
    return output


def analyze(
    base_directory: Path | str | None = None,
    *,
    machine_file: Path | str | None = None,
    operator_file: Path | str | None = None,
) -> AnalysisResult:
    if machine_file is not None and operator_file is not None:
        machine_path = Path(machine_file)
        operator_path = Path(operator_file)
    else:
        base_path = Path(base_directory) if base_directory is not None else Path.cwd()
        machine_path, operator_path = locate_source_files(base_path)
    machine_table = _load_table(machine_path)
    operator_table = _load_table(operator_path)
    machine_times = _extract_machine_times(machine_table)
    operator_intervals = _extract_operator_intervals(operator_table)
    machine_without_operator = _machine_without_operator(
        machine_times, operator_intervals
    )
    operator_without_machine = _operator_without_machine(
        machine_times, operator_intervals
    )
    return AnalysisResult(
        machine_without_operator=machine_without_operator,
        operator_without_machine=operator_without_machine,
        total_machine_events=len(machine_times),
        total_operator_intervals=len(operator_intervals),
    )


def _format_result(result: AnalysisResult) -> str:
    lines = []
    machine_without_count = len(result.machine_without_operator)
    operator_without_count = len(result.operator_without_machine)

    lines.append(
        "Machine activations without operator present "
        f"({machine_without_count}/{result.total_machine_events}):"
    )
    if result.machine_without_operator:
        for timestamp in result.machine_without_operator:
            lines.append(f"  - {timestamp.isoformat()}")
    else:
        lines.append("  None")

    lines.append(
        "\nOperator presence without machine activations "
        f"({operator_without_count}/{result.total_operator_intervals}):"
    )
    if result.operator_without_machine:
        for start, end in result.operator_without_machine:
            lines.append(f"  - {start.isoformat()} to {end.isoformat()}")
    else:
        lines.append("  None")
    return "\n".join(lines)


def export_result_to_csv(result: AnalysisResult, output_path: Path | str) -> None:
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp in result.machine_without_operator:
        rows.append(
            {
                "category": "machine_without_operator",
                "timestamp": timestamp.isoformat(),
                "operator_start": "",
                "operator_end": "",
            }
        )
    for start, end in result.operator_without_machine:
        rows.append(
            {
                "category": "operator_without_machine",
                "timestamp": "",
                "operator_start": start.isoformat(),
                "operator_end": end.isoformat(),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=["category", "timestamp", "operator_start", "operator_end"],
    )
    df.to_csv(path, index=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    export_path: Path | None = None
    try:
        if len(args) == 3:
            export_path = Path(args[2])
            result = analyze(machine_file=args[0], operator_file=args[1])
        elif len(args) == 2:
            result = analyze(machine_file=args[0], operator_file=args[1])
        elif len(args) == 1:
            result = analyze(base_directory=args[0])
        else:
            result = analyze()
    except AnalysisError as exc:
        print(f"Analysis failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"Unexpected error: {exc}")
        return 1
    print(_format_result(result))
    if export_path is not None:
        try:
            export_result_to_csv(result, export_path)
        except Exception as exc:
            print(f"Failed to export CSV: {exc}")
            return 1
        print(f"Results exported to {export_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
