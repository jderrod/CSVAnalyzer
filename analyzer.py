from __future__ import annotations

import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


@dataclass
class MachineReport:
    machine_name: str
    machine_without_operator: List[datetime] = field(default_factory=list)
    operator_without_machine: List[Tuple[datetime, datetime]] = field(
        default_factory=list
    )
    total_machine_events: int = 0
    total_operator_intervals: int = 0

    @property
    def machine_mismatch_count(self) -> int:
        return len(self.machine_without_operator)

    @property
    def operator_mismatch_count(self) -> int:
        return len(self.operator_without_machine)

    @property
    def machine_correct_count(self) -> int:
        return max(self.total_machine_events - self.machine_mismatch_count, 0)

    @property
    def operator_correct_count(self) -> int:
        return max(self.total_operator_intervals - self.operator_mismatch_count, 0)


@dataclass
class AnalysisResult:
    machines: Mapping[str, MachineReport]

    @property
    def total_machine_events(self) -> int:
        return sum(report.total_machine_events for report in self.machines.values())

    @property
    def total_operator_intervals(self) -> int:
        return sum(
            report.total_operator_intervals for report in self.machines.values()
        )

    @property
    def total_machine_mismatches(self) -> int:
        return sum(
            report.machine_mismatch_count for report in self.machines.values()
        )

    @property
    def total_operator_mismatches(self) -> int:
        return sum(
            report.operator_mismatch_count for report in self.machines.values()
        )


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


def _extract_machine_events(df: pd.DataFrame) -> Dict[str, List[datetime]]:
    timestamp_col = _find_matching_column(df.columns, "date", "timestamp", "time")
    machine_col = _find_matching_column(df.columns, "machine name", "machine")
    timestamp_series = pd.to_datetime(df[timestamp_col], errors="coerce")
    machine_series = df[machine_col]

    events: Dict[str, List[datetime]] = defaultdict(list)
    for raw_machine, timestamp in zip(machine_series, timestamp_series):
        if pd.isna(raw_machine) or pd.isna(timestamp):
            continue
        machine_name = str(raw_machine).strip()
        if not machine_name or machine_name.lower() == "machine name":
            continue
        events[machine_name].append(timestamp.to_pydatetime())

    for machine_name in list(events.keys()):
        events[machine_name].sort()

    if not events:
        raise AnalysisError(
            "No machine events found. Verify the machine name and timestamp columns."
        )
    return events


def _extract_operator_intervals(
    df: pd.DataFrame,
) -> Dict[str, List[Tuple[datetime, datetime]]]:
    machine_col = _find_matching_column(df.columns, "machine name", "machine")
    start_col = _find_matching_column(df.columns, "start date", "start", "begin")
    end_col = _find_matching_column(df.columns, "end date", "end", "finish")

    machine_series = df[machine_col]
    start_series = pd.to_datetime(df[start_col], errors="coerce")
    end_series = pd.to_datetime(df[end_col], errors="coerce")

    intervals: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
    for raw_machine, start, end in zip(machine_series, start_series, end_series):
        if pd.isna(raw_machine) or pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            continue
        machine_name = str(raw_machine).strip()
        if not machine_name or machine_name.lower() == "machine name":
            continue
        intervals[machine_name].append((start.to_pydatetime(), end.to_pydatetime()))

    for machine_name in list(intervals.keys()):
        intervals[machine_name].sort(key=lambda pair: pair[0])

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

    machine_events = _extract_machine_events(machine_table)
    operator_intervals = _extract_operator_intervals(operator_table)

    machine_names = OrderedDict()
    for name in machine_events:
        machine_names.setdefault(name, None)
    for name in operator_intervals:
        machine_names.setdefault(name, None)

    reports: OrderedDict[str, MachineReport] = OrderedDict()
    for machine_name in machine_names:
        events = machine_events.get(machine_name, [])
        intervals = operator_intervals.get(machine_name, [])
        report = MachineReport(
            machine_name=machine_name,
            total_machine_events=len(events),
            total_operator_intervals=len(intervals),
        )
        report.machine_without_operator = _machine_without_operator(events, intervals)
        report.operator_without_machine = _operator_without_machine(events, intervals)
        reports[machine_name] = report

    if not reports:
        raise AnalysisError("No overlapping machine names found between the datasets.")

    return AnalysisResult(machines=reports)


def _format_result(result: AnalysisResult) -> str:
    lines = []
    total_machine_events = result.total_machine_events
    total_operator_intervals = result.total_operator_intervals
    total_machine_mismatches = result.total_machine_mismatches
    total_operator_mismatches = result.total_operator_mismatches
    machine_correct = max(total_machine_events - total_machine_mismatches, 0)
    operator_correct = max(total_operator_intervals - total_operator_mismatches, 0)
    lines.append(
        "Overall Summary:"
        f"\n  Machine mismatches: {total_machine_mismatches}/{total_machine_events}"
        f"\n  Machine correct: {machine_correct}"
        f"\n  Operator mismatches: {total_operator_mismatches}/{total_operator_intervals}"
        f"\n  Operator correct: {operator_correct}"
    )

    for machine_name, report in result.machines.items():
        lines.append(
            "\n"
            f"Machine: {machine_name}"
            f"\n  Machine mismatches: {report.machine_mismatch_count}/{report.total_machine_events}"
            f"\n  Machine correct: {report.machine_correct_count}"
            f"\n  Operator mismatches: {report.operator_mismatch_count}/{report.total_operator_intervals}"
            f"\n  Operator correct: {report.operator_correct_count}"
        )
        if report.machine_without_operator:
            lines.append("  Machine activations without operator:")
            for timestamp in report.machine_without_operator:
                lines.append(f"    - {timestamp.isoformat()}")
        else:
            lines.append("  Machine activations without operator: None")

        if report.operator_without_machine:
            lines.append("  Operator presence without machine:")
            for start, end in report.operator_without_machine:
                lines.append(
                    f"    - {start.isoformat()} to {end.isoformat()}"
                )
        else:
            lines.append("  Operator presence without machine: None")

    return "\n".join(lines)


def export_result_to_csv(result: AnalysisResult, output_path: Path | str) -> None:
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for machine_name, report in result.machines.items():
        base_row = {
            "machine": machine_name,
            "total_machine_events": report.total_machine_events,
            "total_operator_intervals": report.total_operator_intervals,
            "machine_mismatches": report.machine_mismatch_count,
            "operator_mismatches": report.operator_mismatch_count,
            "machine_correct": report.machine_correct_count,
            "operator_correct": report.operator_correct_count,
        }
        for timestamp in report.machine_without_operator:
            rows.append(
                {
                    **base_row,
                    "category": "machine_without_operator",
                    "machine_timestamp": timestamp.isoformat(),
                    "operator_start": "",
                    "operator_end": "",
                }
            )
        for start, end in report.operator_without_machine:
            rows.append(
                {
                    **base_row,
                    "category": "operator_without_machine",
                    "machine_timestamp": "",
                    "operator_start": start.isoformat(),
                    "operator_end": end.isoformat(),
                }
            )
        rows.append(
            {
                **base_row,
                "category": "summary",
                "machine_timestamp": "",
                "operator_start": "",
                "operator_end": "",
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "machine",
            "category",
            "machine_timestamp",
            "operator_start",
            "operator_end",
            "total_machine_events",
            "total_operator_intervals",
            "machine_mismatches",
            "operator_mismatches",
            "machine_correct",
            "operator_correct",
        ],
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
