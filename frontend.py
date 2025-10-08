import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from analyzer import (
    AnalysisError,
    _format_result,
    analyze,
    export_result_to_csv,
    locate_source_files,
)


class AnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Machine/Operator Analyzer")
        self.geometry("720x480")
        self._create_widgets()

    def _create_widgets(self) -> None:
        machine_frame = tk.Frame(self)
        machine_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        self._add_header_images()
        tk.Label(machine_frame, text="Machine file:").pack(side=tk.LEFT)
        self.machine_file_var = tk.StringVar()
        machine_entry = tk.Entry(
            machine_frame, textvariable=self.machine_file_var, width=60
        )
        machine_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(
            machine_frame, text="Browse", command=self._select_machine_file
        ).pack(side=tk.LEFT)

        operator_frame = tk.Frame(self)
        operator_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(operator_frame, text="Operator file:").pack(side=tk.LEFT)
        self.operator_file_var = tk.StringVar()
        operator_entry = tk.Entry(
            operator_frame, textvariable=self.operator_file_var, width=60
        )
        operator_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(
            operator_frame, text="Browse", command=self._select_operator_file
        ).pack(side=tk.LEFT)

        control_frame = tk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(control_frame, text="Analyze", command=self._run_analysis).pack(
            side=tk.LEFT
        )
        tk.Button(control_frame, text="Export CSV", command=self._export_csv).pack(
            side=tk.LEFT, padx=5
        )

        self.summary_var = tk.StringVar(value="")
        summary_label = tk.Label(
            self, textvariable=self.summary_var, justify=tk.LEFT, anchor="w"
        )
        summary_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        columns_wrapper = tk.Frame(self)
        columns_wrapper.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 5))

        self.columns_canvas = tk.Canvas(columns_wrapper, highlightthickness=0)
        self.columns_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.columns_vscroll = tk.Scrollbar(
            columns_wrapper, orient=tk.VERTICAL, command=self.columns_canvas.yview
        )
        self.columns_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.columns_canvas.configure(yscrollcommand=self.columns_vscroll.set)

        self.columns_container = tk.Frame(self.columns_canvas)
        self.columns_window = self.columns_canvas.create_window(
            (0, 0), window=self.columns_container, anchor="nw"
        )

        def _on_container_configure(event: tk.Event) -> None:
            self.columns_canvas.configure(scrollregion=self.columns_canvas.bbox("all"))

        self.columns_container.bind("<Configure>", _on_container_configure)

        def _on_canvas_configure(event: tk.Event) -> None:
            self.columns_canvas.itemconfigure(self.columns_window, height=event.height)

        self.columns_canvas.bind("<Configure>", _on_canvas_configure)

        self.columns_hscroll = tk.Scrollbar(
            self, orient=tk.HORIZONTAL, command=self.columns_canvas.xview
        )
        self.columns_hscroll.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.columns_canvas.configure(xscrollcommand=self.columns_hscroll.set)

        self._prefill_defaults()
        self._latest_result = None

    def _add_header_images(self) -> None:
        image_frame = tk.Frame(self)
        image_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

        def _load_image(path: Path) -> ImageTk.PhotoImage | None:
            if not path.exists():
                return None
            with Image.open(path) as img:
                resized = img.resize((64, 64), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(resized)

        base_path = Path(__file__).resolve().parent
        images_path = base_path / "images"

        self._left_image = _load_image(images_path / "pikachu.jpg")
        self._right_image = _load_image(images_path / "kirby.jpg")

        left_label = tk.Label(image_frame, image=self._left_image)
        left_label.pack(side=tk.LEFT, anchor="nw")
        right_label = tk.Label(image_frame, image=self._right_image)
        right_label.pack(side=tk.RIGHT, anchor="ne")

    def _prefill_defaults(self) -> None:
        try:
            machine, operator = locate_source_files(Path.cwd())
        except AnalysisError:
            return
        self.machine_file_var.set(str(machine))
        self.operator_file_var.set(str(operator))

    def _select_machine_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select machine data file",
            filetypes=[("CSV/Excel", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if file_path:
            self.machine_file_var.set(file_path)

    def _select_operator_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select operator data file",
            filetypes=[("CSV/Excel", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if file_path:
            self.operator_file_var.set(file_path)

    def _run_analysis(self) -> None:
        machine_file = Path(self.machine_file_var.get()).expanduser()
        operator_file = Path(self.operator_file_var.get()).expanduser()
        missing = []
        if not machine_file.exists():
            missing.append(f"Machine file not found: {machine_file}")
        if not operator_file.exists():
            missing.append(f"Operator file not found: {operator_file}")
        if missing:
            messagebox.showerror("Missing File", "\n".join(missing))
            return
        self.summary_var.set("")
        try:
            result = analyze(machine_file=machine_file, operator_file=operator_file)
        except AnalysisError as exc:
            messagebox.showerror("Analysis Failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - unexpected failure
            messagebox.showerror("Unexpected Error", str(exc))
            return
        self._latest_result = result
        summary = (
            "Overall Summary\n"
            f"Machine mismatches: {result.total_machine_mismatches}/{result.total_machine_events}"
            f" (correct: {result.total_machine_events - result.total_machine_mismatches})\n"
            f"Operator mismatches: {result.total_operator_mismatches}/{result.total_operator_intervals}"
            f" (correct: {result.total_operator_intervals - result.total_operator_mismatches})"
        )
        self.summary_var.set(summary)
        self._populate_machine_columns(result)

    def _export_csv(self) -> None:
        if self._latest_result is None:
            messagebox.showinfo("No Data", "Run an analysis before exporting.")
            return
        export_path = filedialog.asksaveasfilename(
            title="Export analysis to CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not export_path:
            return
        try:
            export_result_to_csv(self._latest_result, export_path)
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))
            return
        messagebox.showinfo("Export Complete", f"Results saved to {export_path}")

    def _populate_machine_columns(self, result) -> None:
        for child in self.columns_container.winfo_children():
            child.destroy()

        for machine_name, report in result.machines.items():
            column = tk.Frame(self.columns_container, borderwidth=1, relief=tk.GROOVE)
            column.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            column.configure(width=260)
            column.pack_propagate(False)

            title = tk.Label(column, text=machine_name, font=("Segoe UI", 12, "bold"))
            title.pack(anchor="n", pady=(8, 4))

            summary_text = (
                f"Machine mismatches: {report.machine_mismatch_count}/{report.total_machine_events}\n"
                f"Machine correct: {report.machine_correct_count}\n"
                f"Operator mismatches: {report.operator_mismatch_count}/{report.total_operator_intervals}\n"
                f"Operator correct: {report.operator_correct_count}"
            )
            tk.Label(
                column,
                text=summary_text,
                justify=tk.LEFT,
                anchor="w",
                font=("Segoe UI", 9),
            ).pack(fill=tk.X, padx=8)

            tk.Label(
                column, text="Machine activations without operator", font=("Segoe UI", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(8, 2))
            machine_list = tk.Listbox(column, height=6, width=32)
            machine_list.pack(fill=tk.BOTH, expand=True, padx=8)
            for ts in report.machine_without_operator:
                machine_list.insert(tk.END, ts.isoformat())
            if not report.machine_without_operator:
                machine_list.insert(tk.END, "None")

            tk.Label(
                column, text="Operator presence without machine", font=("Segoe UI", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(8, 2))
            operator_list = tk.Listbox(column, height=6, width=32)
            operator_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            for start, end in report.operator_without_machine:
                operator_list.insert(tk.END, f"{start.isoformat()} - {end.isoformat()}")
            if not report.operator_without_machine:
                operator_list.insert(tk.END, "None")

        self.columns_canvas.configure(scrollregion=self.columns_canvas.bbox("all"))


def main() -> None:
    app = AnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
