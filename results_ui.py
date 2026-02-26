#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, Mapping

from assignment_tasks import run_task1, run_task1_task2, run_task2, run_task3, run_task4
from core.paths import NEWS_CORPUS_PATH, ROOT


TASK2_SMOOTH_KEYS_BIGRAM = [
    "ppl_bigram_laplace",
    "ppl_bigram_interpolation",
    "ppl_bigram_backoff",
    "ppl_bigram_kneser_ney",
]

TASK2_SMOOTH_KEYS_TRIGRAM = [
    "ppl_trigram_laplace",
    "ppl_trigram_interpolation",
    "ppl_trigram_backoff",
    "ppl_trigram_kneser_ney",
]

TASK1_KEYS = [
    "num_sentences",
    "vocab_size",
    "ppl_unigram_mle",
    "ppl_bigram_mle",
    "ppl_trigram_mle",
]

TASK2_KEYS = [
    "num_sentences",
    "vocab_size",
    "bigram_interp_lambda1",
    "bigram_interp_lambda2",
    "ppl_bigram_laplace",
    "ppl_bigram_interpolation",
    "ppl_bigram_backoff",
    "ppl_bigram_kneser_ney",
    "interp_lambda1",
    "interp_lambda2",
    "interp_lambda3",
    "ppl_trigram_laplace",
    "ppl_trigram_interpolation",
    "ppl_trigram_backoff",
    "ppl_trigram_kneser_ney",
]


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return str(value)
        return f"{value:.6f}"
    return str(value)


class ResultsUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NLP Project 2 - Results UI")
        self.geometry("1100x720")
        self.minsize(920, 560)

        self._running = False
        self._action_buttons: list[ttk.Button] = []
        self._tables: Dict[str, ttk.Treeview] = {}

        self.news_path_var = tk.StringVar(value=str(NEWS_CORPUS_PATH))
        self.root_path_var = tk.StringVar(value=str(ROOT))
        self.max_sentences_var = tk.StringVar(value="120000")
        self.min_freq_var = tk.StringVar(value="2")
        self.max_docs_var = tk.StringVar(value="30000")
        self.max_examples_var = tk.StringVar(value="60000")
        self.max_vocab_tokens_var = tk.StringVar(value="6000")
        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="News corpus path").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.news_path_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(8, 6)
        )
        ttk.Button(controls, text="Browse", command=self._browse_news_path).grid(
            row=0, column=3, sticky="w", padx=(0, 12)
        )

        ttk.Label(controls, text="Project root").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.root_path_var).grid(
            row=0, column=5, sticky="ew", padx=(8, 6)
        )
        ttk.Button(controls, text="Browse", command=self._browse_root_path).grid(
            row=0, column=6, sticky="w"
        )

        ttk.Label(controls, text="Max sentences (Task 1/2)").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.max_sentences_var, width=14).grid(
            row=1, column=1, sticky="w", padx=(8, 12), pady=(8, 0)
        )

        ttk.Label(controls, text="Min freq (Task 1/2)").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.min_freq_var, width=10).grid(
            row=1, column=3, sticky="w", padx=(8, 12), pady=(8, 0)
        )

        ttk.Label(controls, text="Max docs (Task 4)").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.max_docs_var, width=12).grid(
            row=1, column=5, sticky="w", padx=(8, 12), pady=(8, 0)
        )

        ttk.Label(controls, text="Max examples (Task 4)").grid(row=1, column=6, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.max_examples_var, width=12).grid(
            row=1, column=7, sticky="w", padx=(8, 12), pady=(8, 0)
        )

        ttk.Label(controls, text="Max vocab tokens (Task 4)").grid(row=1, column=8, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.max_vocab_tokens_var, width=12).grid(
            row=1, column=9, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        actions = ttk.Frame(self, padding=(10, 0, 10, 8))
        actions.grid(row=1, column=0, sticky="new")
        actions.columnconfigure(0, weight=1)

        button_row = ttk.Frame(actions)
        button_row.pack(anchor="w", pady=(0, 8))
        self._register_action_button(
            ttk.Button(button_row, text="Run Task 1", command=self._on_run_task1)
        ).pack(side="left", padx=(0, 8))
        self._register_action_button(
            ttk.Button(button_row, text="Run Task 2", command=self._on_run_task2)
        ).pack(side="left", padx=(0, 8))
        self._register_action_button(
            ttk.Button(button_row, text="Run Task 3", command=self._on_run_task3)
        ).pack(side="left", padx=(0, 8))
        self._register_action_button(
            ttk.Button(button_row, text="Run Task 4", command=self._on_run_task4)
        ).pack(side="left", padx=(0, 8))
        self._register_action_button(
            ttk.Button(button_row, text="Run All", command=self._on_run_all)
        ).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Clear Results", command=self._clear_results).pack(side="left")

        tabs_container = ttk.Frame(actions)
        tabs_container.pack(fill="both", expand=True)
        tabs_container.columnconfigure(0, weight=1)
        tabs_container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(tabs_container)
        notebook.grid(row=0, column=0, sticky="nsew")

        for task_name in ["Task 1", "Task 2", "Task 3", "Task 4"]:
            tab = ttk.Frame(notebook, padding=6)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

            table = ttk.Treeview(tab, columns=("metric", "value"), show="headings")
            table.heading("metric", text="Metric")
            table.heading("value", text="Value")
            table.column("metric", width=320, anchor="w")
            table.column("value", width=600, anchor="w")

            y_scroll = ttk.Scrollbar(tab, orient="vertical", command=table.yview)
            table.configure(yscrollcommand=y_scroll.set)

            table.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            notebook.add(tab, text=task_name)
            self._tables[task_name] = table

        status_frame = ttk.Frame(self, padding=(10, 4, 10, 10))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )

    def _register_action_button(self, button: ttk.Button) -> ttk.Button:
        self._action_buttons.append(button)
        return button

    def _browse_news_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Select news corpus file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.news_path_var.set(path)

    def _browse_root_path(self) -> None:
        path = filedialog.askdirectory(title="Select project root directory")
        if path:
            self.root_path_var.set(path)

    def _parse_positive_int(self, raw_value: str, field_name: str) -> int:
        raw = raw_value.strip()
        if not raw:
            raise ValueError(f"{field_name} is required.")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return value

    def _get_news_path(self) -> Path:
        news_path = Path(self.news_path_var.get().strip())
        if not news_path.exists():
            raise ValueError(f"News corpus not found: {news_path}")
        return news_path

    def _get_root_path(self) -> Path:
        root_path = Path(self.root_path_var.get().strip())
        if not root_path.exists():
            raise ValueError(f"Project root not found: {root_path}")
        return root_path

    def _get_lm_params(self) -> Dict[str, Any]:
        return {
            "news_path": self._get_news_path(),
            "max_sentences": self._parse_positive_int(
                self.max_sentences_var.get(), "Max sentences"
            ),
            "min_freq": self._parse_positive_int(self.min_freq_var.get(), "Min freq"),
        }

    def _get_task4_params(self) -> Dict[str, Any]:
        return {
            "news_path": self._get_news_path(),
            "max_docs": self._parse_positive_int(self.max_docs_var.get(), "Max docs"),
            "max_examples": self._parse_positive_int(
                self.max_examples_var.get(), "Max examples"
            ),
            "max_vocab_tokens": self._parse_positive_int(
                self.max_vocab_tokens_var.get(), "Max vocab tokens"
            ),
        }

    def _set_busy(self, busy: bool, status_text: str) -> None:
        self._running = busy
        new_state = "disabled" if busy else "normal"
        for button in self._action_buttons:
            button.config(state=new_state)
        self.status_var.set(status_text)

    def _execute_job(
        self,
        label: str,
        job: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if self._running:
            messagebox.showinfo("Task running", "Please wait for the current run to finish.")
            return

        self._set_busy(True, f"Running {label}...")

        def worker() -> None:
            try:
                result = job()
            except Exception as exc:  # pragma: no cover - UI-only branch
                self.after(0, lambda: self._on_job_error(label, exc))
                return
            self.after(0, lambda: self._on_job_success(label, result, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _on_job_success(
        self,
        label: str,
        result: Any,
        on_success: Callable[[Any], None],
    ) -> None:
        on_success(result)
        self._set_busy(False, f"{label} completed.")

    def _on_job_error(self, label: str, exc: Exception) -> None:
        self._set_busy(False, f"{label} failed.")
        messagebox.showerror("Execution error", f"{label} failed:\n{exc}")

    def _set_table_metrics(self, task_name: str, metrics: Mapping[str, Any]) -> None:
        table = self._tables[task_name]
        for row_id in table.get_children():
            table.delete(row_id)
        for key, value in metrics.items():
            table.insert("", "end", values=(key, format_metric_value(value)))

    def _clear_results(self) -> None:
        for table in self._tables.values():
            for row_id in table.get_children():
                table.delete(row_id)
        self.status_var.set("Results cleared.")

    def _on_run_task1(self) -> None:
        try:
            params = self._get_lm_params()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._execute_job(
            "Task 1",
            lambda: run_task1(**params),
            lambda metrics: self._set_table_metrics("Task 1", metrics),
        )

    def _on_run_task2(self) -> None:
        try:
            params = self._get_lm_params()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        def job() -> Dict[str, Any]:
            metrics = run_task2(**params)
            metrics = dict(metrics)
            metrics["best_bigram_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_BIGRAM, key=lambda key: metrics[key]
            )
            metrics["best_trigram_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_TRIGRAM, key=lambda key: metrics[key]
            )
            metrics["best_overall_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_BIGRAM + TASK2_SMOOTH_KEYS_TRIGRAM,
                key=lambda key: metrics[key],
            )
            return metrics

        self._execute_job(
            "Task 2",
            job,
            lambda metrics: self._set_table_metrics("Task 2", metrics),
        )

    def _on_run_task3(self) -> None:
        try:
            root = self._get_root_path()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._execute_job(
            "Task 3",
            lambda: run_task3(root),
            lambda metrics: self._set_table_metrics("Task 3", metrics),
        )

    def _on_run_task4(self) -> None:
        try:
            params = self._get_task4_params()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._execute_job(
            "Task 4",
            lambda: run_task4(**params),
            lambda metrics: self._set_table_metrics("Task 4", metrics),
        )

    def _on_run_all(self) -> None:
        try:
            lm_params = self._get_lm_params()
            root = self._get_root_path()
            task4_params = self._get_task4_params()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        def job() -> Dict[str, Dict[str, Any]]:
            t12 = run_task1_task2(**lm_params)
            task1 = {k: t12[k] for k in TASK1_KEYS if k in t12}
            task2 = {k: t12[k] for k in TASK2_KEYS if k in t12}
            task2["best_bigram_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_BIGRAM, key=lambda key: task2.get(key, float("inf"))
            )
            task2["best_trigram_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_TRIGRAM, key=lambda key: task2.get(key, float("inf"))
            )
            task2["best_overall_smoothing_by_ppl"] = min(
                TASK2_SMOOTH_KEYS_BIGRAM + TASK2_SMOOTH_KEYS_TRIGRAM,
                key=lambda key: task2.get(key, float("inf")),
            )
            task3 = run_task3(root)
            task4 = run_task4(**task4_params)
            return {
                "Task 1": task1,
                "Task 2": task2,
                "Task 3": task3,
                "Task 4": task4,
            }

        def on_success(results: Dict[str, Dict[str, Any]]) -> None:
            for task_name, metrics in results.items():
                self._set_table_metrics(task_name, metrics)

        self._execute_job("All tasks", job, on_success)


def main() -> None:
    app = ResultsUI()
    app.mainloop()


if __name__ == "__main__":
    main()
