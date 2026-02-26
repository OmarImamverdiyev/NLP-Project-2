# NLP Project 2

This repository contains four NLP assignment tasks with CLI runners, tuning scripts, and local UIs.

## What Is Included

- Task 1: n-gram language modeling perplexity
- Task 2: bigram/trigram smoothing comparison
- Task 3: sentiment classification
- Task 4: dot sentence-boundary detection (v1 and v2 paths)
- `assignment_tasks.py`: run all tasks in one command
- `results_ui.py` and `localhost_ui.py`: local dashboards

## Requirements

- Python 3.10+
- Recommended packages:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn scipy streamlit
```

Notes:
- `scikit-learn`/`scipy` are required for Task 4 v2 and Streamlit demos.
- Task 3 can fall back to custom models if `scikit-learn` is unavailable.

## Data Paths

- Task 1/2: `Corpora/News/corpus.txt`
- Task 3: `sentiment_dataset/dataset_v1.csv` (fallback to `sentiment_dataset/dataset.csv`)
- Task 4 v2 / orchestrator default: `dot_labeled_data.csv`

## Quick Start

Run all tasks with defaults:

```powershell
python assignment_tasks.py
```

Useful options:

```powershell
python assignment_tasks.py --skip-task4
python assignment_tasks.py --task4-dataset dot_labeled_data.csv --max-examples 60000
python assignment_tasks.py --max-sentences 120000 --min-freq 2
```

## Run Tasks Individually

```powershell
python Task1/run_task1.py
python Task2/run_task2.py
python Task3/run_task3.py
python Task4/run_task4_v2.py
```

> Note run task_v2 not first version for better results.

### Task 1: N-gram LM Perplexity

```powershell
python Task1/run_task1.py --news-path Corpora/News/corpus.txt --max-sentences 120000 --min-freq 2
python Task1/run_task1.py --output Task1/custom_task1_results.txt
```

Default output: `Task1/task1_results.txt`

### Task 2: Smoothing Comparison

```powershell
python Task2/run_task2.py --news-path Corpora/News/corpus.txt --max-sentences 120000 --min-freq 2
python Task2/run_task2.py --output Task2/custom_task2_results.txt
python Task2/run_task2.py --txt-dir Task2/custom_txt
```

Default outputs:
- `Task2/task2_results.txt`
- `Task2/txt/bigrams_laplace.txt`
- `Task2/txt/bigrams_interpolation.txt`
- `Task2/txt/bigrams_backoff.txt`
- `Task2/txt/bigrams_kneser_ney.txt`
- `Task2/txt/trigrams_laplace.txt`
- `Task2/txt/trigrams_interpolation.txt`
- `Task2/txt/trigrams_backoff.txt`
- `Task2/txt/trigrams_kneser_ney.txt`

### Task 3: Sentiment Classification

```powershell
python Task3/run_task3.py
python Task3/run_task3.py --dataset-path sentiment_dataset/dataset_v1.csv --max-samples 0
python Task3/run_task3.py --max-samples 5000
```

Defaults:
- Auto dataset selection prefers `dataset_v1.csv`, then falls back to `dataset.csv`.
- If `scikit-learn` is unavailable, default memory cap is `--max-samples 5000`.


### Task 4: Sentence Boundary

```powershell
python Task4/run_task4_v2.py --dataset dot_labeled_data.csv
python Task4/tune_task4_v2.py --dataset dot_labeled_data.csv
```

## Tuning

### Tune Task 3

```powershell
python Task3/tune_task3.py --search-mode extended --selection-metric dev_macro_f1
python Task3/tune_task3.py --dataset-path sentiment_dataset/dataset_v1.csv --max-samples 0 --save-json Task3/tuning_task3_v1.json
```

### Tune Task 4 

```powershell
python Task4/tune_task4_v2.py --search-mode extended --selection-metric dev_accuracy
python Task4/tune_task4_v2.py --save-json Task4/tuning_results.json
```

## Local UIs

Desktop (Tkinter):

```powershell
python results_ui.py
```

Web UI (Streamlit):

```powershell
streamlit run localhost_ui.py
```

Then open: `http://localhost:8501`

## Main Files

- `Task1/run_task1.py`: Task 1 CLI
- `Task2/run_task2.py`: Task 2 CLI
- `Task3/run_task3.py`: Task 3 CLI
- `Task4/run_task4_v2.py`: Task 4 v2 CLI
- `assignment_tasks.py`: combined runner
- `results_ui.py`: desktop dashboard
- `localhost_ui.py`: Streamlit dashboard + demos
