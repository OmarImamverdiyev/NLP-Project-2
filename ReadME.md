## NLP Project 2

This repository supports running each assignment task independently.
The sections below map each task number to the files that implement it and what they do.

Datasets:
1. https://drive.google.com/drive/folders/1zPD4m9aAH4JH1urBDdPm4RBkOWXLnVt4?usp=sharing
2. https://github.com/AzTextCorpus/az-sentiment-analysis-dataset
3. https://drive.google.com/file/d/1-uEH72U_oDzj36H6gvBvvq2fVMuB6gNX/view?usp=drive_link

### Run tasks separately

```powershell
python Task1/run_task1.py
python Task2/run_task2.py
python Task3/run_task3.py
python Task4/run_task4.py
```

### Run results UI (Task 1-4 dashboard)

```powershell
python results_ui.py
```

The UI lets you:
- run Task 1, Task 2, Task 3, Task 4 individually, or run all tasks together;
- adjust key parameters (`max-sentences`, `min-freq`, Task 4 caps, paths);
- view each task's metrics in separate tabs.

### Optional arguments

Task 1 and Task 2:

```powershell
python Task1/run_task1.py --max-sentences 50000 --min-freq 2
python Task2/run_task2.py --max-sentences 50000 --min-freq 2
```

Task 3:

```powershell
python Task3/run_task3.py --root .
```

Run Task 3 explicitly on the reduced dataset:

```powershell
python Task3/run_task3.py --dataset-path sentiment_dataset/dataset_v1.csv
```

If Task 3 runs out of memory, cap the dataset size:

```powershell
python Task3/run_task3.py --max-samples 5000
```

Note: `run_task3.py` now defaults to `sentiment_dataset/dataset_v1.csv` when present (otherwise `dataset.csv`).  
It uses a stratified train/dev/test split by default (`test_ratio=0.2`, `dev_ratio_within_train=0.2`).
When `sklearn` is not available, Task 3 defaults to `5000` samples to avoid dense-matrix memory errors.

Task 4:

```powershell
python Task4/run_task4.py --news-path Corpora/News/corpus.txt
```

If memory is limited for Task 4, reduce sample and vocabulary caps:

```powershell
python Task4/run_task4.py --max-docs 20000 --max-examples 30000 --max-vocab-tokens 4000
```

### Task-to-file map

#### Task 1: N-gram language modeling perplexity
- `Task1/run_task1.py`: Task 1 CLI entry point.
- `core/language_modeling.py` (`run_task1`): Loads the news corpus, tokenizes/splits sentences, builds unigram-bigram-trigram counts, and reports MLE perplexity.
- `core/text_utils.py`: Shared sentence splitting and tokenization helpers.
- `core/paths.py`: Provides default corpus path (`Corpora/News/corpus.txt`).
- `core/reporting.py`: Prints task metrics in a consistent format.

#### Task 2: Trigram smoothing comparison
- `Task2/run_task2.py`: Task 2 CLI entry point.
- `core/language_modeling.py` (`run_task2`): Reuses LM preparation and compares smoothing methods (Laplace, interpolation, backoff, Kneser-Ney) by perplexity.
- `core/text_utils.py`: Shared sentence splitting and tokenization.
- `core/paths.py`: Provides default corpus path (`Corpora/News/corpus.txt`).
- `core/reporting.py`: Prints task metrics and summary output.

#### Task 3: Sentiment classification
- `Task3/run_task3.py`: Task 3 CLI entry point.
- `core/sentiment_task.py` (`run_task3`): Loads `sentiment_dataset/dataset_v1.csv` when available (else `dataset.csv`), creates BoW and lexicon features, trains Multinomial NB, Bernoulli NB, and Logistic Regression, and reports dev/test metrics plus McNemar tests.
- `core/ml.py`: Contains model implementations and evaluation/statistical testing utilities.
- `core/text_utils.py`: Tokenization utilities for feature construction.
- `core/paths.py`: Defines default sentiment dataset path (prefers `sentiment_dataset/dataset_v1.csv`, falls back to `sentiment_dataset/dataset.csv`).

#### Task 4: Dot-as-sentence-boundary detection
- `Task4/run_task4.py`: Task 4 CLI entry point.
- `core/sentence_boundary_task.py` (`run_task4`): Extracts dot-context examples, vectorizes contextual features, trains Logistic Regression with L1 and L2 regularization, compares them, and reports metrics.
- `core/ml.py`: Contains logistic regression, train-test split, metrics, and McNemar test.
- `core/paths.py`: Provides default news corpus path (`Corpora/News/corpus.txt`).
- `core/reporting.py`: Prints task metrics.

### Train/dev/test splitting

- Task 1 and Task 2 split news sentences into train/dev/test = 80%/10%/10% in `core/language_modeling.py` (`train_dev_test_split`).
- Task 3 uses stratified train/dev/test splits in `core/sentiment_task.py` and `Task3/tune_task3.py`: test is 20% of all data; dev is 20% of the remaining train pool (overall 64%/16%/20%).
- Task 4 uses train/dev/test = 72%/8%/20% in `core/sentence_boundary_task.py` (`split_train_dev_test_xy` with test ratio 0.2 and dev as 10% of the remaining train pool).
- Splits are randomized with seed `42` defined in `core/paths.py`.

### Shared runner

- `assignment_tasks.py`: Runs Task 1+2, Task 3, and Task 4 in one execution and prints the best Task 2 smoothing method.

### Data files relevant to tasks

- `Corpora/News/corpus.txt`: Main input corpus for Task 1, Task 2, and Task 4.
- `sentiment_dataset/dataset.csv`: Required and only dataset source for Task 3.
- `Corpora/News/build_corpus.py`, `Corpora/News/content_only.csv`, and files under `Corpora/Youtube/`: Corpus preparation/cleaning assets (useful for data building; not required for normal task execution if corpus files already exist).

### Run all tasks together

```powershell
python assignment_tasks.py
```

### Tune Task 4 (how accuracy was improved)

```powershell
python Task4/tune_task4.py
```

Save results
```powershell
python Task4/tune_task4.py --news-path Corpora/News/corpus.txt --save-json Task4/tuning_results.json
```

Save results (smaller version)
```powershell
python Task4/tune_task4.py --max-docs 20000 --max-examples 30000 --max-vocab-tokens 4000 --save-json Task4/tuning_results.json
```

Optional save:

```powershell
python Task4/tune_task4.py --save-json Task4/tuning_results.json
```

python Task4/tune_task4.py --search-mode extended --save-json Task4/tuning_results.json

### Tune Task 3 (sentiment)

```powershell
python Task3/tune_task3.py
```

Memory-friendly tuning on a smaller subset:

```powershell
python Task3/tune_task3.py --max-samples 10000
```

Run tuning explicitly on the reduced dataset:

```powershell
python Task3/tune_task3.py --dataset-path sentiment_dataset/dataset_v1.csv
```

Optional save:

```powershell
python Task3/tune_task3.py --search-mode extended --save-json Task3/tuning_results.json
```

Task 3 tuning defaults to `sentiment_dataset/dataset_v1.csv` when present (else `dataset.csv`) and uses a stratified train/dev/test workflow (dev for model selection, test for final report).  
When `--max-samples` is used, sampling is stratified and deterministic with seed `42`, so repeated runs with the same arguments are stable.
