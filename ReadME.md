## NLP Project 2

This repository supports running each assignment task independently.
The sections below map each task number to the files that implement it and what they do.

 
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

### Run localhost web UI (results + simple demos)

```powershell
streamlit run localhost_ui.py
```

Then open `http://localhost:8501`.

This web UI includes:
- Results tab for Task 1-4 metrics (individual run buttons + run-all);
- Unigram/Bigram tab to test a custom sentence, reuse a precomputed example output, and compare trigram smoothing (Laplace, interpolation, backoff, Kneser-Ney);
- Sentiment tab to classify your own text with a lightweight demo model;
- Sentence Boundary tab to test dot (`.`) boundary predictions on custom text.

### Optional arguments

Task 1 and Task 2:

```powershell
python Task1/run_task1.py --max-sentences 50000 --min-freq 2
python Task2/run_task2.py --max-sentences 50000 --min-freq 2
```

By default, these commands also save metrics to:
- `Task1/task1_results.txt`
- `Task2/task2_results.txt`
- `Task2/txt/bigrams_laplace.txt`
- `Task2/txt/bigrams_interpolation.txt`
- `Task2/txt/bigrams_backoff.txt`
- `Task2/txt/bigrams_kneser_ney.txt`
- `Task2/txt/trigrams_laplace.txt`
- `Task2/txt/trigrams_interpolation.txt`
- `Task2/txt/trigrams_backoff.txt`
- `Task2/txt/trigrams_kneser_ney.txt`

You can override output path:

```powershell
python Task1/run_task1.py --output Task1/custom_task1_metrics.txt
python Task2/run_task2.py --output Task2/custom_task2_metrics.txt
python Task2/run_task2.py --txt-dir Task2/custom_txt
```

Task 3:

```powershell
python Task3/run_task3.py --root .
```

python Task3\run_task3.py --dataset-path sentiment_dataset\dataset_v1.csv --max-samples 0

python Task3\tune_task3.py --dataset-path sentiment_dataset\dataset_v1.csv --max-samples 0 --search-mode extended --selection-metric dev_macro_f1 --save-json Task3\tuning_task3_v1.json


If Task 3 runs out of memory, cap the dataset size:

```powershell
python Task3/run_task3.py --max-samples 5000
```

Note: when `sklearn` is not available, Task 3 now defaults to `5000` samples to avoid dense-matrix memory errors.

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

#### Task 2: Bigram + trigram smoothing comparison
- `Task2/run_task2.py`: Task 2 CLI entry point.
- `core/language_modeling.py` (`run_task2`): Reuses LM preparation, compares Laplace/interpolation/backoff/Kneser-Ney for both bigram and trigram perplexity, tunes interpolation lambdas on dev data, and exports per-method bigram/trigram probability tables.
- `core/text_utils.py`: Shared sentence splitting and tokenization.
- `core/paths.py`: Provides default corpus path (`Corpora/News/corpus.txt`).
- `core/reporting.py`: Prints task metrics and summary output.

#### Task 3: Sentiment classification
- `Task3/run_task3.py`: Task 3 CLI entry point.
- `core/sentiment_task.py` (`run_task3`): Loads only `sentiment_dataset/dataset.csv`, creates BoW and lexicon features, trains Multinomial NB, Bernoulli NB, and Logistic Regression, and reports metrics plus McNemar tests.
- `core/ml.py`: Contains model implementations and evaluation/statistical testing utilities.
- `core/text_utils.py`: Tokenization utilities for feature construction.
- `core/paths.py`: Defines default sentiment dataset path (`sentiment_dataset/dataset.csv`).

#### Task 4: Dot-as-sentence-boundary detection
- `Task4/run_task4.py`: Task 4 CLI entry point.
- `core/sentence_boundary_task.py` (`run_task4`): Extracts dot-context examples, vectorizes contextual features, trains Logistic Regression with L1 and L2 regularization, compares them, and reports metrics.
- `core/ml.py`: Contains logistic regression, train-test split, metrics, and McNemar test.
- `core/paths.py`: Provides default news corpus path (`Corpora/News/corpus.txt`).
- `core/reporting.py`: Prints task metrics.

### Train/dev/test splitting

- Task 1 and Task 2 split news sentences into train/dev/test = 80%/10%/10% in `core/language_modeling.py` (`train_dev_test_split`).
- Task 3 uses a stratified train/test split of 80%/20% in `core/sentiment_task.py`.
- Task 4 uses train/dev/test = 72%/8%/20% in `core/sentence_boundary_task.py` (`split_train_dev_test_xy` with test ratio 0.2 and dev as 10% of the remaining train pool).
- Splits are randomized with seed `42` defined in `core/paths.py`.

### Shared runner

- `assignment_tasks.py`: Runs Task 1+2, Task 3, and Task 4 in one execution and prints best Task 2 smoothing by bigram, trigram, and overall perplexity.

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

```powershell
python Task3/run_task3.py --dataset-path sentiment_dataset/dataset_v1.csv --max-samples 0
```

python Task3/tune_task3.py --dataset-path sentiment_dataset/dataset_v1.csv --max-samples 0

Optional save:

```powershell
python Task3/tune_task3.py --search-mode extended --save-json Task3/tuning_results.json
```

Task 3 tuning uses a stratified train/dev/test workflow (dev for model selection, test for final report).  
When `--max-samples` is used, sampling is stratified and deterministic with seed `42`, so repeated runs with the same arguments are stable.
