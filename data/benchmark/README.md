---
pretty_name: Semantic Monte Carlo Benchmark
language:
- en
license: mit
size_categories:
- n<1K
task_categories:
- question-answering
language_creators:
- machine-generated
annotations_creators:
- other
source_datasets:
- original
tags:
- synthetic
- benchmark
- forecasting
- numerical-reasoning
- web-search
- uncertainty-estimation
- tabular
- datasets
configs:
- config_name: default
  data_files:
  - split: validation
    path: eval.csv
  - split: test
    path: test.csv
---

# Semantic Monte Carlo Benchmark

An English-language benchmark of numeric questions for evaluating the
[`semantic-montecarlo`](https://github.com/cynosural-ai/semantic-montecarlo)
pipeline. The pipeline researches each question with a web-enabled language
model, produces multiple numeric estimates, and aggregates them into a
probability distribution.

This repository release contains only the benchmark inputs. Cached experiments,
individual run artifacts, and aggregate run results are not part of the dataset.

## Dataset structure

The dataset has no training split:

| Split | Source file | Rows | Intended use |
| --- | --- | ---: | --- |
| `validation` | `eval.csv` | 20 | Parameter selection and development |
| `test` | `test.csv` | 280 | Final benchmark evaluation |

Both splits are balanced across the ten `confidence_mean` levels: the
validation split contains two questions per level, and the test split contains
28 questions per level.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | integer | Row identifier, unique within each split |
| `confidence_mean` | integer | Target expected-confidence level from 5 to 95, expressed as a percentage |
| `domain` | string | Topic category for the question |
| `question` | string | Numeric question to research or forecast |
| `answer_unit` | string | Unit required for the numeric estimate |

`confidence_mean` is the benchmark target used by the current scoring code. It
is not a model prediction, an observed frequency, or a guarantee that the
answer is correct. The method originally used to assign these target levels is
not recorded in this repository and should be treated as a provenance
limitation.

## Data creation and provenance

The questions were synthetically generated with a large language model and
then organized into validation and test splits by project contributors. The
exact generation model and revision, prompt, and human-review procedure were
not retained in this repository. The question set was first committed to the
project in July 2026.

The dataset does not contain source documents or personal user data. Questions
refer to public topics such as economics, companies, climate, sports, and
long-range forecasts.

## Benchmark protocol

The reference implementation is
[`scripts/benchmark.py`](https://github.com/cynosural-ai/semantic-montecarlo/blob/main/scripts/benchmark.py).
For each test question, it:

1. Runs the pipeline using the question and `answer_unit`.
2. Converts the resulting bootstrap-mean distribution into an estimated
   confidence with
   [`norm_var_comp`](https://github.com/cynosural-ai/semantic-montecarlo/blob/main/semantic_montecarlo/stats/norm_var_comp.py).
3. Converts `confidence_mean` to the interval `[0, 1]` by dividing it by 100.
4. Reports mean squared error between expected and estimated confidence.

This protocol evaluates alignment with the dataset's confidence targets. It
does not evaluate numeric answer accuracy because the dataset does not provide
resolved numeric answers.

For comparable results, a benchmark report should record at least the dataset
revision, code commit, run timestamp, model identifier, prompt/search
configuration, number of paraphrases, number of bootstrap resamples, random
seed, token and search usage, failures, and retries.

## Usage

After publication on the Hugging Face Hub:

```python
from datasets import load_dataset

dataset = load_dataset("cynosural-ai/semantic-montecarlo-benchmark")
validation = dataset["validation"]
test = dataset["test"]
```

Use `validation` while choosing parameters. Reserve `test` for the final
reported evaluation.

## Intended uses

- Evaluate confidence distributions produced by web-enabled numeric research
  pipelines.
- Compare pipeline configurations under a fixed dataset revision and recorded
  execution protocol.
- Study how question horizon and domain relate to distribution concentration
  and no-answer behavior.

The dataset is not intended as a factual answer key, a calibrated probability
dataset, or training data for optimizing against the published test questions.

## Limitations

- The questions are LLM-generated and may contain ambiguities, incorrect
  premises, or generator biases.
- The provenance of the `confidence_mean` assignments is incomplete.
- Many questions are time-dependent or refer to future events. Web evidence and
  pipeline outputs therefore change with the execution date.
- The test questions are public, so repeated tuning against them can invalidate
  claims of held-out evaluation and allow benchmark contamination.
- Confidence-target MSE does not measure whether the final numeric estimate is
  factually correct.

## Citation

```bibtex
@dataset{cynosural_ai_semantic_montecarlo_2026,
  title = {Semantic Monte Carlo Benchmark},
  author = {{Cynosural AI contributors}},
  year = {2026},
  url = {https://github.com/cynosural-ai/semantic-montecarlo}
}
```

## License

The benchmark data is released under the [MIT License](LICENSE).
