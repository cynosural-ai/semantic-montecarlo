---
pretty_name: Semantic Monte Carlo Benchmark
language: [en]
license: cc0-1.0
size_categories: [n<1K]
task_categories: [question-answering]
tags: [synthetic, benchmark, forecasting, numerical-reasoning, web-search]
configs: [{config_name: default, data_files: [{split: validation, path: eval.csv}, {split: test, path: test.csv}]}]
---

# Semantic Monte Carlo Benchmark

> A synthetic benchmark of numeric research and forecasting questions for
> evaluating the
> [`semantic-montecarlo`](https://github.com/cynosural-ai/semantic-montecarlo)
> pipeline.

This release contains only benchmark inputs. Cached experiments, individual
run artifacts, and aggregate results are intentionally excluded.

## At a glance

| Questions | Language | Splits | License |
| ---: | --- | --- | --- |
| 300 | English | Validation and test | CC0 1.0 |

## Dataset structure

The dataset has no training split:

| Split | Source file | Rows | Intended use |
| --- | --- | ---: | --- |
| `validation` | `eval.csv` | 20 | Parameter selection and development |
| `test` | `test.csv` | 280 | Final benchmark evaluation |

Both splits are balanced across the ten `confidence_mean` levels. Validation
contains two questions per level; test contains 28.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | integer | Identifier, unique within each split |
| `confidence_mean` | integer | Target expected-confidence level from 5 to 95, expressed as a percentage |
| `domain` | string | Topic category |
| `question` | string | Numeric question to research or forecast |
| `answer_unit` | string | Required unit for the numeric estimate |

`confidence_mean` is the target used by the current benchmark score. It is not
a model prediction, an observed frequency, or a guarantee that the answer is
correct. The original assignment method was not recorded and remains a
provenance limitation.

## Data creation and provenance

The questions were generated with **GPT-5.6 Sol** and organized into validation
and test splits by project contributors. The generation prompt and human-review
procedure were not retained in this repository. The question set was first
committed in July 2026.

The dataset contains no source documents or personal user records. Its
questions cover public topics such as economics, companies, climate, sports,
and long-range forecasts.

## Benchmark protocol

The reference implementation is
[`scripts/benchmark.py`](https://github.com/cynosural-ai/semantic-montecarlo/blob/main/scripts/benchmark.py).
For each test question, it:

1. Runs the pipeline with the question and `answer_unit`.
2. Converts the bootstrap-mean distribution into estimated confidence using
   [`norm_var_comp`](https://github.com/cynosural-ai/semantic-montecarlo/blob/main/semantic_montecarlo/stats/norm_var_comp.py).
3. Converts `confidence_mean` to `[0, 1]` by dividing it by 100.
4. Reports mean squared error between expected and estimated confidence.

The score measures alignment with the benchmark's confidence targets. It does
not measure numeric answer accuracy because resolved numeric answers are not
included.

Comparable benchmark reports should record the dataset revision, code commit,
run timestamp, model identifier, prompt and search configuration, paraphrase
count, bootstrap resamples, random seed, token and search usage, failures, and
retries.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("cynosural-ai/semantic-montecarlo-benchmark")
validation = dataset["validation"]
test = dataset["test"]
```

Use `validation` while choosing parameters. Reserve `test` for the final
reported evaluation.

## Intended use

- Evaluate confidence distributions produced by web-enabled numeric research
  pipelines.
- Compare configurations under a fixed dataset revision and execution
  protocol.
- Study how question horizon and domain relate to distribution concentration
  and no-answer behavior.

This is not a factual answer key, a calibrated probability dataset, or training
data for optimizing against the published test questions.

## Limitations

- LLM-generated questions may contain ambiguities, incorrect premises, or
  generator biases.
- The provenance of the `confidence_mean` assignments is incomplete.
- Many questions are time-dependent or concern future events; available web
  evidence and pipeline outputs change with the execution date.
- The test questions are public. Repeated tuning against them invalidates
  claims of held-out evaluation and can cause benchmark contamination.
- Confidence-target MSE does not establish factual accuracy.

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

To the extent possible under law, the project contributors have dedicated this
benchmark dataset to the public domain under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
It may be copied, modified, and redistributed for any purpose without
conditions.
