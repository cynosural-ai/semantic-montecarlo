# semantic-montecarlo
Method for sampling agents to estimate probability distributions

## Usage

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
uv run semantic-montecarlo
```

With no question, the CLI samples one row from `data/benchmark/test.csv`
and uses its `answer_unit`. Use a seed to repeat the same selection:

```bash
uv run semantic-montecarlo \
  --paraphrases 1 \
  --resamples 100 \
  --seed 42
```

For a question outside the benchmark dataset, provide its unit explicitly:

```bash
uv run semantic-montecarlo \
  "What was the population of France in 2023?" \
  --unit people
```

The empirical and bootstrap-mean distributions are printed to stdout:

```json
{
  "empirical_distribution": {
    "data": [[215.9, 1.0]],
    "no_answer_probability": 0.2
  },
  "bootstrap_mean_distribution": {
    "data": [[215.9, 1.0]],
    "no_answer_probability": 0.0
  }
}
```

The empirical distribution describes the researched values, assigning missing
answers probability by their observed frequency and weighting numeric answers
by confidence. The bootstrap-mean distribution describes the aggregate
estimates produced by repeatedly resampling those answers and calculating each
resample's mean.

In both distributions, numeric probabilities are conditional on receiving an
answer and therefore sum to 1. The bootstrap no-answer probability counts
resamples containing no numeric value.

### Run artifacts

Every run is also persisted to a timestamped directory under `--output-dir`
(default `outputs/`):

```
outputs/20260725_183000/
├── distribution.png  # bootstrap density + empirical estimates + no-answer mass
├── run.log           # captured pipeline logs
├── result.json       # versioned run manifest
└── searches.json     # each paraphrase with its answer and provenance
```

`result.json` is the manifest for the run. It records the schema version,
artifact names, model, elapsed time, CLI parameters, paraphrases, both
distributions, and a `usage` breakdown (`paraphrase` / `search` / `total`
token counts). `searches.json` pairs each paraphrase with its reasoning,
sources, confidence, and numeric answer.

The plot shows a weighted KDE of the bootstrap means, a rug of the empirical
search estimates, the final mean, and its central 90% interval. A single
numeric outcome is drawn as a Dirac-style spike. The no-answer bar is shown
only when the bootstrap-mean no-answer probability is positive.

Redirect the directory with `--output-dir path/to/runs`; raise log verbosity
with `--log-level DEBUG`.
