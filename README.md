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
├── run.log        # captured pipeline logs
├── result.json    # metadata + token usage + both distributions
└── searches.json  # per-paraphrase answers (reasoning, value, confidence, sources)
```

`result.json` records the model, elapsed time, CLI parameters, and a `usage`
breakdown (`paraphrase` / `search` / `total` token counts), alongside both
distributions. `searches.json` holds the provenance behind them — the
reasoning, sources, and confidence for each numeric answer. Redirect the directory with
`--output-dir path/to/runs`; raise log verbosity with `--log-level DEBUG`.
