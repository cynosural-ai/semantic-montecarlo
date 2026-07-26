# semantic-montecarlo
Method for sampling agents to estimate probability distributions

## Usage

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
uv run semantic-montacarlo
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

The distribution is printed to stdout:

```json
{
  "data": [[215.9, 1.0]],
  "no_answer_probability": 0.2
}
```

`no_answer_probability` is the sampled frequency of missing answers. Numeric
probabilities are conditional on receiving an answer and therefore sum to 1.

### Run artifacts

Every run is also persisted to a timestamped directory under `--output-dir`
(default `outputs/`):

```
outputs/20260725_183000/
├── run.log        # captured pipeline logs
├── result.json    # metadata + per-stage token usage + the distribution
└── searches.json  # per-paraphrase answers (reasoning, value, confidence, sources)
```

`result.json` records the model, elapsed time, CLI parameters, and a `usage`
breakdown (`paraphrase` / `search` / `total` token counts). `searches.json`
holds the provenance behind the distribution — the reasoning, sources, and
confidence for each numeric answer. Redirect the directory with
`--output-dir path/to/runs`; raise log verbosity with `--log-level DEBUG`.
