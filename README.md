# semantic-montecarlo
Method for sampling agents to estimate probability distributions

## Usage

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
uv run semantic-montecarlo
```

With no question, the CLI samples one row from `data/benchmark/benchmark.csv`
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
