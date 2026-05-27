# Automated Prompt Optimization for Structured Extraction

A production-quality research engineering system that automatically optimizes LLM prompts for structured field extraction tasks — running entirely locally via **Ollama**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI / Entrypoint                         │
│              python -m app.main [command]                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │        Pipeline Orchestrator   │
        │           app/main.py          │
        └──┬────────┬────────┬──────────┘
           │        │        │
    ┌──────▼──┐ ┌───▼────┐ ┌─▼──────────┐
    │ Dataset │ │  LLM   │ │ Persistence │
    │ Loader  │ │ Client │ │  DB+Cache   │
    └──────┬──┘ └───┬────┘ └─────┬──────┘
           │        │             │
    ┌──────▼────────▼─────────────▼──────┐
    │           Optimizer Loop            │
    │  Beam Search / Greedy / Population  │
    │                                     │
    │  seed → mutate → evaluate → accept  │
    └──────────────┬──────────────────────┘
                   │
           ┌───────▼────────┐
           │  Evaluator /   │
           │ Scoring Engine │
           └───────┬────────┘
                   │
           ┌───────▼────────┐
           │ Report Generator│
           │ Plots + Diffs   │
           └────────────────┘
```

### Key Components

| Module | Purpose |
|--------|---------|
| `app/llm/ollama_client.py` | Ollama REST client with retry, tracking |
| `app/datasets/loader.py` | Load JSON datasets or generate synthetic data |
| `app/extraction/extractor.py` | Apply prompts to documents via LLM |
| `app/scoring/metrics.py` | ExtractBench-compatible scoring (exact, semantic, numeric, array) |
| `app/optimizer/beam_search.py` | Beam search & greedy optimizer |
| `app/optimizer/population.py` | Population-based evolutionary optimizer |
| `app/optimizer/mutation.py` | 8 mutation strategies |
| `app/persistence/database.py` | SQLAlchemy ORM — all runs persisted |
| `app/persistence/cache.py` | Disk-backed LLM response cache |
| `app/reporting/report_generator.py` | Markdown reports with plots |

---

## Setup

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com
```

### 2. Pull Models

```bash
ollama pull mistral
ollama pull llama3
ollama pull qwen2.5
```

### 3. Start Ollama Server

```bash
ollama serve
# Runs at http://localhost:11434
```

### 4. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (default Ollama URL: http://localhost:11434)
```

---

## Configuration

All behavior is controlled through YAML files in `configs/`. **Zero code changes needed** to switch datasets, models, or strategies.

### `configs/base.yaml`
Top-level settings: experiment name, Ollama URL, DB path, pipeline mode.

### `configs/datasets.yaml`
Define dataset paths and field schemas. Add your own:
```yaml
my_dataset:
  type: extractbench
  path: ./data/raw/my_data.json
  input_field: document_text
  fields_to_extract:
    - name: company_name
      type: string_semantic
      required: true
    - name: revenue
      type: number_tolerance
      tolerance: 0.02
```

### `configs/models.yaml`
Configure Ollama models, temperature, and token limits.

### `configs/optimizer.yaml`
Choose optimizer strategy and mutation mix:
```yaml
optimizer:
  beam:
    strategy: beam
    beam_width: 3
    max_iterations: 8
    mutation_strategies:
      - instruction_rewrite
      - hallucination_suppress
      - few_shot_insert
```

---

## Usage

### Run Optimization
```bash
python -m app.main optimize
python -m app.main optimize --dataset invoice_bench --model qwen2_5 --optimizer beam
python -m app.main optimize --budget 50 --experiment-name my_run
```

### Evaluate a Prompt
```bash
python -m app.main evaluate --prompt-file ./my_prompt.txt --split test
```

### Resume Interrupted Run
```bash
python -m app.main resume --experiment-name my_run
```

### Generate Report
```bash
python -m app.main report --experiment-id 3
```

### List All Experiments
```bash
python -m app.main list-experiments
```

---

## Supported Mutation Strategies

| Strategy | Description |
|----------|-------------|
| `instruction_rewrite` | LLM rewrites the full instruction block |
| `output_format_tighten` | Tightens JSON output formatting rules |
| `verbosity_reduce` | Reduces prompt length by ~30% |
| `hallucination_suppress` | Adds anti-hallucination instructions |
| `schema_aware_refine` | Adds field type hints and format examples |
| `field_constraint_add` | Targets the weakest fields with specific rules |
| `chain_of_thought_toggle` | Adds/removes step-by-step reasoning |
| `few_shot_insert` | Injects few-shot examples from training set |

---

## Scoring Metrics (ExtractBench-Compatible)

| Field Type | Scoring Method |
|-----------|----------------|
| `string_exact` | Case-insensitive exact match |
| `string_semantic` | RapidFuzz token sort ratio |
| `integer_exact` | Exact integer comparison |
| `number_tolerance` | Relative tolerance match |
| `array_llm` | Greedy bipartite alignment + F1 |

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

---

## Output Structure

```
runs/
  experiments.db          ← all experiment records (SQLite)
  my_run_best_prompt.txt  ← best prompt found

reports/
  my_run_20240115_120000/
    report.md             ← full Markdown report
    plots/
      score_curve.png
      per_field_scores.png
      strategy_performance.png

logs/
  app.log

data/cache/               ← LLM response cache (disk-backed)
```

---

## Experiment Explanation

The optimizer starts from a **seed prompt** (auto-generated from field schema), evaluates it on a validation subset, then iteratively:

1. **Mutates** the current best prompt using one of 8 strategies (via LLM)
2. **Evaluates** the mutated prompt on a fresh validation sample
3. **Accepts or rejects** based on the configured acceptance policy
4. **Records** every decision, score, and prompt to SQLite
5. **Checkpoints** state for resumability
6. **Reports** with score curves, diffs, and per-field breakdowns

The best prompt is evaluated on a held-out **test set** at the end.

---

## Screenshots

> _Place `score_curve.png`, `per_field_scores.png`, and `strategy_performance.png` here after your first run._

---

## License

MIT
