# Sector Trends API

A FastAPI service that turns a free-text sector (e.g. "generative AI", "oil and gas") into two trends for financial-services professionals:

1. **Sector Funding Snapshot** (`GET /trends/funding`): how much capital is flowing into the sector, at which stages, the biggest rounds, and the sector's share of all funding.
2. **Sector M&A Landscape** (`GET /trends/acquisitions`): exit activity over time, who is buying, strategic vs. financial buyers, deal types, payment terms, and disclosed prices.

Pinecone decides *which companies are in the sector* (vector search over organization embeddings); Postgres computes *what happened to them* (aggregations in SQL). An optional two-sentence GPT takeaway summarizes each result.

## Setup

Requires Python 3.10+.

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `.env`** in the project root (the provided file works as-is):

   ```
   OPENAI_API_KEY=...
   PINECONE_API_KEY=...
   PINECONE_INDEX=...
   POSTGRES_DB_NAME=...
   POSTGRES=...            # username (POSTGRES_USER is also accepted)
   POSTGRES_PASSWORD=...
   POSTGRES_HOST=...
   POSTGRES_PORT=...
   ```

4. **Run the server** from the project root

   ```bash
   uvicorn app.main:app --reload
   ```

   Interactive docs: http://localhost:8000/docs

## Endpoints

### `GET /trends/funding`

| Parameter | Required | Description |
|---|---|---|
| `sector` | yes | Free-text sector, 2-200 characters |
| `start_date`, `end_date` | no | Inclusive date range (`YYYY-MM-DD`) |
| `stage` | no | `seed`, `early_stage_venture`, `late_stage_venture`, `private_equity` |
| `include_insight` | no | Add the GPT takeaway (default `true`) |

Returns `summary` (round count, companies funded, total and median raise), `rounds_by_month`, `stage_mix`, `top_deals`, `market_share` (sector share of all rounds and dollars under the same filters), `companies_matched`, `meta`, and `insight`.

### `GET /trends/acquisitions`

| Parameter | Required | Description |
|---|---|---|
| `sector` | yes | Free-text sector, 2-200 characters |
| `start_year`, `end_year` | no | Inclusive announcement-year range |
| `acquirer_type` | no | `company` (strategic) or `investor` (financial) |
| `include_insight` | no | Add the GPT takeaway (default `true`) |

Returns `deal_count`, `deals_by_year`, `top_acquirers`, `buyer_mix`, `deal_types`, `terms`, `price_stats`, `companies_matched`, `meta`, and `insight`. Deals are acquisitions *of* companies in the sector.

### `GET /health`

Checks Postgres, Pinecone, and OpenAI (a model lookup that uses no tokens). Returns `200` when all three are reachable, otherwise `503`.

### Errors

| Status | When |
|---|---|
| `404` `sector_not_found` | No organization is similar enough to the sector text |
| `422` | Invalid or unknown query parameters |
| `503` `upstream_unavailable` | OpenAI, Pinecone, or Postgres failed or timed out |

## Example calls

Contrasting sectors make the trends easy to read:

```bash
curl "http://localhost:8000/trends/funding?sector=generative%20AI"
curl "http://localhost:8000/trends/funding?sector=oil%20and%20gas"
curl "http://localhost:8000/trends/acquisitions?sector=oil%20and%20gas&start_year=2015"
curl "http://localhost:8000/trends/funding?sector=fintech%20payments&stage=seed&start_date=2024-01-01&end_date=2024-06-30&include_insight=false"
```

Generative AI funding is mostly seed rounds (59% of rounds) with a few very large early-stage raises. Oil and gas funding is almost entirely debt and post-IPO rounds (92% of rounds, reported as stage `other`) with a $71M median.

## How it works

```
routes.py ─► service.py ─► repository.py ─► OpenAI / Pinecone / Postgres
```

- **routes.py:** validates query parameters, calls the service, maps domain errors to HTTP status codes.
- **service.py:** resolves the sector, combines query results, computes shares and coverage notes, adds the optional insight.
- **repository.py:** all data access (embeddings, Pinecone queries, parameterized SQL), with timeouts and an in-memory cache of sector matches.
- **main.py:** builds the app and opens and closes connections at startup and shutdown.

**Sector resolution.** The sector text is embedded with `text-embedding-3-large` and matched against the organization vectors in Pinecone (top 1000, companies only: investors are excluded with a metadata filter). Matches are kept when they score at least `max(0.28, best_score - 0.16)`. Similarity scores vary a lot by query (the best match scores about 0.54 for "oil and gas" and 0.65 for "fintech payments"), so a window relative to the best match sizes the sector better than one fixed threshold. The floor rejects nonsense input. All three values come from the evaluation below and can be overridden in `.env` (`PINECONE_TOP_K`, `MIN_SIMILARITY`, `SIMILARITY_WINDOW`) or changed in `app/config.py`. The response's `meta` shows the cutoff and sample matched companies so you can judge the match.

## Evaluating the sector match

`evals/` holds an offline experiment that scores the sector-match settings (`pinecone_top_k`, `min_similarity`, `similarity_window`, and the query wording) against 54 test queries and recommends defaults. It calls the real OpenAI, Pinecone, and Postgres services and takes under a minute:

```bash
python -m evals.sweep
```

Results are written to `evals/results/`: `sweep_results.csv` (one row per configuration), `query_scores.csv` (per-query scores under the current and recommended settings), and `recommendation.txt`.

### Applied settings

The defaults in `app/config.py` are the configuration the sweep recommended on 2026-11-18. They replaced hand-picked values:

| Setting | Previous | Applied |
|---|---|---|
| `pinecone_top_k` | 500 | 1000 |
| `min_similarity` | 0.30 | 0.28 |
| `similarity_window` | 0.12 | 0.16 |

| Metric (40 labelled queries, macro-averaged) | Previous | Applied |
|---|---|---|
| Precision | 0.725 | 0.641 |
| Recall | 0.466 | 0.617 |
| F1 | 0.523 | 0.587 |
| Mean companies matched | 210 | 366 |
| Nonsense queries rejected | 100% | 100% |

The change trades some precision for recall: each sector now includes more companies, so fewer real ones are missed and a few more loosely related ones are included. Broad sectors gain the most (F1 0.382 to 0.549), because the previous limit of 500 cut them short.

Precision, recall, and F1 are measured against category tags, which are incomplete, so they understate true quality. Use them to compare configurations, not as absolute scores. The settings were also tuned and scored on the same 54 queries, so expect somewhat lower numbers on unseen sectors.

After changing the query set, the embeddings, or the data, run the sweep again and copy the recommended values into `app/config.py` by hand.

## Data caveats

Every response lists these in `meta.coverage_notes` where they apply:

- About 90% of funding rounds (and most acquisitions) are from 2024, so year-over-year growth largely reflects data coverage. `market_share` compares sectors under the same filters.
- Round amounts are disclosed for about 82% of rounds and valuations for about 6%. Acquisition prices are disclosed for about 16% of deals and payment terms for about 10%. Totals and medians use disclosed values only, and the coverage is reported alongside them.
- Grants, debt, and post-IPO rounds have no general funding stage and are grouped as `other`.
- The GPT insight is best-effort: if it fails or takes longer than 15 seconds, the response returns with `insight: null`.
