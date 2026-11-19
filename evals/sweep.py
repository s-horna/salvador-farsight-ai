"""Sweep the sector-match settings and recommend defaults.

For every query in the evaluation set, search once per query wording through
`TrendRepository.search_organizations` (the call the API makes), then score every
combination of top_k, min_similarity, and similarity_window against the weak labels.

Writes to evals/results/:
  sweep_results.csv   one row per configuration
  query_scores.csv    per-query scores under the current and recommended settings
  recommendation.txt  the summary printed at the end of the run

Run from the project root:  python -m evals.sweep
"""

import asyncio
import csv
import logging
from itertools import product
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.config import Settings
from app.repository import TrendRepository
from evals.dataset import LABELLED_KINDS, EvalQuery, load_queries, load_relevant_org_uuids
from evals.metrics import Configuration, QueryScore, RankedSearch, score_query

RESULTS_DIR = Path(__file__).with_name("results")

# How the sector phrase is turned into the text that gets embedded. "template"
# mimics the stored organization descriptions.
WORDINGS = {
    "raw": "{sector}",
    "template": (
        "The company operates within the {sector} subsector, "
        "specifically focusing on {sector}."
    ),
}
TOP_K_VALUES = (250, 500, 1000)
# The template wording scores about 0.25 higher, so the floor range covers both.
MIN_SIMILARITY_VALUES = tuple(round(0.20 + 0.02 * step, 2) for step in range(31))
SIMILARITY_WINDOW_VALUES = tuple(round(0.04 + 0.02 * step, 2) for step in range(14))

# Configurations within this much of the best macro F1 count as tied; the weak
# labels are noisier than that.
F1_TOLERANCE = 0.005
SEARCH_CONCURRENCY = 8
UNLABELLED_SAMPLE_COUNT = 5

# (wording, query text) -> that query's ranked matches and labels
Searches = dict[tuple[str, str], RankedSearch]


async def run_searches(
    repository: TrendRepository,
    queries: list[EvalQuery],
    relevant: dict[str, frozenset[str]],
) -> Searches:
    """Search every query under every wording at the largest top_k in the sweep."""
    top_k = max(TOP_K_VALUES)
    semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def search(wording: str, query: EvalQuery) -> tuple[tuple[str, str], RankedSearch]:
        text = WORDINGS[wording].format(sector=query.text)
        async with semaphore:
            matches = await repository.search_organizations(text, top_k, org_type="company")
        ranked = RankedSearch.build(matches, relevant.get(query.text, frozenset()))
        return (wording, query.text), ranked

    results = await asyncio.gather(
        *(search(wording, query) for wording in WORDINGS for query in queries)
    )
    return dict(results)


def score_queries(
    configuration: Configuration, queries: list[EvalQuery], searches: Searches
) -> dict[str, QueryScore]:
    """Score every query under one configuration, keyed by query text."""
    return {
        query.text: score_query(searches[configuration.wording, query.text], configuration)
        for query in queries
    }


def summarize(
    configuration: Configuration, queries: list[EvalQuery], searches: Searches
) -> dict[str, Any]:
    """One sweep row: macro-averaged quality on labelled queries plus rejection rates."""
    scores = score_queries(configuration, queries, searches)

    def of_kind(*kinds: str) -> list[QueryScore]:
        return [scores[query.text] for query in queries if query.kind in kinds]

    def rejection_rate(kind_scores: list[QueryScore]) -> float:
        return mean(score.rejected for score in kind_scores) if kind_scores else 0.0

    labelled = of_kind(*LABELLED_KINDS)
    row: dict[str, Any] = {
        "wording": configuration.wording,
        "top_k": configuration.top_k,
        "min_similarity": configuration.min_similarity,
        "similarity_window": configuration.similarity_window,
        "precision": mean(score.precision for score in labelled),
        "recall": mean(score.recall for score in labelled),
        "f1": mean(score.f1 for score in labelled),
    }
    for kind in LABELLED_KINDS:
        row[f"f1_{kind}"] = mean(score.f1 for score in of_kind(kind))
    row |= {
        "labelled_rejection_rate": rejection_rate(labelled),
        "nonsense_rejection_rate": rejection_rate(of_kind("nonsense")),
        "ambiguous_rejection_rate": rejection_rate(of_kind("ambiguous")),
        "mean_companies_matched": mean(score.companies_matched for score in labelled),
        "capped_at_top_k_rate": mean(score.capped_at_top_k for score in labelled),
    }
    return row


def top_similarity_ranges(queries: list[EvalQuery], searches: Searches) -> dict[str, dict[str, float]]:
    """Per wording: where real, ambiguous, and nonsense queries' best scores fall."""
    ranges = {}
    for wording in WORDINGS:

        def tops(*kinds: str) -> list[float]:
            return [
                searches[wording, query.text].top_similarity
                for query in queries
                if query.kind in kinds
            ]

        labelled, ambiguous, nonsense = tops(*LABELLED_KINDS), tops("ambiguous"), tops("nonsense")
        ranges[wording] = {
            "labelled_min": min(labelled),
            "labelled_median": median(labelled),
            "labelled_max": max(labelled),
            "ambiguous_min": min(ambiguous),
            "ambiguous_max": max(ambiguous),
            "nonsense_max": max(nonsense),
        }
    return ranges


def recommend(rows: list[dict[str, Any]], ranges: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Pick the configuration to recommend.

    1. Keep configurations with the highest nonsense rejection rate.
    2. Among those, keep the ones within F1_TOLERANCE of the best macro F1.
    3. Of those ties, prefer the floor that sits farthest from both the highest
       nonsense score and the lowest real-sector score, then the higher F1.
    """

    def floor_margin(row: dict[str, Any]) -> float:
        wording_range = ranges[row["wording"]]
        return min(
            row["min_similarity"] - wording_range["nonsense_max"],
            wording_range["labelled_min"] - row["min_similarity"],
        )

    best_rejection = max(row["nonsense_rejection_rate"] for row in rows)
    eligible = [row for row in rows if row["nonsense_rejection_rate"] == best_rejection]
    best_f1 = max(row["f1"] for row in eligible)
    tied = [row for row in eligible if row["f1"] >= best_f1 - F1_TOLERANCE]
    return max(tied, key=lambda row: (floor_margin(row), row["f1"]))


def to_configuration(row: dict[str, Any]) -> Configuration:
    return Configuration(
        row["wording"], row["top_k"], row["min_similarity"], row["similarity_window"]
    )


def build_query_rows(
    name: str, configuration: Configuration, queries: list[EvalQuery], searches: Searches
) -> list[dict[str, Any]]:
    """Per-query detail for one named configuration, with unlabelled matches to spot-check."""
    scores = score_queries(configuration, queries, searches)
    rows = []
    for query in queries:
        search, score = searches[configuration.wording, query.text], scores[query.text]
        unlabelled = [
            match.name
            for match in search.matches[: score.companies_matched]
            if match.name and match.org_uuid not in search.relevant_org_uuids
        ]
        rows.append(
            {
                "configuration": name,
                "wording": configuration.wording,
                "top_k": configuration.top_k,
                "min_similarity": configuration.min_similarity,
                "similarity_window": configuration.similarity_window,
                "kind": query.kind,
                "query": query.text,
                "labelled_companies": len(search.relevant_org_uuids),
                "top_similarity": round(search.top_similarity, 4),
                "rejected": score.rejected,
                "companies_matched": score.companies_matched,
                "relevant_matched": score.relevant_matched,
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
                "top_unlabelled_matches": "; ".join(unlabelled[:UNLABELLED_SAMPLE_COUNT])
                if query.is_labelled
                else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, rounding floats to 4 decimals."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}
            )


def describe(row: dict[str, Any]) -> str:
    """One configuration and its headline numbers on a single line."""
    return (
        f"wording={row['wording']} top_k={row['top_k']} "
        f"min_similarity={row['min_similarity']:.2f} similarity_window={row['similarity_window']:.2f}"
        f" | precision {row['precision']:.3f} recall {row['recall']:.3f} F1 {row['f1']:.3f}"
        f" (broad {row['f1_broad']:.3f}, medium {row['f1_medium']:.3f}, narrow {row['f1_narrow']:.3f})"
        f" | nonsense rejected {row['nonsense_rejection_rate']:.0%},"
        f" real rejected {row['labelled_rejection_rate']:.0%},"
        f" ambiguous rejected {row['ambiguous_rejection_rate']:.0%}"
        f" | mean matched {row['mean_companies_matched']:.0f},"
        f" capped at top_k {row['capped_at_top_k_rate']:.0%}"
    )


def build_report(
    queries: list[EvalQuery],
    rows: list[dict[str, Any]],
    ranges: dict[str, dict[str, float]],
    current: dict[str, Any],
    recommended: dict[str, Any],
) -> str:
    """The human-readable summary: score ranges, current vs. best, and the trade-off."""
    labelled_count = sum(query.is_labelled for query in queries)
    lines = [
        "Sector-match sweep",
        "==================",
        f"{len(queries)} queries ({labelled_count} labelled), {len(rows)} configurations.",
        "Precision, recall, and F1 are macro-averaged over labelled queries against weak",
        "category labels, so they understate true quality; compare configurations, do not",
        "read the values as absolute.",
        "",
        "Best-match similarity by wording",
    ]
    for wording, values in ranges.items():
        lines.append(
            f"  {wording}: real sectors {values['labelled_min']:.3f}-{values['labelled_max']:.3f}"
            f" (median {values['labelled_median']:.3f}),"
            f" ambiguous {values['ambiguous_min']:.3f}-{values['ambiguous_max']:.3f},"
            f" nonsense up to {values['nonsense_max']:.3f}."
            f" Gap between nonsense and real: {values['labelled_min'] - values['nonsense_max']:+.3f}"
        )

    lines += ["", "Current defaults", f"  {describe(current)}", "", "Best per wording"]
    for wording in WORDINGS:
        best = recommend([row for row in rows if row["wording"] == wording], ranges)
        lines.append(f"  {describe(best)}")

    lines += [
        "",
        "Recommended",
        f"  {describe(recommended)}",
        "",
        "Trade-off against the current defaults",
        f"  precision {current['precision']:.3f} -> {recommended['precision']:.3f},"
        f" recall {current['recall']:.3f} -> {recommended['recall']:.3f},"
        f" F1 {current['f1']:.3f} -> {recommended['f1']:.3f},"
        f" mean companies matched {current['mean_companies_matched']:.0f}"
        f" -> {recommended['mean_companies_matched']:.0f}.",
    ]
    return "\n".join(lines)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = Settings()
    queries = load_queries()
    relevant = await load_relevant_org_uuids(settings, queries)

    repository = await TrendRepository.connect(settings)
    try:
        searches = await run_searches(repository, queries, relevant)
    finally:
        await repository.close()

    rows = [
        summarize(Configuration(*values), queries, searches)
        for values in product(
            WORDINGS, TOP_K_VALUES, MIN_SIMILARITY_VALUES, SIMILARITY_WINDOW_VALUES
        )
    ]
    ranges = top_similarity_ranges(queries, searches)
    current_configuration = Configuration(
        "raw", settings.pinecone_top_k, settings.min_similarity, settings.similarity_window
    )
    current = summarize(current_configuration, queries, searches)
    recommended = recommend(rows, ranges)

    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "sweep_results.csv", rows)
    write_csv(
        RESULTS_DIR / "query_scores.csv",
        build_query_rows("current", current_configuration, queries, searches)
        + build_query_rows("recommended", to_configuration(recommended), queries, searches),
    )
    report = build_report(queries, rows, ranges, current, recommended)
    (RESULTS_DIR / "recommendation.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
