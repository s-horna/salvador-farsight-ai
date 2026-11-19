"""Pure scoring: apply one cutoff configuration to one query's search result."""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

from app.schemas import OrganizationMatch
from app.service import compute_similarity_cutoff


@dataclass(frozen=True)
class Configuration:
    wording: str
    top_k: int
    min_similarity: float
    similarity_window: float


@dataclass(frozen=True)
class RankedSearch:
    """One query's matches (best first) with its labels laid out for fast scoring."""

    matches: list[OrganizationMatch]
    relevant_org_uuids: frozenset[str]
    # Scores negated so they ascend, which is what bisect needs.
    negated_scores: list[float]
    # relevant_within[n] = how many of the first n matches are relevant.
    relevant_within: list[int]

    @classmethod
    def build(
        cls, matches: list[OrganizationMatch], relevant_org_uuids: frozenset[str]
    ) -> "RankedSearch":
        hits = (match.org_uuid in relevant_org_uuids for match in matches)
        return cls(
            matches=matches,
            relevant_org_uuids=relevant_org_uuids,
            negated_scores=[-match.score for match in matches],
            relevant_within=[0, *accumulate(int(hit) for hit in hits)],
        )

    @property
    def top_similarity(self) -> float:
        return self.matches[0].score if self.matches else 0.0


@dataclass(frozen=True)
class QueryScore:
    rejected: bool
    companies_matched: int
    relevant_matched: int
    precision: float
    recall: float
    f1: float
    capped_at_top_k: bool


def score_query(search: RankedSearch, configuration: Configuration) -> QueryScore:
    """Keep the matches the API would keep under `configuration` and score them.

    A smaller top_k is simulated by taking a prefix of the ranked matches. A rejected
    query (best match below the floor) keeps nothing and scores zero.
    """
    searched = min(configuration.top_k, len(search.matches))
    cutoff = (
        compute_similarity_cutoff(
            search.top_similarity, configuration.min_similarity, configuration.similarity_window
        )
        if searched
        else None
    )
    if cutoff is None:
        return QueryScore(True, 0, 0, 0.0, 0.0, 0.0, False)

    kept = bisect_right(search.negated_scores, -cutoff, 0, searched)
    relevant_matched = search.relevant_within[kept]
    relevant_total = len(search.relevant_org_uuids)
    precision = relevant_matched / kept if kept else 0.0
    recall = relevant_matched / relevant_total if relevant_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return QueryScore(
        rejected=False,
        companies_matched=kept,
        relevant_matched=relevant_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        capped_at_top_k=kept == searched == configuration.top_k,
    )
