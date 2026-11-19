"""Business logic: sector resolution, both trends, coverage notes, optional insight.

No SQL and no HTTP here. Data comes from TrendRepository; results are returned
as response schemas.
"""

import asyncio
import json
import logging
from typing import Any

from app.config import Settings
from app.exceptions import SectorNotFoundError, UpstreamServiceError
from app.repository import TrendRepository
from app.schemas import (
    AcquisitionTrendRequest,
    AcquisitionTrendResponse,
    CategoryCount,
    FundingDeal,
    FundingSummary,
    FundingTrendRequest,
    FundingTrendResponse,
    HealthResponse,
    MarketShare,
    MonthlyFunding,
    PriceStats,
    SectorMatch,
    StageBreakdown,
    TopAcquirer,
    TrendMeta,
    YearlyDeals,
)

logger = logging.getLogger(__name__)

BUYER_LABELS = {"company": "strategic", "investor": "financial"}
SAMPLE_COMPANY_COUNT = 10

INSIGHT_SYSTEM_PROMPT = (
    "You are a financial analyst writing for investment and corporate development "
    "professionals. Given aggregated data about one sector, write exactly two sentences "
    "with the most useful takeaway. Cite specific numbers. Do not speculate beyond the "
    "data. The meta.coverage_notes describe data limitations; respect them (for example, "
    "do not present growth that the notes attribute to dataset coverage as a real trend)."
)


def _ratio(part: int | float, whole: int | float) -> float:
    """part / whole rounded to 4 decimals, or 0.0 when whole is 0."""
    return round(part / whole, 4) if whole else 0.0


def compute_similarity_cutoff(
    top_similarity: float, min_similarity: float, similarity_window: float
) -> float | None:
    """Lowest score a match may have to count as part of the sector.

    Returns None when even the best match is below the absolute floor, meaning the
    sector text matches nothing. Shared with `evals/` so the sweep scores the same
    rule the API applies.
    """
    if top_similarity < min_similarity:
        return None
    return max(min_similarity, top_similarity - similarity_window)


class TrendService:
    """Builds the funding and acquisition trends for a free-text sector."""

    def __init__(self, repository: TrendRepository, settings: Settings) -> None:
        """Store the shared repository and settings."""
        self._repository = repository
        self._settings = settings

    # ----------------------------------------------------------------------- #
    # Health
    # ----------------------------------------------------------------------- #

    async def check_health(self) -> HealthResponse:
        """Ping Postgres, Pinecone, and OpenAI concurrently."""
        postgres_ok, pinecone_ok, openai_ok = await asyncio.gather(
            self._repository.ping_postgres(),
            self._repository.ping_pinecone(),
            self._repository.ping_openai(),
        )
        return HealthResponse(
            ok=postgres_ok and pinecone_ok and openai_ok,
            postgres="ok" if postgres_ok else "unavailable",
            pinecone="ok" if pinecone_ok else "unavailable",
            openai="ok" if openai_ok else "unavailable",
        )

    # ----------------------------------------------------------------------- #
    # Sector resolution
    # ----------------------------------------------------------------------- #

    async def _resolve_sector(self, sector: str) -> SectorMatch:
        """Find the organizations in a sector.

        Pinecone scores drift by query (top scores range roughly 0.54-0.65 for real
        sectors and stay under 0.25 for nonsense), so a single fixed threshold either
        drops real companies or lets noise in. Instead keep matches that are both
        above an absolute floor and within a window of the best score.
        """
        top_k = self._settings.pinecone_top_k
        # Search companies only. Pinecone also holds investors (VC and PE firms) whose
        # profiles describe the sectors they invest in, so "generative AI" would match
        # AI-focused funds. Those are not companies in the sector: counting them would
        # inflate companies_matched and, because investors are rarely the funded or
        # acquired party, would add matches that contribute no rounds or deals.
        matches = await self._repository.search_organizations(sector, top_k, org_type="company")
        if not matches:
            raise SectorNotFoundError(sector)

        top_similarity = matches[0].score
        cutoff = compute_similarity_cutoff(
            top_similarity, self._settings.min_similarity, self._settings.similarity_window
        )
        if cutoff is None:
            raise SectorNotFoundError(sector)
        kept = [match for match in matches if match.score >= cutoff]

        return SectorMatch(
            sector=sector,
            org_uuids=[match.org_uuid for match in kept],
            similarity_cutoff=round(cutoff, 4),
            top_similarity=round(top_similarity, 4),
            sample_companies=[match.name for match in kept[:SAMPLE_COMPANY_COUNT] if match.name],
            capped_at_top_k=len(kept) == len(matches) == top_k,
        )

    def _build_meta(self, match: SectorMatch, filters: dict[str, Any], notes: list[str]) -> TrendMeta:
        """Assemble the meta block: query echo, match quality, and coverage notes."""
        sector_notes = [
            f"Sector defined as the {len(match.org_uuids)} companies with similarity "
            f">= {match.similarity_cutoff} to '{match.sector}' (best match {match.top_similarity})."
        ]
        if match.capped_at_top_k:
            sector_notes.append(
                f"Match count hit the search limit of {self._settings.pinecone_top_k}; "
                "a broad sector may include more companies than shown."
            )
        return TrendMeta(
            sector=match.sector,
            filters={key: value for key, value in filters.items() if value is not None},
            similarity_cutoff=match.similarity_cutoff,
            top_similarity=match.top_similarity,
            sample_companies=match.sample_companies,
            coverage_notes=sector_notes + notes,
        )

    # ----------------------------------------------------------------------- #
    # Funding
    # ----------------------------------------------------------------------- #

    async def get_funding_trend(self, request: FundingTrendRequest) -> FundingTrendResponse:
        """Funding snapshot for the sector: summary, monthly series, stages, top deals, share."""
        # match: the companies in the sector plus how they were chosen (cutoff, samples)
        match = await self._resolve_sector(request.sector)
        # uuids: the sector's company IDs, passed to every query
        uuids = match.org_uuids
        # filters: the optional user filters, in the order every funding query expects
        filters = (request.start_date, request.end_date, request.stage)

        # Run all queries concurrently. Each result is a dict (one row) or a list of dicts:
        #   summary_row: sector totals and disclosure counts
        #   month_rows:  one row per YYYY-MM
        #   stage_rows:  one row per general funding stage
        #   deal_rows:   the largest disclosed rounds
        #   market_row:  the same totals as summary_row for all companies (uuids=None)
        summary_row, month_rows, stage_rows, deal_rows, market_row = await asyncio.gather(
            self._repository.get_funding_summary(uuids, *filters),
            self._repository.get_funding_by_month(uuids, *filters),
            self._repository.get_funding_stage_mix(uuids, *filters),
            self._repository.get_top_funding_deals(uuids, *filters, self._settings.top_results_limit),
            self._repository.get_funding_summary(None, *filters),
        )

        # summary: headline numbers for the sector
        summary = FundingSummary(
            round_count=summary_row["round_count"],
            rounds_with_amount=summary_row["rounds_with_amount"],
            companies_funded=summary_row["companies_funded"],
            total_raised_usd=summary_row["total_raised_usd"],
            median_raise_usd=summary_row["median_raise_usd"],
        )
        # stage_mix: each stage's rounds and dollars, plus its share of the sector's rounds
        stage_mix = [
            StageBreakdown(
                stage=row["stage"],
                round_count=row["round_count"],
                total_raised_usd=row["total_raised_usd"],
                share_of_rounds=_ratio(row["round_count"], summary.round_count),
            )
            for row in stage_rows
        ]
        # market_share: the sector's fraction of all rounds and dollars under the same
        # filters. Both sides share the dataset's 2024 skew, so the ratio cancels it out.
        market_share = MarketShare(
            sector_rounds=summary.round_count,
            all_rounds=market_row["round_count"],
            share_of_rounds=_ratio(summary.round_count, market_row["round_count"]),
            sector_raised_usd=summary.total_raised_usd,
            all_raised_usd=market_row["total_raised_usd"],
            share_of_dollars=_ratio(summary.total_raised_usd, market_row["total_raised_usd"]),
        )

        # response: the full payload; the insight is added afterwards because it is
        # generated from the finished response
        response = FundingTrendResponse(
            summary=summary,
            rounds_by_month=[MonthlyFunding(**row) for row in month_rows],
            stage_mix=stage_mix,
            top_deals=[FundingDeal(**row) for row in deal_rows],
            market_share=market_share,
            companies_matched=len(uuids),
            meta=self._build_meta(
                match,
                {
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "stage": request.stage,
                },
                self._funding_notes(summary_row, month_rows, stage_rows),
            ),
        )
        if request.include_insight and summary.round_count:
            response.insight = await self._generate_insight(
                "funding", response.model_dump(mode="json", exclude={"insight"})
            )
        return response

    @staticmethod
    def _funding_notes(
        summary_row: dict[str, Any], month_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]
    ) -> list[str]:
        """Coverage notes for the funding trend: disclosure rates, stage 'other', 2024 skew."""
        round_count = summary_row["round_count"]
        if not round_count:
            return ["No funding rounds found for these companies and filters."]

        rounds_with_amount = summary_row["rounds_with_amount"]
        notes = [
            f"Amount disclosed for {rounds_with_amount} of {round_count} rounds "
            f"({_ratio(rounds_with_amount, round_count):.0%}); totals and medians use "
            "disclosed amounts only.",
            f"Valuation disclosed for {summary_row['rounds_with_valuation']} of {round_count} rounds.",
        ]
        if any(row["stage"] == "other" for row in stage_rows):
            notes.append(
                "Stage 'other' covers grants, debt, and post-IPO rounds, which have no general "
                "funding stage."
            )
        rounds_2024 = sum(row["round_count"] for row in month_rows if row["month"].startswith("2024"))
        share_2024 = _ratio(rounds_2024, round_count)
        if share_2024 >= 0.5:
            notes.append(
                f"{share_2024:.0%} of these rounds are from 2024 (the dataset is concentrated "
                "there); compare sectors with market_share rather than raw counts."
            )
        return notes

    # ----------------------------------------------------------------------- #
    # Acquisitions
    # ----------------------------------------------------------------------- #

    async def get_acquisition_trend(self, request: AcquisitionTrendRequest) -> AcquisitionTrendResponse:
        """M&A landscape for the sector: deals by year, acquirers, buyer mix, types, terms, prices."""
        # match: the companies in the sector plus how they were chosen (cutoff, samples)
        match = await self._resolve_sector(request.sector)
        # uuids: the sector's company IDs; each query counts deals where one was acquired
        uuids = match.org_uuids
        # filters: the optional user filters, in the order every acquisition query expects
        filters = (request.start_year, request.end_year, request.acquirer_type)

        # Run all queries concurrently. Each result is a dict (one row) or a list of dicts:
        #   year_rows:     one row per announcement year
        #   acquirer_rows: the most active buyers
        #   buyer_rows:    deal counts per acquirer_type (company / investor)
        #   type_rows:     deal counts per acquisition_type
        #   term_rows:     deal counts per payment terms, including 'unknown'
        #   price_row:     total deal count and disclosed-price statistics
        year_rows, acquirer_rows, buyer_rows, type_rows, term_rows, price_row = await asyncio.gather(
            self._repository.get_acquisitions_by_year(uuids, *filters),
            self._repository.get_top_acquirers(uuids, *filters, self._settings.top_results_limit),
            self._repository.count_acquisitions_by("acquirer_type", uuids, *filters),
            self._repository.count_acquisitions_by("acquisition_type", uuids, *filters),
            self._repository.count_acquisitions_by("terms", uuids, *filters),
            self._repository.get_acquisition_price_stats(uuids, *filters),
        )

        # deal_count: every matching deal; the denominator for buyer and deal-type shares
        deal_count = price_row["total_deals"]
        # known_terms / terms_known: terms are undisclosed for ~90% of deals, so terms
        # shares are computed over disclosed deals only (otherwise 'unknown' dominates)
        known_terms = [row for row in term_rows if row["label"] != "unknown"]
        terms_known = sum(row["count"] for row in known_terms)

        # price_stats: disclosed-price statistics next to how many deals disclosed a price
        price_stats = PriceStats(
            total_deals=deal_count,
            priced_deals=price_row["priced_deals"],
            price_coverage=_ratio(price_row["priced_deals"], deal_count),
            total_price_usd=price_row["total_price_usd"],
            median_price_usd=price_row["median_price_usd"],
            max_price_usd=price_row["max_price_usd"],
        )

        # response: the full payload; buyer_mix relabels company/investor as
        # strategic/financial; the insight is added afterwards from the finished response
        response = AcquisitionTrendResponse(
            deal_count=deal_count,
            deals_by_year=[YearlyDeals(**row) for row in year_rows],
            top_acquirers=[TopAcquirer(**row) for row in acquirer_rows],
            buyer_mix=[
                CategoryCount(
                    label=BUYER_LABELS.get(row["label"], row["label"]),
                    count=row["count"],
                    share=_ratio(row["count"], deal_count),
                )
                for row in buyer_rows
            ],
            deal_types=self._to_category_counts(type_rows, deal_count),
            terms=self._to_category_counts(known_terms, terms_known),
            price_stats=price_stats,
            companies_matched=len(uuids),
            meta=self._build_meta(
                match,
                {
                    "start_year": request.start_year,
                    "end_year": request.end_year,
                    "acquirer_type": request.acquirer_type,
                },
                self._acquisition_notes(deal_count, price_stats.priced_deals, terms_known, year_rows),
            ),
        )
        if request.include_insight and deal_count:
            response.insight = await self._generate_insight(
                "acquisitions", response.model_dump(mode="json", exclude={"insight"})
            )
        return response

    @staticmethod
    def _to_category_counts(rows: list[dict[str, Any]], total: int) -> list[CategoryCount]:
        """Turn label/count rows into CategoryCounts with each label's share of `total`."""
        return [
            CategoryCount(label=row["label"], count=row["count"], share=_ratio(row["count"], total))
            for row in rows
        ]

    @staticmethod
    def _acquisition_notes(
        deal_count: int, priced_deals: int, terms_known: int, year_rows: list[dict[str, Any]]
    ) -> list[str]:
        """Coverage notes for the acquisition trend: disclosure rates, definitions, 2024 skew."""
        if not deal_count:
            return ["No acquisitions of these companies found for the given filters."]
        notes = [
            f"Price disclosed for {priced_deals} of {deal_count} deals "
            f"({_ratio(priced_deals, deal_count):.0%}); price_stats use disclosed prices only.",
            f"Payment terms disclosed for {terms_known} of {deal_count} deals; "
            "terms shares are of disclosed deals.",
            "Deals are acquisitions of companies in the sector; buyer_mix 'strategic' means a "
            "company buyer, 'financial' an investor buyer.",
        ]
        deals_2024 = sum(row["deal_count"] for row in year_rows if row["year"] == 2024)
        share_2024 = _ratio(deals_2024, deal_count)
        if share_2024 >= 0.5:
            notes.append(
                f"{share_2024:.0%} of these deals are from 2024; the dataset is concentrated "
                "there, so year-over-year growth partly reflects data coverage."
            )
        return notes

    # ----------------------------------------------------------------------- #
    # Insight
    # ----------------------------------------------------------------------- #

    async def _generate_insight(self, trend_name: str, data: dict[str, Any]) -> str | None:
        """Two-sentence GPT takeaway. Never fails the request."""
        messages = [
            {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Trend: {trend_name}\nData (JSON):\n{json.dumps(data, separators=(',', ':'))}",
            },
        ]
        try:
            return await self._repository.generate_text(messages) or None
        except UpstreamServiceError:
            logger.warning("Insight generation failed; returning response without insight", exc_info=True)
            return None
