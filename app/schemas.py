"""Pydantic request/response schemas and small domain types. No I/O."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FundingStage = Literal["seed", "early_stage_venture", "late_stage_venture", "private_equity"]
AcquirerType = Literal["company", "investor"]


# --------------------------------------------------------------------------- #
# Domain types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrganizationMatch:
    org_uuid: str
    name: str | None
    score: float


@dataclass(frozen=True)
class SectorMatch:
    sector: str
    org_uuids: list[str]
    similarity_cutoff: float
    top_similarity: float
    sample_companies: list[str]
    capped_at_top_k: bool


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class TrendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str = Field(
        min_length=2,
        max_length=200,
        description="Free-text sector description, e.g. 'generative AI' or 'oil and gas'.",
    )
    include_insight: bool = Field(
        default=True, description="Add a two-sentence GPT takeaway to the response."
    )

    @field_validator("sector")
    @classmethod
    def strip_sector(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("sector must contain at least 2 non-space characters")
        return value


class FundingTrendRequest(TrendRequest):
    start_date: date | None = Field(default=None, description="Inclusive lower bound on round date.")
    end_date: date | None = Field(default=None, description="Inclusive upper bound on round date.")
    stage: FundingStage | None = Field(default=None, description="Filter by general funding stage.")

    @model_validator(mode="after")
    def check_date_range(self) -> "FundingTrendRequest":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class AcquisitionTrendRequest(TrendRequest):
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)
    acquirer_type: AcquirerType | None = Field(
        default=None, description="company = strategic buyer, investor = financial buyer."
    )

    @model_validator(mode="after")
    def check_year_range(self) -> "AcquisitionTrendRequest":
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be on or before end_year")
        return self


# --------------------------------------------------------------------------- #
# Shared response parts
# --------------------------------------------------------------------------- #


class TrendMeta(BaseModel):
    sector: str
    filters: dict[str, Any]
    similarity_cutoff: float
    top_similarity: float
    sample_companies: list[str]
    coverage_notes: list[str]


class CategoryCount(BaseModel):
    label: str
    count: int
    share: float = Field(description="Fraction of the total, 0-1.")


# --------------------------------------------------------------------------- #
# Funding response
# --------------------------------------------------------------------------- #


class FundingSummary(BaseModel):
    round_count: int
    rounds_with_amount: int
    companies_funded: int
    total_raised_usd: int
    median_raise_usd: float | None


class MonthlyFunding(BaseModel):
    month: str = Field(description="YYYY-MM")
    round_count: int
    total_raised_usd: int


class StageBreakdown(BaseModel):
    stage: str
    round_count: int
    total_raised_usd: int
    share_of_rounds: float


class FundingDeal(BaseModel):
    org_uuid: str
    company_name: str | None
    investment_date: date | None
    stage: str | None
    fundraise_amount_usd: int
    valuation_usd: int | None


class MarketShare(BaseModel):
    sector_rounds: int
    all_rounds: int
    share_of_rounds: float
    sector_raised_usd: int
    all_raised_usd: int
    share_of_dollars: float


class FundingTrendResponse(BaseModel):
    summary: FundingSummary
    rounds_by_month: list[MonthlyFunding]
    stage_mix: list[StageBreakdown]
    top_deals: list[FundingDeal]
    market_share: MarketShare
    companies_matched: int
    meta: TrendMeta
    insight: str | None = None


# --------------------------------------------------------------------------- #
# Acquisition response
# --------------------------------------------------------------------------- #


class YearlyDeals(BaseModel):
    year: int
    deal_count: int
    priced_deal_count: int
    total_price_usd: int


class TopAcquirer(BaseModel):
    acquirer_uuid: str
    name: str | None
    acquirer_type: str | None
    deal_count: int
    total_price_usd: int


class PriceStats(BaseModel):
    total_deals: int
    priced_deals: int
    price_coverage: float
    total_price_usd: int
    median_price_usd: float | None
    max_price_usd: int | None


class AcquisitionTrendResponse(BaseModel):
    deal_count: int
    deals_by_year: list[YearlyDeals]
    top_acquirers: list[TopAcquirer]
    buyer_mix: list[CategoryCount]
    deal_types: list[CategoryCount]
    terms: list[CategoryCount]
    price_stats: PriceStats
    companies_matched: int
    meta: TrendMeta
    insight: str | None = None


# --------------------------------------------------------------------------- #
# Health and errors
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    ok: bool
    postgres: Literal["ok", "unavailable"]
    pinecone: Literal["ok", "unavailable"]
    openai: Literal["ok", "unavailable"]


class ErrorResponse(BaseModel):
    error: str
    detail: str
