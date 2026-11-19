"""Data access: OpenAI embeddings and chat, Pinecone vector search, Postgres SQL.

No business decisions here. Every external failure is raised as
UpstreamServiceError so the routes layer can map it to a 503.
"""

import asyncio
import logging
from collections import OrderedDict
from datetime import date
from typing import Any

import asyncpg
from openai import AsyncOpenAI, OpenAIError
from pinecone import Pinecone

from app.config import Settings
from app.exceptions import UpstreamServiceError
from app.schemas import OrganizationMatch

logger = logging.getLogger(__name__)

Row = dict[str, Any]
# (lowercased sector text, top_k, org_type)
CacheKey = tuple[str, int, str | None]


class TrendRepository:
    """Data access for the trends: sector search plus funding and acquisition SQL."""

    def __init__(
        self,
        settings: Settings,
        openai_client: AsyncOpenAI,
        pinecone_index: Any,
        pool: asyncpg.Pool,
    ) -> None:
        """Store already-opened clients; use `connect` to create them."""
        self._settings = settings
        self._openai = openai_client
        self._index = pinecone_index
        self._pool = pool
        self._sector_cache: OrderedDict[CacheKey, list[OrganizationMatch]] = OrderedDict()

    # ----------------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------------- #

    @classmethod
    async def connect(cls, settings: Settings) -> "TrendRepository":
        """Open the OpenAI client, Pinecone index, and Postgres pool."""
        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        try:
            pinecone_index = Pinecone(
                api_key=settings.pinecone_api_key.get_secret_value()
            ).Index(settings.pinecone_index)
            pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password.get_secret_value(),
                database=settings.postgres_db_name,
                min_size=settings.postgres_pool_min_size,
                max_size=settings.postgres_pool_max_size,
                command_timeout=settings.postgres_command_timeout_seconds,
                timeout=settings.postgres_command_timeout_seconds,
            )
        except Exception:
            await openai_client.close()
            raise
        return cls(settings, openai_client, pinecone_index, pool)

    async def close(self) -> None:
        """Close the Postgres pool and the OpenAI client."""
        await self._pool.close()
        await self._openai.close()

    # ----------------------------------------------------------------------- #
    # Health
    # ----------------------------------------------------------------------- #

    async def ping_postgres(self) -> bool:
        """Return True if Postgres answers a trivial query."""
        try:
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            logger.warning("Postgres health check failed", exc_info=True)
            return False

    async def ping_pinecone(self) -> bool:
        """Return True if the Pinecone index answers a stats request."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._index.describe_index_stats),
                timeout=self._settings.pinecone_timeout_seconds,
            )
            return True
        except Exception:
            logger.warning("Pinecone health check failed", exc_info=True)
            return False

    async def ping_openai(self) -> bool:
        """Return True if OpenAI confirms access to the embedding model (no tokens used)."""
        try:
            await self._openai.with_options(max_retries=0).models.retrieve(
                self._settings.embedding_model
            )
            return True
        except Exception:
            logger.warning("OpenAI health check failed", exc_info=True)
            return False

    # ----------------------------------------------------------------------- #
    # OpenAI + Pinecone
    # ----------------------------------------------------------------------- #

    async def search_organizations(
        self, text: str, top_k: int, org_type: str | None = None
    ) -> list[OrganizationMatch]:
        """Embed `text` and return the top_k most similar organizations (cached).

        `org_type` ('company' or 'investor') restricts the search with a Pinecone
        metadata filter, so the top_k slots are not spent on excluded organizations.
        """
        cache_key = (text.lower(), top_k, org_type)
        if cache_key in self._sector_cache:
            self._sector_cache.move_to_end(cache_key)
            return self._sector_cache[cache_key]

        vector = await self._embed(text)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._index.query,
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"org_type": {"$eq": org_type}} if org_type else None,
                ),
                timeout=self._settings.pinecone_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise UpstreamServiceError("Pinecone", "query timed out") from exc
        except Exception as exc:
            logger.exception("Pinecone query failed")
            raise UpstreamServiceError("Pinecone") from exc

        matches = [
            OrganizationMatch(
                org_uuid=match.id,
                name=(match.metadata or {}).get("name"),
                score=float(match.score),
            )
            for match in result.matches
        ]
        self._remember(cache_key, matches)
        return matches

    async def _embed(self, text: str) -> list[float]:
        """Embed text with the same model used to build the Pinecone index."""
        try:
            response = await self._openai.embeddings.create(
                model=self._settings.embedding_model, input=text
            )
        except OpenAIError as exc:
            logger.exception("OpenAI embedding failed")
            raise UpstreamServiceError("OpenAI") from exc
        return response.data[0].embedding

    def _remember(self, key: CacheKey, matches: list[OrganizationMatch]) -> None:
        """Store matches in the LRU cache, evicting the least recently used entry."""
        if self._settings.sector_cache_size == 0:
            return
        self._sector_cache[key] = matches
        self._sector_cache.move_to_end(key)
        while len(self._sector_cache) > self._settings.sector_cache_size:
            self._sector_cache.popitem(last=False)

    async def generate_text(self, messages: list[dict[str, str]]) -> str:
        """Single chat completion, capped at the insight timeout with no retries."""
        try:
            completion = await self._openai.with_options(
                timeout=self._settings.insight_timeout_seconds, max_retries=0
            ).chat.completions.create(model=self._settings.insight_model, messages=messages)
        except OpenAIError as exc:
            raise UpstreamServiceError("OpenAI", "chat completion failed") from exc
        return (completion.choices[0].message.content or "").strip()

    # ----------------------------------------------------------------------- #
    # Postgres helpers
    # ----------------------------------------------------------------------- #

    async def _fetch(self, query: str, *args: Any) -> list[Row]:
        """Run a parameterized query and return rows as dicts."""
        try:
            rows = await self._pool.fetch(query, *args)
        except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as exc:
            logger.exception("Postgres query failed")
            raise UpstreamServiceError("Postgres") from exc
        return [dict(row) for row in rows]

    # ----------------------------------------------------------------------- #
    # Funding rounds
    # ----------------------------------------------------------------------- #

    # $1 = org_uuids (NULL means the whole market), $2 = start_date, $3 = end_date, $4 = stage
    _FUNDING_FILTER = """
        ($1::uuid[] IS NULL OR f.org_uuid = ANY($1::uuid[]))
        AND ($2::date IS NULL OR f.investment_date >= $2::date)
        AND ($3::date IS NULL OR f.investment_date < $3::date + 1)
        AND ($4::text IS NULL OR f.general_funding_stage = $4::text)
    """

    async def get_funding_summary(
        self, org_uuids: list[str] | None, start_date: date | None, end_date: date | None, stage: str | None
    ) -> Row:
        """Round count, disclosure counts, total and median raise (one row).

        Pass org_uuids=None for market-wide totals under the same filters.
        """
        query = f"""
            SELECT
                count(*)::int                                  AS round_count,
                count(f.fundraise_amount_usd)::int             AS rounds_with_amount,
                count(f.valuation_usd)::int                    AS rounds_with_valuation,
                count(DISTINCT f.org_uuid)::int                AS companies_funded,
                coalesce(sum(f.fundraise_amount_usd), 0)::bigint AS total_raised_usd,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY f.fundraise_amount_usd)
                                                               AS median_raise_usd
            FROM fundingrounds f
            WHERE {self._FUNDING_FILTER}
        """
        # Aggregate without GROUP BY: always exactly one row.
        return (await self._fetch(query, org_uuids, start_date, end_date, stage))[0]

    async def get_funding_by_month(
        self, org_uuids: list[str], start_date: date | None, end_date: date | None, stage: str | None
    ) -> list[Row]:
        """Round count and total raised per month (YYYY-MM)."""
        query = f"""
            SELECT
                to_char(date_trunc('month', f.investment_date), 'YYYY-MM') AS month,
                count(*)::int                                    AS round_count,
                coalesce(sum(f.fundraise_amount_usd), 0)::bigint AS total_raised_usd
            FROM fundingrounds f
            WHERE {self._FUNDING_FILTER} AND f.investment_date IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """
        return await self._fetch(query, org_uuids, start_date, end_date, stage)

    async def get_funding_stage_mix(
        self, org_uuids: list[str], start_date: date | None, end_date: date | None, stage: str | None
    ) -> list[Row]:
        """Round count and total raised per general funding stage."""
        query = f"""
            SELECT
                coalesce(f.general_funding_stage, 'other')       AS stage,
                count(*)::int                                    AS round_count,
                coalesce(sum(f.fundraise_amount_usd), 0)::bigint AS total_raised_usd
            FROM fundingrounds f
            WHERE {self._FUNDING_FILTER}
            GROUP BY 1
            ORDER BY round_count DESC
        """
        return await self._fetch(query, org_uuids, start_date, end_date, stage)

    async def get_top_funding_deals(
        self,
        org_uuids: list[str],
        start_date: date | None,
        end_date: date | None,
        stage: str | None,
        limit: int,
    ) -> list[Row]:
        """Largest rounds with a disclosed amount, with company names."""
        query = f"""
            SELECT
                f.org_uuid::text      AS org_uuid,
                o.name                AS company_name,
                f.investment_date::date AS investment_date,
                f.stage               AS stage,
                f.fundraise_amount_usd,
                f.valuation_usd
            FROM fundingrounds f
            LEFT JOIN organizations o ON o.org_uuid = f.org_uuid
            WHERE {self._FUNDING_FILTER} AND f.fundraise_amount_usd IS NOT NULL
            ORDER BY f.fundraise_amount_usd DESC
            LIMIT $5
        """
        return await self._fetch(query, org_uuids, start_date, end_date, stage, limit)

    # ----------------------------------------------------------------------- #
    # Acquisitions
    # ----------------------------------------------------------------------- #

    # $1 = acquiree org_uuids, $2 = start_year, $3 = end_year, $4 = acquirer_type
    _ACQUISITION_FILTER = """
        a.acquiree_uuid = ANY($1::uuid[])
        AND ($2::int IS NULL OR extract(year FROM a.acquisition_announce_date) >= $2::int)
        AND ($3::int IS NULL OR extract(year FROM a.acquisition_announce_date) <= $3::int)
        AND ($4::text IS NULL OR a.acquirer_type = $4::text)
    """

    async def get_acquisitions_by_year(
        self, org_uuids: list[str], start_year: int | None, end_year: int | None, acquirer_type: str | None
    ) -> list[Row]:
        """Deal count, priced deal count, and total price per announcement year."""
        query = f"""
            SELECT
                extract(year FROM a.acquisition_announce_date)::int AS year,
                count(*)::int                                       AS deal_count,
                count(a.acquisition_price_usd)::int                 AS priced_deal_count,
                coalesce(sum(a.acquisition_price_usd), 0)::bigint   AS total_price_usd
            FROM acquisitions a
            WHERE {self._ACQUISITION_FILTER} AND a.acquisition_announce_date IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """
        return await self._fetch(query, org_uuids, start_year, end_year, acquirer_type)

    async def get_top_acquirers(
        self,
        org_uuids: list[str],
        start_year: int | None,
        end_year: int | None,
        acquirer_type: str | None,
        limit: int,
    ) -> list[Row]:
        """Most active acquirers, ranked by deal count then total price."""
        query = f"""
            SELECT
                a.acquirer_uuid::text                              AS acquirer_uuid,
                max(o.name)                                        AS name,
                max(a.acquirer_type)                               AS acquirer_type,
                count(*)::int                                      AS deal_count,
                coalesce(sum(a.acquisition_price_usd), 0)::bigint  AS total_price_usd
            FROM acquisitions a
            LEFT JOIN organizations o ON o.org_uuid = a.acquirer_uuid
            WHERE {self._ACQUISITION_FILTER}
            GROUP BY a.acquirer_uuid
            ORDER BY deal_count DESC, total_price_usd DESC
            LIMIT $5
        """
        return await self._fetch(query, org_uuids, start_year, end_year, acquirer_type, limit)

    async def count_acquisitions_by(
        self,
        column: str,
        org_uuids: list[str],
        start_year: int | None,
        end_year: int | None,
        acquirer_type: str | None,
    ) -> list[Row]:
        """Count deals grouped by one categorical column (NULL becomes 'unknown')."""
        allowed = {"acquirer_type", "acquisition_type", "terms"}
        if column not in allowed:
            raise ValueError(f"Unsupported grouping column: {column}")
        query = f"""
            SELECT coalesce(a.{column}, 'unknown') AS label, count(*)::int AS count
            FROM acquisitions a
            WHERE {self._ACQUISITION_FILTER}
            GROUP BY 1
            ORDER BY count DESC
        """
        return await self._fetch(query, org_uuids, start_year, end_year, acquirer_type)

    async def get_acquisition_price_stats(
        self, org_uuids: list[str], start_year: int | None, end_year: int | None, acquirer_type: str | None
    ) -> Row:
        """Deal count and disclosed-price statistics (one row)."""
        query = f"""
            SELECT
                count(*)::int                                      AS total_deals,
                count(a.acquisition_price_usd)::int                AS priced_deals,
                coalesce(sum(a.acquisition_price_usd), 0)::bigint  AS total_price_usd,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY a.acquisition_price_usd)
                                                                   AS median_price_usd,
                max(a.acquisition_price_usd)                       AS max_price_usd
            FROM acquisitions a
            WHERE {self._ACQUISITION_FILTER}
        """
        # Aggregate without GROUP BY: always exactly one row.
        return (await self._fetch(query, org_uuids, start_year, end_year, acquirer_type))[0]
