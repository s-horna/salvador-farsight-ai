"""The evaluation query set and its weak relevance labels."""

import json
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from app.config import Settings

QUERIES_FILE = Path(__file__).with_name("sector_queries.json")

# Kinds with a set of relevant companies, scored with precision / recall / F1.
LABELLED_KINDS = ("broad", "medium", "narrow")
# Kinds with no labels: nonsense should be rejected; ambiguous is reported only.
UNLABELLED_KINDS = ("ambiguous", "nonsense")


@dataclass(frozen=True)
class EvalQuery:
    text: str
    kind: str
    categories: tuple[str, ...]

    @property
    def is_labelled(self) -> bool:
        return self.kind in LABELLED_KINDS


def load_queries(path: Path = QUERIES_FILE) -> list[EvalQuery]:
    """Read the query set, checking each entry's kind and labels are consistent."""
    queries = [
        EvalQuery(text=item["text"], kind=item["kind"], categories=tuple(item["categories"]))
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    for query in queries:
        if query.kind not in LABELLED_KINDS + UNLABELLED_KINDS:
            raise ValueError(f"Unknown kind '{query.kind}' for query '{query.text}'")
        if query.is_labelled != bool(query.categories):
            raise ValueError(
                f"Query '{query.text}' ({query.kind}): labelled kinds need categories, "
                "unlabelled kinds must have none"
            )
    return queries


async def load_relevant_org_uuids(
    settings: Settings, queries: list[EvalQuery]
) -> dict[str, frozenset[str]]:
    """Map each labelled query's text to the companies carrying any of its categories.

    Labels come from `organizations.categories` in Postgres, the same values as the
    Pinecone `categories` metadata, read here because SQL returns the full set where
    a Pinecone query is limited to top_k. They are weak labels: a company can belong
    to a sector without carrying the category.
    """
    connection = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        database=settings.postgres_db_name,
        timeout=settings.postgres_command_timeout_seconds,
    )
    try:
        labelled = [query for query in queries if query.is_labelled]
        wanted = sorted({category for query in labelled for category in query.categories})
        rows = await connection.fetch(
            """
            SELECT DISTINCT category
            FROM organizations, unnest(categories) AS category
            WHERE org_type = 'company' AND category = ANY($1::text[])
            """,
            wanted,
        )
        unknown = set(wanted) - {row["category"] for row in rows}
        if unknown:
            raise ValueError(f"Categories not found on any company: {sorted(unknown)}")

        relevant: dict[str, frozenset[str]] = {}
        for query in labelled:
            rows = await connection.fetch(
                """
                SELECT org_uuid::text AS org_uuid
                FROM organizations
                WHERE org_type = 'company' AND categories && $1::text[]
                """,
                list(query.categories),
            )
            relevant[query.text] = frozenset(row["org_uuid"] for row in rows)
        return relevant
    finally:
        await connection.close()
