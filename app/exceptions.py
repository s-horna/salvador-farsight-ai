"""Domain exceptions raised by the service and repository layers."""


class SectorNotFoundError(Exception):
    """No organizations matched the requested sector above the similarity cutoff."""

    def __init__(self, sector: str) -> None:
        self.sector = sector
        super().__init__(
            f"No companies matched the sector '{sector}'. Try a broader or more common description."
        )


class UpstreamServiceError(Exception):
    """An external dependency (OpenAI, Pinecone, Postgres) failed or timed out."""

    def __init__(self, service: str, message: str = "request failed") -> None:
        self.service = service
        super().__init__(f"{service} {message}")
