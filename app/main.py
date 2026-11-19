"""Application entry point: builds the FastAPI app and wires its dependencies.

This is the composition root. Long-lived resources
(Postgres pool, Pinecone index, OpenAI client) are opened once at startup and
closed at shutdown. The service is built here and stored on app.state, where
routes.get_trend_service picks it up for each request.

Run with:  uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.repository import TrendRepository
from app.routes import register_exception_handlers, router
from app.service import TrendService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass `settings` to override .env values (e.g. in tests)."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: open connections once and build the object graph by hand.
        repository = await TrendRepository.connect(settings)
        try:
            app.state.service = TrendService(repository, settings)
            logger.info("Startup complete: repository connected, service ready")
            yield
        finally:
            # Shutdown: release the Postgres pool and HTTP clients.
            await repository.close()
            logger.info("Shutdown complete: repository closed")

    app = FastAPI(
        title="Sector Trends API",
        description=(
            "Funding and M&A trends for a user-supplied sector, combining "
            "Pinecone (which companies are in the sector) with Postgres "
            "(what happened to them)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
