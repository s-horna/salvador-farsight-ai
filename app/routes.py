"""HTTP layer: routes, input validation, dependency wiring, and error mapping.

Rules: no SQL and no SDK calls here. Each route validates its
input, calls TrendService, and returns the response model. Domain errors raised
by the service are mapped to HTTP status codes by the handlers registered in
`register_exception_handlers`.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response, status
from fastapi.responses import JSONResponse

from app.exceptions import SectorNotFoundError, UpstreamServiceError
from app.schemas import (
    AcquisitionTrendRequest,
    AcquisitionTrendResponse,
    ErrorResponse,
    FundingTrendRequest,
    FundingTrendResponse,
    HealthResponse,
)
from app.service import TrendService

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_trend_service(request: Request) -> TrendService:
    """Return the TrendService built once at startup.

    Kept as a dependency so tests can swap it via app.dependency_overrides.
    """
    return request.app.state.service


TrendServiceDep = Annotated[TrendService, Depends(get_trend_service)]

_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No companies matched the sector above the similarity cutoff.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "An upstream dependency (OpenAI, Pinecone, Postgres) failed.",
    },
}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check Postgres, Pinecone, and OpenAI connectivity",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def check_health(service: TrendServiceDep, response: Response) -> HealthResponse:
    result = await service.check_health()
    if not result.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get(
    "/trends/funding",
    response_model=FundingTrendResponse,
    tags=["trends"],
    summary="Sector Funding Snapshot",
    description=(
        "How much capital is flowing into a sector and at what stage: monthly "
        "series, stage mix, top deals, and the sector's share of all rounds."
    ),
    responses=_ERROR_RESPONSES,
)
async def get_funding_trend(
    params: Annotated[FundingTrendRequest, Query()],
    service: TrendServiceDep,
) -> FundingTrendResponse:
    return await service.get_funding_trend(params)


@router.get(
    "/trends/acquisitions",
    response_model=AcquisitionTrendResponse,
    tags=["trends"],
    summary="Sector M&A Landscape",
    description=(
        "Who is consolidating a sector: deals by year, top acquirers, "
        "strategic vs. financial buyer mix, deal types, terms, and price stats."
    ),
    responses=_ERROR_RESPONSES,
)
async def get_acquisition_trend(
    params: Annotated[AcquisitionTrendRequest, Query()],
    service: TrendServiceDep,
) -> AcquisitionTrendResponse:
    return await service.get_acquisition_trend(params)


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


async def _sector_not_found_handler(request: Request, exc: SectorNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(error="sector_not_found", detail=str(exc)).model_dump(),
    )


async def _upstream_error_handler(request: Request, exc: UpstreamServiceError) -> JSONResponse:
    logger.error("Upstream failure on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(error="upstream_unavailable", detail=str(exc)).model_dump(),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_error", detail="Unexpected server error.").model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions to HTTP responses. Called from main.create_app()."""
    app.add_exception_handler(SectorNotFoundError, _sector_not_found_handler)
    app.add_exception_handler(UpstreamServiceError, _upstream_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
