from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → shutdown."""
    logger.info("Starting Location Platform...")
    #await init_clients(app)
    #logger.info("All clients initialised. Ready.")
    #yield
    #logger.info("Shutting down...")
    #await close_clients(app)

app = FastAPI(
    title="Maverick - Location Infrastructure Platform",
    description="Multilingual RAG-based hotel concierge powered by Claude.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

