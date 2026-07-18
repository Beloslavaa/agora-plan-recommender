import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agora.backend.cinemas import CINEMA_SOURCES
from agora.backend.config import settings
from agora.backend.ingestion import store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: open the DB pool eagerly so a bad DATABASE_URL fails the boot
    # instead of the first request, then let store own the schema (plans + interactions)
    store.pool.open()
    store.init_db()
    logger.info("Agora API started — DB ready")
    yield
    store.pool.close()


app = FastAPI(title="Agora Plan Recommender", version="0.1.0", lifespan=lifespan)

# CORS: restrict to an explicit allowlist instead of "*".
# The UI is served same-origin from this app, so cross-origin access is only
# needed for separate front-ends you control — list those in settings.cors_origins.
# allow_credentials stays False: we use no cookies/auth headers, and "*"-style
# wildcards with credentials are invalid anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Response models ──────────────────────────────────────

class PlanOut(BaseModel):
    id: int
    title: str
    short_title: str = ""
    description: str
    start_date: str | None = None
    end_date: str | None = None
    url: str | None = None
    ticket_url: str | None = None
    location: str | None = None
    image_url: str | None = None
    price: float | None = None
    tags: list[str] = []
    category: str | None = None
    source_url: str
    source_type: str


class InteractionIn(BaseModel):
    user_id: str
    plan_id: int
    interaction_type: str  # "click" | "saved" | "view_link"


class AuthIn(BaseModel):
    username: str
    password: str


class RecommendationOut(BaseModel):
    plan: PlanOut
    score: float


class CinemaOut(BaseModel):
    key: str
    name: str
    image_url: str | None = None
    movie_count: int


# ── Helpers ──────────────────────────────────────────────

def _row_to_plan(row: dict) -> PlanOut:
    return PlanOut(
        id=row["id"],
        title=row["title"],
        short_title=row.get("short_title") or "",
        description=row["description"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        url=row.get("url"),
        ticket_url=row["ticket_url"],
        location=row["location"],
        image_url=row["image_url"],
        price=row["price"],
        tags=json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or []),
        category=row["category"],
        source_url=row["source_url"],
        source_type=row["source_type"],
    )


@app.get("/")
def index():
    # This is served straight off disk with no build/versioning step, so a
    # browser cache silently serving a stale copy after an edit is a real
    # trap during dev — always revalidate.
    return HTMLResponse(
        Path("index.html").read_text(encoding="utf-8"),
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


# ── Routes ───────────────────────────────────────────────

@app.post("/auth")
def auth(body: AuthIn):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username is required")
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")
    try:
        store.authenticate_user(username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"user_id": username}


@app.get("/plans")
def list_plans(
    category: str | None = None,
    location: str | None = None,
    search: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[PlanOut]:
    rows = store.list_plans(
        category=category, location=location, search=search, limit=limit, offset=offset
    )
    return [_row_to_plan(r) for r in rows]


@app.get("/plans/{plan_id}")
def get_plan(plan_id: int) -> PlanOut:
    row = store.get_plan(plan_id)
    if not row:
        raise HTTPException(status_code=404, detail="Plan not found")
    return _row_to_plan(row)


@app.post("/interactions")
def record_interaction(body: InteractionIn):
    if body.interaction_type not in ("click", "saved", "view_link"):
        raise HTTPException(status_code=422, detail="interaction_type must be 'click', 'saved', or 'view_link'")
    try:
        store.record_interaction(body.user_id, body.plan_id, body.interaction_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.delete("/interactions")
def delete_interaction(body: InteractionIn):
    # Only "saved" is meant to be reversible — click/view_link are historical
    # facts the recommender uses and aren't exposed as something to undo.
    if body.interaction_type != "saved":
        raise HTTPException(status_code=422, detail="Only 'saved' interactions can be removed")
    store.remove_interaction(body.user_id, body.plan_id, body.interaction_type)
    return {"ok": True}


@app.get("/saved/{user_id}")
def saved_plans(user_id: str) -> list[PlanOut]:
    rows = store.get_saved_plans(user_id)
    return [_row_to_plan(r) for r in rows]


@app.get("/recommendations/{user_id}")
def recommend(user_id: str, limit: int = Query(default=10, le=50)) -> list[RecommendationOut]:
    rows = store.get_recommendations(user_id, limit)
    return [
        RecommendationOut(plan=_row_to_plan(r), score=float(r["score"]))
        for r in rows
    ]


@app.get("/cinemas")
def list_cinemas() -> list[CinemaOut]:
    return [CinemaOut(**c) for c in store.list_cinemas()]


@app.get("/cinemas/{key}/plans")
def cinema_plans(key: str) -> list[PlanOut]:
    if key not in CINEMA_SOURCES:
        raise HTTPException(status_code=404, detail="Unknown cinema")
    rows = store.list_cinema_plans(key)
    return [_row_to_plan(r) for r in rows]