from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    serpapi_key: str = ""
    serper_api_key: str = ""
    google_api_key: str = ""
    google_cse_id: str = ""

    # Single source of truth for the database location. Everything that opens
    # the DB derives its path from here (previously store.py hardcoded
    # "data/agora.db" while an unused database_url pointed somewhere else).
    db_path: str = "data/agora.db"
    # Postgres connection string (Supabase). Required in production — store.py's
    # connection pool reads this. Use Supabase's pooled ("Transaction mode",
    # port 6543) URI, not the direct 5432 one.
    database_url: str = ""

    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_plans_per_run: int = 30
    search_provider: str = "serpapi"
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.0-flash"

    # ── CORS ─────────────────────────────────────────────────
    # Comma-separated list of origins allowed to call the API.
    # The bundled UI is served same-origin, so the default only needs the
    # local dev hosts. Add your deployed origin here (or via the CORS_ORIGINS
    # env var) instead of using "*".
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ── Scraper safety (SSRF / DoS guards) ───────────────────
    # When False (default), the scraper refuses URLs that resolve to private,
    # loopback, link-local or otherwise non-public IP addresses. Set to True
    # only if you deliberately need to scrape hosts on a trusted internal network.
    scraper_allow_private_hosts: bool = False
    # Redirects are followed manually and each hop is re-validated.
    scraper_max_redirects: int = 3
    # Hard cap on a fetched response body, to bound memory on hostile pages.
    scraper_max_bytes: int = 5_000_000

    model_config = {"env_prefix": "", "env_file": ".env"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()