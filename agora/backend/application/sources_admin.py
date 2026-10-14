"""Use-case for managing the fixed-source list: promoting a newly-found
site, through the SourceStore port."""

from agora.backend.application.ports import SourceStore
from agora.backend.domain.schemas import FixedSource


def promote_source(
    store: SourceStore,
    name: str, url: str, city: str, promoted_by: str | None = None, is_general_listing: bool = False,
) -> bool:
    sources = store.load_fixed_sources()
    if any(s.url == url for s in sources):
        return False
    sources.append(FixedSource(
        name=name, url=url, city=city, promoted_by=promoted_by, is_general_listing=is_general_listing,
    ))
    store.save_fixed_sources(sources)
    return True
