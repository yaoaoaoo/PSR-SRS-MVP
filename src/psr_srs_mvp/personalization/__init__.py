"""User behaviour profiling and personalized re-ranking."""

from psr_srs_mvp.personalization.split import split_events, load_events
from psr_srs_mvp.personalization.profiles import (
    UserProfile, build_profiles, load_items, load_users_map,
)
from psr_srs_mvp.personalization.reranker import (
    PersonalizationConfig, RankedItem, rerank_candidates,
)
from psr_srs_mvp.personalization.evaluation import (
    compute_behavior_metrics, compute_qrels_metrics, macro_average_dict,
    compute_candidate_coverage,
)

__all__ = [
    "split_events", "load_events",
    "UserProfile", "build_profiles", "load_items", "load_users_map",
    "PersonalizationConfig", "RankedItem", "rerank_candidates",
    "compute_behavior_metrics", "compute_qrels_metrics", "macro_average_dict",
    "compute_candidate_coverage",
]
