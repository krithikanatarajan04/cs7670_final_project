from dataclasses import dataclass, field
from typing import Optional
import uuid
import hashlib
import time

from pipeline.embeddings import embed_batch


@dataclass
class SourceNode:
    url: str
    fetch_timestamp: float
    content_hash: str
    snippet: Optional[str] = None
    parsed_content: Optional[str] = None
    retrieval_score: Optional[float] = None
    discovery_round: int = 0
    discovery_query: str = ""
    was_sanitized: bool = False
    snippet_embedding: Optional[list] = None
    body_embedding: Optional[list] = None
    entity_counts: dict = field(default_factory=dict)


@dataclass
class ClaimNode:
    claim_id: str
    text: str
    source_url: str
    subject_entity: str
    dimension: str
    lineage_query: str
    embedding: Optional[list] = None


@dataclass
class RecommendationNode:
    recommendation_id: str
    subject_entity: str
    rank: int
    supporting_claim_ids: list
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisActivityNode:
    activity_id: str
    timestamp: float
    input_claim_ids: list
    ordered_entities: list
    metadata: dict = field(default_factory=dict)


class ProvenanceGraph:
    def __init__(self):
        self.sources:             dict[str, SourceNode] = {}
        self.claims:              dict[str, ClaimNode] = {}
        self.recommendations:     dict[str, RecommendationNode] = {}
        self.analysis_activities: dict[str, AnalysisActivityNode] = {}
        self.entity_index:        dict[str, list[str]] = {}
        self.source_url_index:    dict[str, list[str]] = {}
        self.dimension_index:     dict[str, list[str]] = {}
        self.coverage_snapshots:  list = []

    def add_source(self, url, snippet=None, parsed_content=None,
                   retrieval_score=None, discovery_round=0,
                   discovery_query="", was_sanitized=False, dimension=None) -> str:
        content_hash = hashlib.sha256(parsed_content.encode()).hexdigest() if parsed_content else ""
        self.sources[url] = SourceNode(
            url=url,
            fetch_timestamp=time.time(),
            content_hash=content_hash,
            snippet=snippet,
            parsed_content=parsed_content,
            retrieval_score=retrieval_score,
            discovery_round=discovery_round,
            discovery_query=discovery_query,
            was_sanitized=was_sanitized,
        )
        if dimension is not None:
            self.dimension_index.setdefault(dimension, [])
            if url not in self.dimension_index[dimension]:
                self.dimension_index[dimension].append(url)
        return url

    def add_claim(self, text, source_url, subject_entity, dimension, lineage_query="") -> str:
        claim_id = str(uuid.uuid4())
        self.claims[claim_id] = ClaimNode(
            claim_id=claim_id,
            text=text,
            source_url=source_url,
            subject_entity=subject_entity,
            dimension=dimension,
            lineage_query=lineage_query,
        )
        self.entity_index.setdefault(subject_entity, []).append(claim_id)
        self.source_url_index.setdefault(source_url, []).append(claim_id)
        self.dimension_index.setdefault(dimension, [])
        if source_url not in self.dimension_index[dimension]:
            self.dimension_index[dimension].append(source_url)

        if source_url in self.sources:
            ec = self.sources[source_url].entity_counts
            ec[subject_entity] = ec.get(subject_entity, 0) + 1

        return claim_id

    def add_coverage_snapshots(self, snapshots: list) -> None:
        self.coverage_snapshots = snapshots

    def add_recommendation(self, subject_entity, rank, supporting_claim_ids, reasoning) -> str:
        rec_id = str(uuid.uuid4())
        self.recommendations[rec_id] = RecommendationNode(
            recommendation_id=rec_id,
            subject_entity=subject_entity,
            rank=rank,
            supporting_claim_ids=supporting_claim_ids,
            metadata={"reasoning": reasoning},
        )
        return rec_id

    def add_analysis_activity(self, input_claim_ids, ordered_entities, reasoning, **metadata) -> str:
        activity_id = str(uuid.uuid4())
        self.analysis_activities[activity_id] = AnalysisActivityNode(
            activity_id=activity_id,
            timestamp=time.time(),
            input_claim_ids=input_claim_ids,
            ordered_entities=ordered_entities,
            metadata={**metadata, "reasoning": reasoning},
        )
        return activity_id

    # --- Traversal ---

    def get_source(self, url: str) -> SourceNode:
        return self.sources[url]

    def get_claims_for_source(self, url: str) -> list:
        return [self.claims[cid] for cid in self.source_url_index.get(url, [])
                if cid in self.claims]

    def get_claims_for_entity(self, entity: str) -> list:
        return [self.claims[cid] for cid in self.entity_index.get(entity, [])
                if cid in self.claims]

    def get_sources_for_dimension(self, dimension: str) -> list:
        return [self.sources[url] for url in self.dimension_index.get(dimension, [])
                if url in self.sources]

    def get_snippet_and_body(self, url: str) -> tuple:
        s = self.sources[url]
        return s.snippet, s.parsed_content

    def get_all_claim_texts(self) -> list:
        return [(c.text, c.source_url) for c in self.claims.values()]

    def get_coverage_trajectory(self) -> list:
        return self.coverage_snapshots

    def get_all_entities(self) -> list:
        return list(self.entity_index.keys())

    def to_dict(self) -> dict:
        return {
            "sources": {
                url: {
                    "url":             s.url,
                    "snippet":         s.snippet,
                    "retrieval_score": s.retrieval_score,
                    "discovery_round": s.discovery_round,
                    "discovery_query": s.discovery_query,
                    "was_sanitized":   s.was_sanitized,
                    "content_hash":    s.content_hash,
                    "entity_counts":   s.entity_counts,
                }
                for url, s in self.sources.items()
            },
            "claims": {
                cid: {
                    "text":           c.text,
                    "source_url":     c.source_url,
                    "subject_entity": c.subject_entity,
                    "dimension":      c.dimension,
                    "lineage_query":  c.lineage_query,
                }
                for cid, c in self.claims.items()
            },
            "activities": {
                aid: {"ordered": a.ordered_entities, "metadata": a.metadata}
                for aid, a in self.analysis_activities.items()
            },
            "coverage_snapshots": self.coverage_snapshots,
        }