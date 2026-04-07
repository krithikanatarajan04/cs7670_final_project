from dataclasses import dataclass, field
from typing import Optional, Any
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
        self.sources: dict[str, SourceNode] = {}         # keyed by URL
        self.claims: dict[str, ClaimNode] = {}           # keyed by claim_id UUID
        self.recommendations: dict[str, RecommendationNode] = {}
        self.analysis_activities: dict[str, AnalysisActivityNode] = {}

        # Secondary indices
        self.entity_index: dict[str, list[str]] = {}     # entity -> [claim_ids]
        self.source_url_index: dict[str, list[str]] = {} # url -> [claim_ids]
        self.dimension_index: dict[str, list[str]] = {}  # dimension -> [urls]

        self.coverage_snapshots: list = []

    def add_source(self, url: str, snippet: str = None, parsed_content: str = None,
                   retrieval_score: float = None, discovery_round: int = 0,
                   discovery_query: str = "", was_sanitized: bool = False,
                   dimension: str = None) -> str:
        content_hash = hashlib.sha256(parsed_content.encode()).hexdigest() if parsed_content is not None else ""
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
            if dimension not in self.dimension_index:
                self.dimension_index[dimension] = []
            if url not in self.dimension_index[dimension]:
                self.dimension_index[dimension].append(url)
        return url

    def add_claim(self, text: str, source_url: str, subject_entity: str,
                  dimension: str, lineage_query: str = "") -> str:
        claim_id = str(uuid.uuid4())
        self.claims[claim_id] = ClaimNode(
            claim_id=claim_id,
            text=text,
            source_url=source_url,
            subject_entity=subject_entity,
            dimension=dimension,
            lineage_query=lineage_query,
        )
        if subject_entity not in self.entity_index:
            self.entity_index[subject_entity] = []
        self.entity_index[subject_entity].append(claim_id)

        if source_url not in self.source_url_index:
            self.source_url_index[source_url] = []
        self.source_url_index[source_url].append(claim_id)

        if dimension not in self.dimension_index:
            self.dimension_index[dimension] = []
        if source_url not in self.dimension_index[dimension]:
            self.dimension_index[dimension].append(source_url)

        # Maintain entity_counts on SourceNode — free during construction
        if source_url in self.sources:
            self.sources[source_url].entity_counts[subject_entity] = (
                self.sources[source_url].entity_counts.get(subject_entity, 0) + 1
            )

        return claim_id

    def add_coverage_snapshots(self, snapshots: list) -> None:
        self.coverage_snapshots = snapshots

    def add_recommendation(self, subject_entity: str, rank: int,
                           supporting_claim_ids: list, reasoning: str) -> str:
        rec_id = str(uuid.uuid4())
        self.recommendations[rec_id] = RecommendationNode(
            recommendation_id=rec_id,
            subject_entity=subject_entity,
            rank=rank,
            supporting_claim_ids=supporting_claim_ids,
            metadata={"reasoning": reasoning}
        )
        return rec_id

    def add_analysis_activity(self, input_claim_ids: list, ordered_entities: list,
                              reasoning: str, **metadata) -> str:
        activity_id = str(uuid.uuid4())
        self.analysis_activities[activity_id] = AnalysisActivityNode(
            activity_id=activity_id,
            timestamp=time.time(),
            input_claim_ids=input_claim_ids,
            ordered_entities=ordered_entities,
            metadata={**metadata, "reasoning": reasoning}
        )
        return activity_id

    # --- Traversal Methods ---

    def get_source(self, url: str) -> SourceNode:
        return self.sources[url]

    def get_claims_for_source(self, url: str) -> list:
        claim_ids = self.source_url_index.get(url, [])
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def get_claims_for_entity(self, entity: str) -> list:
        claim_ids = self.entity_index.get(entity, [])
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def get_sources_for_dimension(self, dimension: str) -> list:
        urls = self.dimension_index.get(dimension, [])
        return [self.sources[url] for url in urls if url in self.sources]

    def get_snippet_and_body(self, url: str) -> tuple:
        source = self.sources[url]
        return (source.snippet, source.parsed_content)

    def get_all_claim_texts(self) -> list:
        return [(c.text, c.source_url) for c in self.claims.values()]

    def get_coverage_trajectory(self) -> list:
        return self.coverage_snapshots

    def get_all_entities(self) -> list:
        return list(self.entity_index.keys())

    def populate_embeddings_for_signal(self, signal_type: str) -> None:
        import numpy as np

        if signal_type in ("focus", "coverage_entropy"):
            return

        if signal_type == "snippet_divergence":
            sources_needing = [
                (url, s) for url, s in self.sources.items()
                if s.snippet is not None and s.parsed_content is not None
                and (s.snippet_embedding is None or s.body_embedding is None)
            ]
            if not sources_needing:
                return
            texts = []
            for _, s in sources_needing:
                texts.append(s.snippet)
                texts.append(s.parsed_content)
            embeddings = embed_batch(texts)
            for i, (url, s) in enumerate(sources_needing):
                s.snippet_embedding = embeddings[i * 2].tolist()
                s.body_embedding = embeddings[i * 2 + 1].tolist()

        elif signal_type == "corroboration":
            claims_needing = [c for c in self.claims.values() if c.embedding is None]
            if not claims_needing:
                return
            embeddings = embed_batch([c.text for c in claims_needing])
            for claim, emb in zip(claims_needing, embeddings):
                claim.embedding = emb.tolist()

    def populate_all_embeddings(self, signals: list) -> None:
        """
        Combined batched embedding call when multiple phase 2 signals are active.
        Collapses snippet_divergence and corroboration into a single embed_batch call.
        Falls back to individual signal behavior if only one is active.
        """
        import numpy as np

        do_snippet = "snippet_divergence" in signals
        do_corroboration = "corroboration" in signals

        if not do_snippet and not do_corroboration:
            return

        if do_snippet and not do_corroboration:
            self.populate_embeddings_for_signal("snippet_divergence")
            return

        if do_corroboration and not do_snippet:
            self.populate_embeddings_for_signal("corroboration")
            return

        # Both active — single batched call
        # Collect snippet/body pairs
        sources_needing_snippets = [
            (url, s) for url, s in self.sources.items()
            if s.snippet is not None and s.parsed_content is not None
            and (s.snippet_embedding is None or s.body_embedding is None)
        ]
        # Collect claims needing embeddings
        claims_needing = [c for c in self.claims.values() if c.embedding is None]

        if not sources_needing_snippets and not claims_needing:
            return

        # Build flat labeled list: (label, text)
        labeled = []
        for url, s in sources_needing_snippets:
            labeled.append(("snippet:" + url, s.snippet))
            labeled.append(("body:" + url, s.parsed_content))
        for c in claims_needing:
            labeled.append(("claim:" + c.claim_id, c.text))

        if not labeled:
            return

        texts = [t for _, t in labeled]
        embeddings = embed_batch(texts)

        # Distribute results back
        for i, (label, _) in enumerate(labeled):
            emb = embeddings[i].tolist()
            if label.startswith("snippet:"):
                url = label[len("snippet:"):]
                if url in self.sources:
                    self.sources[url].snippet_embedding = emb
            elif label.startswith("body:"):
                url = label[len("body:"):]
                if url in self.sources:
                    self.sources[url].body_embedding = emb
            elif label.startswith("claim:"):
                cid = label[len("claim:"):]
                if cid in self.claims:
                    self.claims[cid].embedding = emb

    def to_dict(self) -> dict:
        return {
            "sources": {
                url: {
                    "url": s.url,
                    "snippet": s.snippet,
                    "retrieval_score": s.retrieval_score,
                    "discovery_round": s.discovery_round,
                    "discovery_query": s.discovery_query,
                    "was_sanitized": s.was_sanitized,
                    "content_hash": s.content_hash,
                    "entity_counts": s.entity_counts,
                }
                for url, s in self.sources.items()
            },
            "claims": {
                cid: {
                    "text": c.text,
                    "source_url": c.source_url,
                    "subject_entity": c.subject_entity,
                    "dimension": c.dimension,
                    "lineage_query": c.lineage_query,
                }
                for cid, c in self.claims.items()
            },
            "activities": {
                aid: {"ordered": a.ordered_entities, "metadata": a.metadata}
                for aid, a in self.analysis_activities.items()
            },
            "coverage_snapshots": self.coverage_snapshots,
        }