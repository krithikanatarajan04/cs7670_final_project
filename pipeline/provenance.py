from dataclasses import dataclass, field
from typing import Optional
import uuid
import hashlib
import time
from pipeline.embeddings import embed_batch

@dataclass
class SourceNode:
    source_id: str
    url: str
    fetch_timestamp: float
    raw_content_hash: str

@dataclass  
class ClaimNode:
    claim_id: str
    claim_text: str
    source_id: str
    research_target: str
    embedding: Optional[list] = None

@dataclass
class RecommendationNode:
    recommendation_id: str
    research_target: str
    rank: int
    supporting_claim_ids: list
    reasoning: str

@dataclass
class SourceToClaimEdge:
    source_id: str
    claim_id: str

@dataclass
class ClaimToRecommendationEdge:
    claim_id: str
    recommendation_id: str
    influence_weight: float = 1.0

class ProvenanceGraph:
    def __init__(self):
        self.sources: dict[str, SourceNode] = {}
        self.claims: dict[str, ClaimNode] = {}
        self.recommendations: dict[str, RecommendationNode] = {}
        self.source_to_claim_edges: list[SourceToClaimEdge] = []
        self.claim_to_rec_edges: list[ClaimToRecommendationEdge] = []
    
    def add_source(self, url: str, content: str) -> str:
        source_id = str(uuid.uuid4())
        self.sources[source_id] = SourceNode(
            source_id=source_id,
            url=url,
            fetch_timestamp=time.time(),
            raw_content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        return source_id
    
    def add_claim(self, claim_text: str, source_id: str, research_target: str) -> str:
        claim_id = str(uuid.uuid4())
        self.claims[claim_id] = ClaimNode(
            claim_id=claim_id,
            claim_text=claim_text,
            source_id=source_id,
            research_target=research_target
        )
        self.source_to_claim_edges.append(SourceToClaimEdge(source_id, claim_id))
        return claim_id
    
    def add_recommendation(self, research_target: str, rank: int, 
                           supporting_claim_ids: list, reasoning: str) -> str:
        rec_id = str(uuid.uuid4())
        self.recommendations[rec_id] = RecommendationNode(
            recommendation_id=rec_id,
            research_target=research_target,
            rank=rank,
            supporting_claim_ids=supporting_claim_ids,
            reasoning=reasoning
        )
        for claim_id in supporting_claim_ids:
            self.claim_to_rec_edges.append(
                ClaimToRecommendationEdge(claim_id, rec_id)
            )
        return rec_id
    
    def get_claims_for_target(self, research_target: str) -> list[ClaimNode]:
        return [c for c in self.claims.values() if c.research_target == research_target]
    
    def get_sources_for_target(self, research_target: str) -> list[SourceNode]:
        claim_ids = {c.claim_id for c in self.get_claims_for_target(research_target)}
        source_ids = {e.source_id for e in self.source_to_claim_edges if e.claim_id in claim_ids}
        return [self.sources[sid] for sid in source_ids]
    
    def get_claims_for_source_and_target(self, source_id: str, research_target: str) -> list[ClaimNode]:
        claim_ids_for_source = {e.claim_id for e in self.source_to_claim_edges if e.source_id == source_id}
        return [c for c in self.get_claims_for_target(research_target) if c.claim_id in claim_ids_for_source]

    def get_claims_grouped_by_source(self, research_target: str) -> dict[str, list[ClaimNode]]:
        """
        Traverses the graph backwards from a research target to produce
        a mapping of source_url -> [ClaimNode, ...] for all claims
        supporting that target.

        This is the core graph traversal the Verifier uses for both
        Check A (source concentration) and Check B (cross-source
        coordination). The traversal path is:

            RecommendationNode
                <- ClaimToRecommendationEdge
                    <- ClaimNode
                        <- SourceToClaimEdge
                            <- SourceNode (url)

        By grouping at the source level, the Verifier can reason about
        which sources contributed what claims — something impossible
        with a flat claim list.

        Args:
            research_target: The hotel name (matches ClaimNode.research_target).

        Returns:
            Dict mapping source URL (str) to list of ClaimNode objects.
            Sources with zero claims for this target are excluded.
        """
        # Step 1: get all claims for this target
        target_claims = self.get_claims_for_target(research_target)
        if not target_claims:
            return {}

        # Step 2: build a lookup from source_id -> source_url
        # using the SourceNode objects already in the graph
        source_id_to_url = {
            sid: node.url
            for sid, node in self.sources.items()
        }

        # Step 3: group claims by their source URL
        grouped: dict[str, list[ClaimNode]] = {}
        for claim in target_claims:
            url = source_id_to_url.get(claim.source_id, "unknown")
            if url not in grouped:
                grouped[url] = []
            grouped[url].append(claim)

        return grouped

    def to_dict(self) -> dict:
        return {
            "sources": {sid: {"url": s.url} for sid, s in self.sources.items()},
            "claims": {cid: {"text": c.claim_text} for cid, c in self.claims.items()},
            "recommendations": {rid: {"rank": r.rank} for rid, r in self.recommendations.items()}
        }
    
    def populate_embeddings(self, research_target: str):
        """Computes and stores embeddings for all claims about a given
        target. Called by the Verifier before running detection.
        Kept separate from claim creation so the Researcher and
        Analyzer don't pay the embedding cost."""
        claims = self.get_claims_for_target(research_target)
        if not claims:
            return
        
        texts = [c.claim_text for c in claims]
        embeddings = embed_batch(texts)
        
        for claim, embedding in zip(claims, embeddings):
            claim.embedding = embedding.tolist()