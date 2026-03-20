"""
agents/verifier.py

The Verifier is the core detection component of the defense system.
It sits between the Analyzer and the RecommendationAgent.

defense_config behaviour:
    "no_defense" / "controlvalve_only":
        CFG transition recorded. All detection fields set to None/False.
        State passes through untouched.

    "observe":
        Full detection runs. All fields populated. But defense_triggered
        is always False — the pipeline output is never altered. Use this
        to measure the signal without intervening.

    "full_system":
        Full detection runs. defense_triggered reflects actual detections.

Detection has two checks, both grounded in graph traversal:

    Check A — Source Concentration
        For the top-ranked hotel, compute what fraction of its claims
        come from each source URL. Flag any source whose claim fraction
        exceeds CONCENTRATION_THRESHOLD. Catches attacks where a single
        attacker page dominates the evidence set.

    Check B — Cross-Source Coordination
        For every pair of sources that contributed claims to the top
        hotel, compute mean pairwise cosine similarity between Source A's
        claims and Source B's claims. Flag pairs whose cross-source
        similarity is anomalously high relative to all other source pairs
        in the same run. Catches coordinated attacker pages generated to
        assert the same facts.

        This check requires the provenance graph. Without knowing which
        claims came from which source, cross-source similarity is
        impossible to compute. RAGDEFENDER cannot do this because it
        discards source identity after retrieval.

Both checks use:
    graph.get_claims_grouped_by_source(research_target)
        -> {source_url: [ClaimNode, ...]}

Design notes:
    - Content hashing deferred (see provenance.py).
    - criterion preserved from state["claims"] in provenance log.
    - flagged_sources is the primary detection output.
"""

import numpy as np
from itertools import combinations
from typing import Optional

from pipeline.cfg import record_transition
from pipeline.provenance import ProvenanceGraph
from pipeline.embeddings import embed_batch, pairwise_cosine_similarity


# ---------------------------------------------------------------------- #
# Thresholds                                                               #
# ---------------------------------------------------------------------- #

# Check A: flag a source if its claim fraction is more than this many
# standard deviations above the mean claim fraction across all sources
# for the top hotel. Using a relative threshold rather than a fixed one
# means the check works at any k — with 3 sources or 30 sources, an
# attacker page that dominates the evidence will always be anomalously
# far above the mean. 1.0 std flags the top ~16% of sources, which is
# appropriate when you expect 0-2 attacker pages in a retrieved set.
CONCENTRATION_STD_MULTIPLIER = 1.0

# Check B: flag a source pair if their cross-source similarity exceeds
# the mean cross-source similarity by this many standard deviations.
COORDINATION_STD_MULTIPLIER = 1.0

# Check B minimum source count. With fewer than this many eligible
# sources, the std computation across pairwise scores is too noisy
# to be reliable. At k=5 with one dominant attacker page, you typically
# end up with 2-3 eligible sources — not enough. At k=8+ you reliably
# get 4+. This guard makes Check B only fire when it has enough data.
MIN_SOURCES_FOR_COORDINATION = 4

# Minimum claims per source to be eligible for Check B embedding.
MIN_CLAIMS_FOR_COORDINATION = 2


# ---------------------------------------------------------------------- #
# Internal helpers                                                         #
# ---------------------------------------------------------------------- #

def _mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    sim_matrix = pairwise_cosine_similarity(embeddings)
    upper_i, upper_j = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[upper_i, upper_j]))


def _mean_cross_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> float:
    """
    Mean cosine similarity between every claim in source A and every
    claim in source B. Cross-source only — no within-source pairs.

    Two coordinated attacker sources will score high here because they
    assert the same fabricated facts. Two independent legitimate sources
    will score lower because they were written independently.
    """
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return 0.0
    cross_matrix = (embeddings_a @ embeddings_b.T).astype(np.float32)
    return float(np.mean(cross_matrix))


def _embed_claims(claims: list) -> np.ndarray:
    if not claims:
        return np.array([], dtype=np.float32).reshape(0, 384)
    return embed_batch([c.claim_text for c in claims])


# ---------------------------------------------------------------------- #
# Verifier node                                                            #
# ---------------------------------------------------------------------- #

def verifier_node(state: dict, defense_config: str = "no_defense") -> dict:
    """
    Entry point for the Verifier agent.

    Args:
        state:          Pipeline state dict passed by LangGraph.
        defense_config: Injected by orchestrator via functools.partial.

    Returns:
        Updated state dict with all Verifier fields populated.
    """

    # 1. CFG guardrail — always fires
    record_transition("Verifier")

    # 2. Early return for non-detection configs
    if defense_config not in ("full_system", "observe"):
        state["concentration_flag"] = False
        state["provenance_log"] = None
        state["concentration_score"] = None
        state["concentration_flagged"] = None
        state["flagged_sources"] = None
        state["defense_triggered"] = None
        return state

    print(f"\n[Verifier] Running detection (mode: {defense_config}).")

    claims = state.get("claims", [])
    rankings = state.get("rankings", [])
    reasoning = state.get("reasoning", "")

    # 3. Degenerate case
    if not rankings:
        print("[Verifier] No rankings — skipping.")
        state["concentration_flag"] = False
        state["provenance_log"] = None
        state["concentration_score"] = None
        state["concentration_flagged"] = False
        state["flagged_sources"] = []
        state["defense_triggered"] = False
        return state

    top_hotel = rankings[0]
    print(f"[Verifier] Top-ranked hotel: {top_hotel}")

    # 4. Build provenance graph
    graph = ProvenanceGraph()
    url_to_source_id: dict[str, str] = {}

    for claim in claims:
        url = claim["source_url"]
        if url not in url_to_source_id:
            source_id = graph.add_source(url=url, content="")
            url_to_source_id[url] = source_id

    for claim in claims:
        graph.add_claim(
            claim_text=claim["text"],
            source_id=url_to_source_id[claim["source_url"]],
            research_target=claim["hotel"]
        )

    top_claim_nodes = graph.get_claims_for_target(top_hotel)
    top_claim_ids = [c.claim_id for c in top_claim_nodes]

    graph.add_recommendation(
        research_target=top_hotel,
        rank=1,
        supporting_claim_ids=top_claim_ids,
        reasoning=reasoning
    )

    print(f"[Verifier] Graph built: {len(graph.sources)} sources, "
          f"{len(top_claim_nodes)} claims for {top_hotel}.")

    # 5. Graph traversal — group claims by source
    # This is the key step. Everything downstream uses this structure.
    grouped = graph.get_claims_grouped_by_source(top_hotel)
    total_claims = len(top_claim_nodes)

    if total_claims == 0:
        print(f"[Verifier] No claims for {top_hotel} — clean.")
        state["concentration_flag"] = False
        state["provenance_log"] = _build_provenance_log(claims, top_hotel, set())
        state["concentration_score"] = 0.0
        state["concentration_flagged"] = False
        state["flagged_sources"] = []
        state["defense_triggered"] = False
        return state

    print(f"[Verifier] Sources: " + ", ".join(
        f"{url.split('/')[-2]}({len(c)})" for url, c in grouped.items()
    ))

    # ------------------------------------------------------------------ #
    # Check A — Source Concentration                                       #
    # ------------------------------------------------------------------ #
    # Compute claim fraction per source, then flag any source whose
    # fraction is more than CONCENTRATION_STD_MULTIPLIER standard
    # deviations above the mean fraction across all sources.
    #
    # Relative threshold generalises across any k. A legitimate evidence
    # set has a roughly flat fraction distribution — no single source
    # dominates. An attacker page flooding evidence for one hotel will
    # be far above the mean regardless of total source count.

    concentration_results: dict[str, float] = {
        url: len(source_claims) / total_claims
        for url, source_claims in grouped.items()
    }

    fractions = list(concentration_results.values())
    mean_frac = float(np.mean(fractions))
    std_frac = float(np.std(fractions))
    concentration_threshold = mean_frac + CONCENTRATION_STD_MULTIPLIER * std_frac

    concentration_flagged_urls = {
        url for url, frac in concentration_results.items()
        if frac > concentration_threshold
    }

    concentration_score = max(fractions) if fractions else 0.0
    concentration_flagged = len(concentration_flagged_urls) > 0

    print(f"[Verifier] Check A: mean={mean_frac:.1%}, std={std_frac:.1%}, "
          f"threshold={concentration_threshold:.1%}, max={concentration_score:.1%}, "
          f"flagged={set(u.split('/')[-2] for u in concentration_flagged_urls) or 'none'}")

    # ------------------------------------------------------------------ #
    # Check B — Cross-Source Coordination                                  #
    # ------------------------------------------------------------------ #
    # Only runs when there are enough eligible sources to make the std
    # computation meaningful. Below MIN_SOURCES_FOR_COORDINATION, the
    # pairwise score distribution is too small to distinguish signal
    # from natural topic overlap between legitimate sources.
    # At k=5 this typically means Check B is skipped (3 sources after
    # attacker dominance). At k=8+ it fires reliably.

    eligible_sources = {
        url: source_claims
        for url, source_claims in grouped.items()
        if len(source_claims) >= MIN_CLAIMS_FOR_COORDINATION
    }

    coordination_flagged_pairs: list[tuple[str, str]] = []
    coordination_flagged_urls: set[str] = set()
    cross_sim_scores: dict[tuple[str, str], float] = {}

    if len(eligible_sources) < MIN_SOURCES_FOR_COORDINATION:
        print(f"[Verifier] Check B skipped: only {len(eligible_sources)} eligible "
              f"sources (need {MIN_SOURCES_FOR_COORDINATION}).")
    else:
        # Embed each source's claims in one forward pass per source
        source_embeddings: dict[str, np.ndarray] = {
            url: _embed_claims(source_claims)
            for url, source_claims in eligible_sources.items()
        }

        # Compute cross-source similarity for every eligible pair
        for url_a, url_b in combinations(list(eligible_sources.keys()), 2):
            score = _mean_cross_similarity(
                source_embeddings[url_a],
                source_embeddings[url_b]
            )
            cross_sim_scores[(url_a, url_b)] = score
            print(f"[Verifier] Check B: "
                  f"{url_a.split('/')[-2]} x {url_b.split('/')[-2]} "
                  f"= {score:.4f}")

        scores = list(cross_sim_scores.values())
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        threshold = mean_score + COORDINATION_STD_MULTIPLIER * std_score
        print(f"[Verifier] Check B stats: mean={mean_score:.4f}, "
              f"std={std_score:.4f}, threshold={threshold:.4f}")

        for (url_a, url_b), score in cross_sim_scores.items():
            if score > threshold:
                coordination_flagged_pairs.append((url_a, url_b))
                coordination_flagged_urls.update([url_a, url_b])
                print(f"[Verifier] Check B FIRED: "
                      f"{url_a.split('/')[-2]} x {url_b.split('/')[-2]} "
                      f"(score={score:.4f} > threshold={threshold:.4f})")

    # ------------------------------------------------------------------ #
    # Build flagged_sources                                                #
    # ------------------------------------------------------------------ #

    all_flagged_urls = concentration_flagged_urls | coordination_flagged_urls

    # coordination partner map
    coordination_partners: dict[str, list[str]] = {}
    for url_a, url_b in coordination_flagged_pairs:
        coordination_partners.setdefault(url_a, []).append(url_b)
        coordination_partners.setdefault(url_b, []).append(url_a)

    # max cross-sim per url
    max_cross_sim: dict[str, float] = {}
    for (url_a, url_b), score in cross_sim_scores.items():
        max_cross_sim[url_a] = max(max_cross_sim.get(url_a, 0.0), score)
        max_cross_sim[url_b] = max(max_cross_sim.get(url_b, 0.0), score)

    flagged_sources: list[dict] = []
    for url in all_flagged_urls:
        in_conc = url in concentration_flagged_urls
        in_coord = url in coordination_flagged_urls
        reason = "both" if (in_conc and in_coord) else ("concentration" if in_conc else "coordination")

        flagged_sources.append({
            "url": url,
            "reason": reason,
            "claim_count": len(grouped.get(url, [])),
            "claim_fraction": concentration_results.get(url, 0.0),
            "coordination_partners": coordination_partners.get(url, []),
            "max_cross_similarity": max_cross_sim.get(url, 0.0),
        })

    flagged_sources.sort(key=lambda x: x["claim_fraction"], reverse=True)

    defense_triggered = (
        len(flagged_sources) > 0
        if defense_config == "full_system"
        else False
    )

    print(f"[Verifier] Done. Flagged={len(flagged_sources)}, "
          f"defense_triggered={defense_triggered}")

    # Build provenance log and write back to state
    provenance_log = _build_provenance_log(
        claims=claims,
        hotel=top_hotel,
        flagged_urls=all_flagged_urls,
        concentration_score=concentration_score,
        concentration_flagged=concentration_flagged,
        flagged_sources=flagged_sources,
    )

    state["concentration_flag"] = concentration_flagged
    state["concentration_score"] = concentration_score
    state["concentration_flagged"] = concentration_flagged
    state["provenance_log"] = provenance_log
    state["flagged_sources"] = flagged_sources
    state["defense_triggered"] = defense_triggered

    return state


# ---------------------------------------------------------------------- #
# Provenance log builder                                                   #
# ---------------------------------------------------------------------- #

def _build_provenance_log(
    claims: list[dict],
    hotel: str,
    flagged_urls: set[str],
    concentration_score: Optional[float] = None,
    concentration_flagged: Optional[bool] = None,
    flagged_sources: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Claim-level audit log for the top-ranked hotel.
    Built from state["claims"] so criterion is preserved.
    Detection summary attached to first entry.
    """
    hotel_claims = [c for c in claims if c.get("hotel") == hotel]
    log = []
    for i, claim in enumerate(hotel_claims):
        entry = {
            "claim": claim["text"],
            "source_url": claim["source_url"],
            "criterion": claim.get("criterion", "unknown"),
            "source_flagged": claim["source_url"] in flagged_urls,
        }
        if i == 0:
            entry["detection_summary"] = {
                "hotel": hotel,
                "concentration_score": concentration_score,
                "concentration_flagged": concentration_flagged,
                "flagged_sources": flagged_sources or [],
            }
        log.append(entry)
    return log