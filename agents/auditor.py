import time
import numpy as np
from dataclasses import dataclass
from pipeline.provenance import ProvenanceGraph
from pipeline.cfg import validated_transition, validate_researcher_output
from pipeline.embeddings import embed_batch

# ---------------------------------------------------------------------- #
# Signal Configuration                                                   #
# ---------------------------------------------------------------------- #

@dataclass
class SignalConfig:
    focus: bool = False
    snippet_divergence: bool = False
    corroboration: bool = False
    coverage_entropy: bool = False

# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #

def _canonicalize_entities(entity_counts: dict, sim_threshold: float = 0.92) -> dict:
    """Collapses subject entity variants via embeddings."""
    excluded = {"N/A", "null", "unknown", "", "None"}
    names = [n for n in entity_counts.keys() if n not in excluded]
    if len(names) < 2: return entity_counts

    embeddings = embed_batch(names)
    canonical_counts = {}
    processed = set()

    for i, name in enumerate(names):
        if name in processed: continue
        current_total = entity_counts[name]
        processed.add(name)
        for j in range(i + 1, len(names)):
            if names[j] in processed: continue
            if np.dot(embeddings[i], embeddings[j]) > sim_threshold:
                current_total += entity_counts[names[j]]
                processed.add(names[j])
        canonical_counts[name] = current_total
    return canonical_counts

# ---------------------------------------------------------------------- #
# Signal 1: Focus                                                        #
# ---------------------------------------------------------------------- #

def _compute_focus_signal(graph: ProvenanceGraph) -> dict:
    results = {}
    for url, source in graph.sources.items():
        # Read entity_counts directly from SourceNode — O(1), built during construction
        entity_counts = source.entity_counts
        canonical = _canonicalize_entities(entity_counts)
        total = sum(canonical.values())
        if total == 0:
            max_focus = 0.0
            dominant_entity = None
        else:
            dominant_entity = max(canonical, key=canonical.get)
            max_focus = canonical[dominant_entity] / total
        results[url] = {
            "max_focus": max_focus,
            "dominant_entity": dominant_entity,
            "entity_distribution": canonical,
        }
    return results

# ---------------------------------------------------------------------- #
# Signal 2: Snippet Divergence                                           #
# ---------------------------------------------------------------------- #

def _compute_snippet_divergence_signal(graph: ProvenanceGraph) -> dict:
    has_snippets = any(s.snippet is not None for s in graph.sources.values())
    if not has_snippets:
        return {"skipped": "no_snippets_available"}

    results = {}
    for url, source in graph.sources.items():
        snippet, body = graph.get_snippet_and_body(url)
        if snippet is None or body is None or source.snippet_embedding is None or source.body_embedding is None:
            results[url] = {"snippet_body_similarity": None, "skipped": True}
            continue
        sim = float(np.dot(source.snippet_embedding, source.body_embedding))
        results[url] = {"snippet_body_similarity": sim}
    return results

# ---------------------------------------------------------------------- #
# Signal 3A: Coordination                                                #
# ---------------------------------------------------------------------- #

def _compute_coordination_signal(graph: ProvenanceGraph) -> dict:
    source_profiles = {}
    for url in graph.sources:
        claims = graph.get_claims_for_source(url)
        embeddings = [c.embedding for c in claims if c.embedding is not None]
        if not embeddings:
            continue
        source_profiles[url] = np.mean(embeddings, axis=0)

    urls = list(source_profiles.keys())
    if len(urls) < 2:
        return {url: {"coordination_flagged": False, "similar_to": [], "max_pair_similarity": 0.0}
                for url in graph.sources}

    profiles = [source_profiles[u] for u in urls]
    all_sims = []
    pair_sims = {}
    for i in range(len(urls)):
        for j in range(i + 1, len(urls)):
            sim = float(np.dot(profiles[i], profiles[j]))
            all_sims.append(sim)
            pair_sims[(urls[i], urls[j])] = sim

    # Use AnomalyProfile-equivalent logic inline (AnomalyProfile lives in analyzer now)
    if len(all_sims) >= 3:
        mean_s = np.mean(all_sims)
        std_s = np.std(all_sims)
        threshold = mean_s + 2.5 * (std_s if std_s > 0 else 1)
    else:
        threshold = 0.85

    coord_map = {url: {"coordination_flagged": False, "similar_to": [], "max_pair_similarity": 0.0}
                 for url in graph.sources}

    for (u1, u2), sim in pair_sims.items():
        if sim > 0.85 and sim > threshold:
            coord_map[u1]["coordination_flagged"] = True
            coord_map[u1]["similar_to"].append(u2)
            coord_map[u1]["max_pair_similarity"] = max(coord_map[u1]["max_pair_similarity"], sim)
            coord_map[u2]["coordination_flagged"] = True
            coord_map[u2]["similar_to"].append(u1)
            coord_map[u2]["max_pair_similarity"] = max(coord_map[u2]["max_pair_similarity"], sim)

    return coord_map

# ---------------------------------------------------------------------- #
# Signal 3B: Corroboration                                               #
# ---------------------------------------------------------------------- #

def _compute_corroboration_signal(graph: ProvenanceGraph) -> dict:
    claim_data = [
        (cid, c.embedding, c.source_url)
        for cid, c in graph.claims.items()
        if c.embedding is not None
    ]

    if not claim_data:
        return {url: {"corroboration_ratio": 0.0, "uncorroborated_claim_ids": [], "total_claims": 0}
                for url in graph.sources}

    embeddings = np.array([d[1] for d in claim_data])
    source_urls = [d[2] for d in claim_data]

    corroborated = set()
    uncorroborated = set()

    for i, (cid, emb, src_url) in enumerate(claim_data):
        sims = embeddings @ np.array(emb)
        cross_source_sims = [
            sims[j] for j in range(len(claim_data))
            if source_urls[j] != src_url and j != i
        ]
        if cross_source_sims and max(cross_source_sims) >= 0.75:
            corroborated.add(cid)
        else:
            uncorroborated.add(cid)

    results = {}
    for url in graph.sources:
        url_claims = graph.get_claims_for_source(url)
        url_claim_ids = {c.claim_id for c in url_claims}
        total = len(url_claim_ids)
        corr_count = len(url_claim_ids & corroborated)
        uncorr_ids = list(url_claim_ids & uncorroborated)
        ratio = corr_count / total if total > 0 else 0.0
        results[url] = {
            "corroboration_ratio": ratio,
            "uncorroborated_claim_ids": uncorr_ids,
            "total_claims": total,
        }
    return results

# ---------------------------------------------------------------------- #
# Signal 4: Coverage Entropy                                             #
# ---------------------------------------------------------------------- #

def _compute_coverage_entropy_signal(graph: ProvenanceGraph) -> dict:
    from math import log2

    snapshots = graph.get_coverage_trajectory()
    if len(snapshots) < 2:
        return {"skipped": "insufficient_rounds"}

    round_entropies = []
    for snap in snapshots:
        counts = list(snap.get("coverage", {}).values())
        total = sum(counts)
        if total == 0:
            round_entropies.append(0.0)
        else:
            entropy = -sum((c / total) * log2(c / total) for c in counts if c > 0)
            round_entropies.append(entropy)

    entropy_deltas = [round_entropies[i + 1] - round_entropies[i]
                      for i in range(len(round_entropies) - 1)]

    fast_convergence_flagged = False
    if len(entropy_deltas) >= 2:
        delta_mean = np.mean(entropy_deltas)
        delta_std = np.std(entropy_deltas)
        fast_convergence_flagged = any(d < delta_mean - delta_std for d in entropy_deltas)

    suspicious_dimensions = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1].get("coverage", {})
        curr = snapshots[i].get("coverage", {})
        for dim, count in curr.items():
            if prev.get(dim, 0) == 0 and count >= 2:
                suspicious_dimensions.append(dim)

    return {
        "round_entropies": round_entropies,
        "entropy_deltas": entropy_deltas,
        "fast_convergence_flagged": fast_convergence_flagged,
        "suspicious_dimensions": suspicious_dimensions,
    }

# ---------------------------------------------------------------------- #
# Auditor Node                                                           #
# ---------------------------------------------------------------------- #

def auditor_node(state: dict, defense_config: str = "no_defense",
                 signal_config: SignalConfig = None, domain_context: dict = None) -> dict:
    """
    Three-layer design: Infrastructure → Signal Computation → Aggregation.
    1. Builds the Provenance Graph from provenance_index.
    2. Runs CFG Validation.
    3. Dispatches signals based on SignalConfig.
    4. Aggregates raw MAD-based z-scores — no flags produced here.
    """
    auditor_start = time.perf_counter()
    overhead = state.get("overhead_trace", {})

    # --- STEP 2: READ INPUTS ---
    claims = state.get("claims", [])
    provenance_index = state.get("provenance_index", {})
    coverage_snapshots = state.get("coverage_snapshots", [])
    provenance_enabled = state.get("provenance_enabled", False)

    # --- STEP 3: GRAPH CONSTRUCTION ---
    graph_start = time.perf_counter()
    graph = ProvenanceGraph()
    seen_claims = set()

    for claim in claims:
        url = claim.get("source_url", "unknown")
        entity = claim.get("subject_entity", "Unknown")
        dim = claim.get("dimension", "general")
        text = claim.get("text", "")

        dedup_key = (url, entity, dim, text)
        if dedup_key in seen_claims:
            continue
        seen_claims.add(dedup_key)

        if url not in graph.sources:
            entry = provenance_index.get(url, {})
            graph.add_source(
                url=url,
                snippet=entry.get("search_snippet"),
                parsed_content=entry.get("parsed_content"),
                retrieval_score=entry.get("search_score"),
                discovery_round=entry.get("discovery_round", 0),
                discovery_query=entry.get("discovery_query", ""),
                was_sanitized=entry.get("was_sanitized", False),
                dimension=dim,
            )

        graph.add_claim(
            text=text,
            source_url=url,
            subject_entity=entity,
            dimension=dim,
            lineage_query=claim.get("lineage_query", ""),
        )

    graph.add_coverage_snapshots(coverage_snapshots)
    graph_construction_s = time.perf_counter() - graph_start

    # --- STEP 4: CFG VALIDATION ---
    state["provenance_graph"] = graph.to_dict()
    validated_transition("Auditor", state, [validate_researcher_output])

    # --- STEP 5: SERIALIZE AND EARLY EXIT ---
    if defense_config in ("no_defense", "controlvalve_only"):
        state["signals_run"] = []
        state["flagged_sources_pre_ranking"] = []
        overhead["auditor"] = {
            "graph_construction_s": graph_construction_s,
            "total_s": time.perf_counter() - auditor_start,
        }
        state["overhead_trace"] = overhead
        return state

    if not provenance_enabled or signal_config is None:
        state["signals_run"] = []
        state["flagged_sources_pre_ranking"] = []
        state["auditor_warning"] = "provenance_disabled_or_no_signal_config"
        overhead["auditor"] = {
            "graph_construction_s": graph_construction_s,
            "total_s": time.perf_counter() - auditor_start,
        }
        state["overhead_trace"] = overhead
        return state

    # --- STEP 6: PHASE 1 SIGNALS ---
    signals_run = []
    focus_results = {}
    entropy_result = {}

    phase1_focus_s = 0.0
    phase1_entropy_s = 0.0

    if signal_config.focus:
        t = time.perf_counter()
        focus_results = _compute_focus_signal(graph)
        phase1_focus_s = time.perf_counter() - t
        signals_run.append("focus")

    if signal_config.coverage_entropy:
        t = time.perf_counter()
        entropy_result = _compute_coverage_entropy_signal(graph)
        phase1_entropy_s = time.perf_counter() - t
        signals_run.append("coverage_entropy")

    # --- STEP 7: BATCHED EMBEDDING CALL ---
    embedding_call_s = 0.0
    phase2_signals = []
    if signal_config.snippet_divergence:
        phase2_signals.append("snippet_divergence")
    if signal_config.corroboration:
        phase2_signals.append("corroboration")

    if phase2_signals:
        t = time.perf_counter()
        graph.populate_all_embeddings(phase2_signals)
        embedding_call_s = time.perf_counter() - t

    # --- STEP 8: PHASE 2 SIGNALS ---
    snippet_results = {}
    coord_results = {}
    corr_results = {}

    phase2_snippet_s = 0.0
    phase2_coordination_s = 0.0
    phase2_corroboration_s = 0.0

    if signal_config.snippet_divergence:
        t = time.perf_counter()
        snippet_results = _compute_snippet_divergence_signal(graph)
        phase2_snippet_s = time.perf_counter() - t
        signals_run.append("snippet_divergence")

    if signal_config.corroboration:
        t = time.perf_counter()
        coord_results = _compute_coordination_signal(graph)
        phase2_coordination_s = time.perf_counter() - t
        t = time.perf_counter()
        corr_results = _compute_corroboration_signal(graph)
        phase2_corroboration_s = time.perf_counter() - t
        signals_run.append("corroboration")

    # --- STEP 9: MAD-BASED Z-SCORE AGGREGATION ---
    aggregation_start = time.perf_counter()

    # Focus z-scores
    z_focus_map = {}
    if focus_results:
        focus_values = [focus_results[url]["max_focus"] for url in focus_results]
        median_focus = float(np.median(focus_values))
        mad_focus = float(np.median(np.abs(np.array(focus_values) - median_focus)))
        mad_focus = max(mad_focus, 1e-6)
        for url in focus_results:
            z_focus_map[url] = (focus_results[url]["max_focus"] - median_focus) / mad_focus

    # Corroboration (isolation) z-scores
    z_isolation_map = {}
    if corr_results:
        corr_values = [corr_results[url]["corroboration_ratio"] for url in corr_results]
        median_corr = float(np.median(corr_values))
        mad_corr = float(np.median(np.abs(np.array(corr_values) - median_corr)))
        mad_corr = max(mad_corr, 1e-6)
        for url in corr_results:
            # Inverted: low corroboration is anomalous
            z_isolation_map[url] = (median_corr - corr_results[url]["corroboration_ratio"]) / mad_corr

    # Snippet divergence z-scores
    z_divergence_map = {}
    if snippet_results and "skipped" not in snippet_results:
        sim_values = [
            snippet_results[url]["snippet_body_similarity"]
            for url in snippet_results
            if isinstance(snippet_results[url], dict)
            and snippet_results[url].get("snippet_body_similarity") is not None
        ]
        if sim_values:
            median_sim = float(np.median(sim_values))
            mad_sim = float(np.median(np.abs(np.array(sim_values) - median_sim)))
            mad_sim = max(mad_sim, 1e-6)
            for url in snippet_results:
                sr = snippet_results[url]
                if isinstance(sr, dict) and sr.get("snippet_body_similarity") is not None:
                    # Inverted: low similarity is anomalous
                    z_divergence_map[url] = (median_sim - sr["snippet_body_similarity"]) / mad_sim

    # Build unified anomaly_scores — raw scores only, no flags
    anomaly_scores = []
    for url in graph.sources:
        signal_scores = {}

        z_focus = z_focus_map.get(url, 0.0)
        z_isolation = z_isolation_map.get(url, 0.0)
        z_divergence = z_divergence_map.get(url, 0.0)

        if url in focus_results:
            signal_scores["max_focus"] = focus_results[url]["max_focus"]
        if url in corr_results:
            signal_scores["corroboration_ratio"] = corr_results[url]["corroboration_ratio"]
        if url in snippet_results and isinstance(snippet_results[url], dict):
            sim = snippet_results[url].get("snippet_body_similarity")
            if sim is not None:
                signal_scores["snippet_body_similarity"] = sim

        signal_scores["z_focus"] = z_focus
        signal_scores["z_isolation"] = z_isolation
        signal_scores["z_divergence"] = z_divergence
        signal_scores["joint_check1"] = z_divergence + z_focus
        signal_scores["joint_check2"] = z_focus + z_isolation

        coordination_flagged = coord_results.get(url, {}).get("coordination_flagged", False)
        max_pair_similarity = coord_results.get(url, {}).get("max_pair_similarity", 0.0)

        anomaly_scores.append({
            "url": url,
            "signal_scores": signal_scores,
            "coordination_flagged": coordination_flagged,
            "max_pair_similarity": max_pair_similarity,
        })

    aggregation_s = time.perf_counter() - aggregation_start

    # --- STEP 10: WRITE STATE ---
    state["anomaly_scores"] = anomaly_scores
    state["flagged_sources_pre_ranking"] = []  # backward compat; backfilled by analyzer
    state["coverage_entropy_result"] = entropy_result if signal_config.coverage_entropy else {}
    state["signals_run"] = signals_run

    # Backward compatibility
    state["concentration_score"] = max(
        (s["signal_scores"].get("max_focus", 0.0) for s in anomaly_scores), default=0.0
    )
    state["concentration_flag"] = any(
        s["signal_scores"].get("z_focus", 0.0) > 2.0 for s in anomaly_scores
    )

    overhead["auditor"] = {
        "graph_construction_s": graph_construction_s,
        "phase1_focus_s": phase1_focus_s,
        "phase1_entropy_s": phase1_entropy_s,
        "embedding_call_s": embedding_call_s,
        "phase2_snippet_s": phase2_snippet_s,
        "phase2_coordination_s": phase2_coordination_s,
        "phase2_corroboration_s": phase2_corroboration_s,
        "aggregation_s": aggregation_s,
        "total_s": time.perf_counter() - auditor_start,
    }
    state["overhead_trace"] = overhead

    return state