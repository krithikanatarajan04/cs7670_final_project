import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from pipeline.provenance import ProvenanceGraph
from pipeline.embeddings import embed_batch


@dataclass
class SignalConfig:
    fact1: bool = False
    fact2: bool = False
    coverage_entropy: bool = False
    grubbs_alpha: float = 0.05
    val_asym_lambda: float = 0.5
    corr_conc_min_shared: int = 2
    embedding_dim: int = 384
    entity_merge_threshold: float = 0.92


# ---------------------------------------------------------------------- #
# Entity canonicalization                                                 #
# ---------------------------------------------------------------------- #

def _canonicalize_entities(entity_counts: dict, sim_threshold: float) -> dict:
    excluded = {"N/A", "null", "unknown", "", "None"}
    names = [n for n in entity_counts if n not in excluded]
    if len(names) < 2:
        return entity_counts

    embeddings = embed_batch(names)
    canonical_counts = {}
    processed = set()

    for i, name in enumerate(names):
        if name in processed:
            continue
        total = entity_counts[name]
        processed.add(name)
        for j in range(i + 1, len(names)):
            if names[j] in processed:
                continue
            if np.dot(embeddings[i], embeddings[j]) > sim_threshold:
                total += entity_counts[names[j]]
                processed.add(names[j])
        canonical_counts[name] = total
    return canonical_counts


# ---------------------------------------------------------------------- #
# Bipartite graph                                                         #
# ---------------------------------------------------------------------- #

def _build_bipartite(claims: List[dict]) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, int],
    Dict[str, set],
    Dict[str, str],
]:
    c: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    S_e: Dict[str, set] = defaultdict(set)

    for claim in claims:
        url    = claim.get("source_url", "unknown")
        entity = claim.get("subject_entity", "Unknown")
        if not url or not entity:
            continue
        c[url][entity] += 1
        S_e[entity].add(url)

    c   = {url: dict(ents) for url, ents in c.items()}
    S_e = dict(S_e)
    C   = {url: sum(counts.values()) for url, counts in c.items()}
    e_star = {url: max(ents, key=ents.get) for url, ents in c.items() if ents}

    return c, C, S_e, e_star


# ---------------------------------------------------------------------- #
# Fact 1 signals                                                          #
# ---------------------------------------------------------------------- #

def _compute_focus(c, C, e_star) -> Dict[str, float]:
    result = {}
    for url in c:
        total = C.get(url, 0)
        dom   = e_star.get(url)
        result[url] = (c[url].get(dom, 0) / total) if total and dom else 0.0
    return result


def _compute_isolation(e_star, S_e) -> Dict[str, float]:
    all_urls = list(e_star.keys())
    n = len(all_urls)
    if n <= 1:
        return {url: 1.0 for url in all_urls}

    mentions = {entity: len(sources) for entity, sources in S_e.items()}
    support  = {url: mentions.get(e_star[url], 1) for url in all_urls}
    sorted_urls = sorted(all_urls, key=lambda u: support[u])
    ranks = {url: i + 1 for i, url in enumerate(sorted_urls)}

    return {url: 1.0 - (ranks[url] - 1) / (n - 1) for url in all_urls}


def _compute_asymmetry(c, C, n_entities) -> Dict[str, float]:
    if n_entities <= 1:
        return {url: 0.0 for url in c}

    log_E  = math.log(n_entities)
    result = {}
    for url, entities in c.items():
        total = C.get(url, 0)
        if total == 0:
            result[url] = 0.0
            continue
        entropy = -sum((cnt / total) * math.log(cnt / total)
                       for cnt in entities.values() if cnt > 0)
        result[url] = 1.0 - (entropy / log_E)
    return result


# ---------------------------------------------------------------------- #
# Fact 2 signals                                                          #
# ---------------------------------------------------------------------- #

def _build_source_entity_embeddings(claims, e_star) -> Dict[str, Dict[str, np.ndarray]]:
    texts_to_embed: List[str] = []
    text_index: Dict[str, int] = {}
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for claim in claims:
        url    = claim.get("source_url", "")
        entity = claim.get("subject_entity", "")
        text   = claim.get("text", "").strip()
        if not url or not entity or not text:
            continue
        grouped[(url, entity)].append(text)
        if text not in text_index:
            text_index[text] = len(texts_to_embed)
            texts_to_embed.append(text)

    if not texts_to_embed:
        return {}

    all_embeddings = embed_batch(texts_to_embed)
    mu: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    for (url, entity), texts in grouped.items():
        vecs = [all_embeddings[text_index[t]] for t in texts if t in text_index]
        if not vecs:
            continue
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        mu[url][entity] = mean_vec / norm if norm > 0 else mean_vec

    return dict(mu)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _compute_clustering(mu, e_star, S_e) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for url in e_star:
        dom = e_star[url]
        if dom not in mu.get(url, {}):
            result[url] = None
            continue
        partners = [u for u in S_e.get(dom, set())
                    if u != url and dom in mu.get(u, {})]
        if not partners:
            result[url] = None
            continue
        result[url] = float(np.mean([_cosine(mu[url][dom], mu[p][dom]) for p in partners]))
    return result


def _compute_corr_conc(mu, e_star, min_shared) -> Dict[str, Optional[float]]:
    eps = 1e-9
    result: Dict[str, Optional[float]] = {}

    for url in e_star:
        dom       = e_star[url]
        entities_s = set(mu.get(url, {}).keys())
        if dom not in entities_s:
            result[url] = None
            continue

        rhos = []
        for other_url, other_mu in mu.items():
            if other_url == url or dom not in other_mu:
                continue
            shared = entities_s & set(other_mu.keys())
            if len(shared) < min_shared:
                continue
            sim_star = _cosine(mu[url][dom], other_mu[dom])
            shared_sims = [_cosine(mu[url][e], other_mu[e])
                           for e in shared if e in mu[url] and e in other_mu]
            if not shared_sims:
                continue
            rhos.append(sim_star / (sum(shared_sims) / len(shared_sims) + eps))

        result[url] = float(np.mean(rhos)) if rhos else None
    return result


def _compute_val_asym(claims, e_star) -> Dict[str, Optional[float]]:
    POS = {"excellent","outstanding","perfect","exceptional","best","great",
           "amazing","fantastic","wonderful","superb","top","premier","luxurious",
           "stunning","gorgeous","beautiful","brilliant","highly","recommend",
           "loved","favorite","spotless","spacious","incredible","impressive"}
    NEG = {"bad","terrible","awful","horrible","worst","poor","disappointing",
           "disappoints","dirty","noisy","rude","overpriced","mediocre",
           "unacceptable","cramped","uncomfortable","slow","broken","disgusting"}

    def _sentiment(text: str) -> float:
        words = set(text.lower().split())
        pos = len(words & POS)
        neg = len(words & NEG)
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    sums: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for claim in claims:
        url    = claim.get("source_url", "")
        entity = claim.get("subject_entity", "")
        text   = claim.get("text", "")
        if url and entity and text:
            sums[url][entity].append(_sentiment(text))

    result: Dict[str, Optional[float]] = {}
    for url, dom in e_star.items():
        ent_sentiments = sums.get(url, {})
        if not ent_sentiments or dom not in ent_sentiments:
            result[url] = None
            continue
        v_star = float(np.mean(ent_sentiments[dom]))
        others = [float(np.mean(vals)) for ent, vals in ent_sentiments.items()
                  if ent != dom and vals]
        result[url] = (v_star - float(np.mean(others))) if others else None
    return result


# ---------------------------------------------------------------------- #
# Coverage entropy                                                        #
# ---------------------------------------------------------------------- #

def _compute_coverage_entropy(graph: ProvenanceGraph) -> dict:
    from math import log2

    snapshots = graph.get_coverage_trajectory()
    if len(snapshots) < 2:
        return {"skipped": "insufficient_rounds"}

    round_entropies = []
    for snap in snapshots:
        counts = list(snap.get("coverage", {}).values())
        total  = sum(counts)
        if total == 0:
            round_entropies.append(0.0)
        else:
            entropy = -sum((c / total) * log2(c / total) for c in counts if c > 0)
            round_entropies.append(entropy)

    deltas = [round_entropies[i + 1] - round_entropies[i]
              for i in range(len(round_entropies) - 1)]

    flagged = False
    if len(deltas) >= 2:
        mean = np.mean(deltas)
        std  = np.std(deltas)
        flagged = any(d < mean - std for d in deltas)

    suspicious = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1].get("coverage", {})
        curr = snapshots[i].get("coverage", {})
        for dim, count in curr.items():
            if prev.get(dim, 0) == 0 and count >= 2:
                suspicious.append(dim)

    return {
        "round_entropies":        round_entropies,
        "entropy_deltas":         deltas,
        "fast_convergence_flagged": flagged,
        "suspicious_dimensions":  suspicious,
    }


# ---------------------------------------------------------------------- #
# Score assembly                                                          #
# ---------------------------------------------------------------------- #

def _build_anomaly_scores(
    all_urls, focus, isolation, asymmetry,
    clustering, corr_conc, val_asym,
    val_asym_lambda, embedding_dim, corr_conc_min_shared,
    signals_run,
) -> List[dict]:
    sqrt2_over_d    = math.sqrt(2.0 / embedding_dim)
    corr_conc_tau   = 1.0 + 2.0 * sqrt2_over_d / math.sqrt(max(corr_conc_min_shared, 1))

    scores = []
    for url in all_urls:
        ss      = {}
        defined = []

        f   = focus.get(url)
        iso = isolation.get(url)
        asy = asymmetry.get(url)

        if f   is not None: ss["focus"]     = f;   defined.append("focus")
        if iso is not None: ss["isolation"] = iso; defined.append("isolation")
        if asy is not None: ss["asymmetry"] = asy; defined.append("asymmetry")

        fact1_vals = [v for v in [f, iso, asy] if v is not None]
        ss["score1"] = sum(fact1_vals) / len(fact1_vals) if fact1_vals else 0.0

        clust = clustering.get(url)
        cc    = corr_conc.get(url)
        va    = val_asym.get(url)

        cc_norm = min(max(cc - 1.0, 0.0), 1.0) if cc is not None else None
        va_norm = max(va, 0.0) / 2.0              if va is not None else None

        if clust   is not None: ss["clustering"]     = clust;   defined.append("clustering")
        if cc      is not None: ss["corr_conc"]      = cc;      defined.append("corr_conc")
        if cc_norm is not None: ss["corr_conc_norm"] = cc_norm
        if va      is not None: ss["val_asym"]       = va;      defined.append("val_asym")
        if va_norm is not None: ss["val_asym_norm"]  = va_norm

        coord_vals = [v for v in [clust, cc_norm] if v is not None]
        coord      = max(coord_vals) if coord_vals else None

        if coord is not None:
            ss["score2"] = ((coord + val_asym_lambda * va_norm) / (1.0 + val_asym_lambda)
                            if va_norm is not None else coord)
        elif va_norm is not None:
            ss["score2"] = (val_asym_lambda * va_norm) / (1.0 + val_asym_lambda)
        else:
            ss["score2"] = 0.0

        ss["corr_conc_null_tau"] = corr_conc_tau

        scores.append({
            "url":             url,
            "signal_scores":   ss,
            "signals_defined": defined,
        })

    return scores


# ---------------------------------------------------------------------- #
# Auditor node                                                            #
# ---------------------------------------------------------------------- #

def auditor_node(
    state: dict,
    signal_config: SignalConfig = None,
    signals_active: bool = False,
) -> dict:
    auditor_start = time.perf_counter()
    overhead      = state.get("overhead_trace", {})

    claims             = state.get("claims", [])
    provenance_index   = state.get("provenance_index", {})
    coverage_snapshots = state.get("coverage_snapshots", [])
    provenance_enabled = state.get("provenance_enabled", False)

    if signal_config is None:
        signal_config = SignalConfig()

    # --- Build provenance graph (always, for coverage entropy) ---
    graph_start = time.perf_counter()
    graph       = ProvenanceGraph()
    seen_claims: set = set()
    deduped_claims: List[dict] = []

    for claim in claims:
        url    = claim.get("source_url", "unknown")
        entity = claim.get("subject_entity", "Unknown")
        dim    = claim.get("dimension", "general")
        text   = claim.get("text", "")
        key    = (url, entity, dim, text)

        if key in seen_claims:
            continue
        seen_claims.add(key)
        deduped_claims.append(claim)

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
        graph.add_claim(text=text, source_url=url, subject_entity=entity,
                        dimension=dim, lineage_query=claim.get("lineage_query", ""))

    graph.add_coverage_snapshots(coverage_snapshots)
    graph_construction_s = time.perf_counter() - graph_start
    state["provenance_graph"] = graph.to_dict()

    # --- Early exit when no signals configured ---
    if not signals_active or not provenance_enabled:
        state.update({
            "signals_run":                [],
            "flagged_sources_pre_ranking": [],
            "anomaly_scores":             [],
            "coverage_entropy_result":    {},
            "concentration_flag":         False,
            "concentration_score":        0.0,
        })
        overhead["auditor"] = {
            "graph_construction_s": graph_construction_s,
            "total_s": time.perf_counter() - auditor_start,
        }
        state["overhead_trace"] = overhead
        return state

    # --- Build bipartite graph ---
    c, C, S_e, e_star = _build_bipartite(deduped_claims)
    all_urls   = list(c.keys())
    n_entities = len(S_e)
    signals_run: List[str] = []

    # --- Fact 1 ---
    focus_vals = isolation_vals = asymmetry_vals = {}
    phase1_s = 0.0

    if signal_config.fact1 and all_urls:
        t = time.perf_counter()
        focus_vals     = _compute_focus(c, C, e_star)
        isolation_vals = _compute_isolation(e_star, S_e)
        asymmetry_vals = _compute_asymmetry(c, C, n_entities)
        phase1_s       = time.perf_counter() - t
        signals_run.extend(["focus", "isolation", "asymmetry"])

    # --- Coverage entropy ---
    entropy_result  = {}
    phase1_entropy_s = 0.0

    if signal_config.coverage_entropy:
        t = time.perf_counter()
        entropy_result   = _compute_coverage_entropy(graph)
        phase1_entropy_s = time.perf_counter() - t
        signals_run.append("coverage_entropy")

    # --- Fact 2 ---
    clustering_vals = corr_conc_vals = val_asym_vals = {}
    embedding_call_s = phase2_s = 0.0

    if signal_config.fact2 and all_urls:
        t = time.perf_counter()
        val_asym_vals    = _compute_val_asym(deduped_claims, e_star)
        t2 = time.perf_counter()
        mu               = _build_source_entity_embeddings(deduped_claims, e_star)
        embedding_call_s = time.perf_counter() - t2
        t3 = time.perf_counter()
        clustering_vals  = _compute_clustering(mu, e_star, S_e)
        corr_conc_vals   = _compute_corr_conc(mu, e_star, signal_config.corr_conc_min_shared)
        phase2_s         = time.perf_counter() - t3
        signals_run.extend(["clustering", "corr_conc", "val_asym"])

    # --- Assemble scores ---
    aggregation_start = time.perf_counter()
    anomaly_scores = _build_anomaly_scores(
        all_urls=all_urls,
        focus=focus_vals,
        isolation=isolation_vals,
        asymmetry=asymmetry_vals,
        clustering=clustering_vals,
        corr_conc=corr_conc_vals,
        val_asym=val_asym_vals,
        val_asym_lambda=signal_config.val_asym_lambda,
        embedding_dim=signal_config.embedding_dim,
        corr_conc_min_shared=signal_config.corr_conc_min_shared,
        signals_run=signals_run,
    )
    aggregation_s = time.perf_counter() - aggregation_start

    # --- Concentration flag (focus-based Grubbs, for downstream diagnostics) ---
    focus_map = {e["url"]: e["signal_scores"]["focus"]
                 for e in anomaly_scores if "focus" in e["signal_scores"]}

    def _grubbs_flag(scores: dict, alpha: float) -> bool:
        vals = list(scores.values())
        n    = len(vals)
        if n < 3:
            return False
        mean = np.mean(vals)
        std  = np.std(vals, ddof=1)
        if std == 0:
            return False
        from scipy import stats as _stats
        top_url = max(scores, key=scores.get)
        g_stat  = (scores[top_url] - mean) / std
        t_crit  = _stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        g_crit  = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        return g_stat > g_crit

    state["concentration_flag"]  = _grubbs_flag(focus_map, signal_config.grubbs_alpha) if focus_map else False
    state["concentration_score"] = max((e["signal_scores"].get("focus", 0.0) for e in anomaly_scores), default=0.0)

    state.update({
        "anomaly_scores":              anomaly_scores,
        "flagged_sources_pre_ranking": [],
        "coverage_entropy_result":     entropy_result,
        "signals_run":                 signals_run,
    })

    overhead["auditor"] = {
        "graph_construction_s": graph_construction_s,
        "phase1_s":             phase1_s,
        "phase1_entropy_s":     phase1_entropy_s,
        "embedding_call_s":     embedding_call_s,
        "phase2_s":             phase2_s,
        "aggregation_s":        aggregation_s,
        "total_s":              time.perf_counter() - auditor_start,
        "n_sources":            len(all_urls),
        "n_entities":           n_entities,
        "signals_run":          signals_run,
    }
    state["overhead_trace"] = overhead
    return state