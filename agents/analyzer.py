import time
import numpy as np
from pipeline.cfg import validate_anomaly_scores_present


def analyzer_node(
    state: dict,
    analyzer_exclusion: bool = False,
) -> dict:
    analyzer_start   = time.perf_counter()
    overhead         = state.get("overhead_trace", {})
    anomaly_scores   = state.get("anomaly_scores", [])
    claims           = state.get("claims", [])
    exclusion_active = state.get("analyzer_exclusion", analyzer_exclusion)

    score_map     = {entry["url"]: entry for entry in anomaly_scores}
    excluded_urls: set = set()
    exclusion_reasons: dict = {}

    if exclusion_active and score_map:
        # Build RMS tension magnitude per source.
        # Axes use additive mean (not multiplication) so a zero on one axis
        # doesn't collapse the whole signal — e.g., low isolation doesn't
        # cancel a high focus score.
        source_vectors = {}
        for url, entry in score_map.items():
            ss = entry.get("signal_scores", {})

            # Axis 1 — Plantedness: mean(focus, isolation)
            f   = ss.get("focus")
            iso = ss.get("isolation")
            planted_components = [v for v in [f, iso] if v is not None]
            t1 = sum(planted_components) / len(planted_components) if planted_components else 0.0

            # Axis 2 — Coordination: mean(clustering, corr_conc_norm)
            clust    = ss.get("clustering")
            cc_norm  = ss.get("corr_conc_norm")
            coord_components = [v for v in [clust, cc_norm] if v is not None]
            t2 = (sum(coord_components) / len(coord_components)) if coord_components else None

            # Axis 3 — Promotional bias: mean(asymmetry, val_asym_norm)
            asy     = ss.get("asymmetry")
            va_norm = ss.get("val_asym_norm")
            bias_components = [v for v in [asy, va_norm] if v is not None]
            t3 = (sum(bias_components) / len(bias_components)) if bias_components else None

            active = [t1] + [v for v in [t2, t3] if v is not None]
            magnitude = np.sqrt(sum(v ** 2 for v in active) / len(active))

            source_vectors[url] = {
                "magnitude":  magnitude,
                "axes_used":  len(active),
                "details":    {"planted": t1, "coord": t2, "bias": t3},
            }

        # Iterative MAD peeling — removes one outlier per pass until
        # z < 3.5 or fewer than 3 sources remain.
        while True:
            remaining = [u for u in score_map if u not in excluded_urls]
            if len(remaining) < 3:
                break

            mags    = np.array([source_vectors[u]["magnitude"] for u in remaining])
            median  = np.median(mags)
            mad     = np.median(np.abs(mags - median))

            if mad == 0:
                break

            mod_z   = 0.6745 * np.abs(mags - median) / mad
            top_idx = int(np.argmax(mod_z))

            if mod_z[top_idx] <= 3.5:
                break

            outlier = remaining[top_idx]
            excluded_urls.add(outlier)
            exclusion_reasons[outlier] = {
                "check":            "Robust-RMS-Tension",
                "z_score":          round(float(mod_z[top_idx]), 2),
                "axes_counted":     source_vectors[outlier]["axes_used"],
                "vector_components": source_vectors[outlier]["details"],
            }

    # Single-pass exclusion: filter claims, then derive surviving entities
    # from that filtered set (not from raw claims).
    filtered_claims = [c for c in claims if c.get("source_url") not in excluded_urls]
    surviving_entities = {c["subject_entity"] for c in filtered_claims}
    clean_claims = [c for c in filtered_claims if c.get("subject_entity") in surviving_entities]

    state["excluded_sources"]  = list(excluded_urls)
    state["exclusion_reasons"] = exclusion_reasons
    state["clean_claims"]      = clean_claims
    state["clean_claim_count"] = len(clean_claims)

    # Surface coverage entropy caveat if auditor flagged it
    entropy_result = state.get("coverage_entropy_result", {})
    state["process_caveat"]       = bool(entropy_result.get("fast_convergence_flagged"))
    state["suspicious_dimensions"] = entropy_result.get("suspicious_dimensions", [])

    overhead["analyzer"] = {
        "total_s":         time.perf_counter() - analyzer_start,
        "excluded_count":  len(excluded_urls),
        "clean_claim_count": len(clean_claims),
        "exclusion_active": exclusion_active,
    }
    state["overhead_trace"] = overhead
    return state