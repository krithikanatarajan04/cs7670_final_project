import pytest
from pipeline.provenance import ProvenanceGraph

def test_provenance_graph_logic():
    # 1. Initialize Graph
    pg = ProvenanceGraph()

    # 2. Add two sources 
    # (Note: add_source now generates its own UUIDs and takes URL/Content)
    s1_id = pg.add_source(url="https://journal-a.com", content="Review of Grand Hotel")
    s2_id = pg.add_source(url="https://clinic-b.com", content="Safety report for Grand Hotel")

    # The target being researched
    hotel_target = "Grand Hotel"

    # 3. Add three claims 
    # Two from source 1
    c1_id = pg.add_claim(
        claim_text="The pool is heated.", 
        source_id=s1_id, 
        research_target=hotel_target
    )
    c2_id = pg.add_claim(
        claim_text="The gym is open 24/7.", 
        source_id=s1_id, 
        research_target=hotel_target
    )
    # One from source 2
    c3_id = pg.add_claim(
        claim_text="The elevator is certified safe.", 
        source_id=s2_id, 
        research_target=hotel_target
    )

    # 4. Add one recommendation
    rec_id = pg.add_recommendation(
        research_target=hotel_target,
        rank=1,
        supporting_claim_ids=[c1_id, c2_id, c3_id],
        reasoning="This hotel has great amenities and safety records."
    )

    # --- Assertions ---

    # Assert that get_claims_for_target returns all three
    all_claims = pg.get_claims_for_target(hotel_target)
    assert len(all_claims) == 3
    claim_ids = [c.claim_id for c in all_claims]
    assert c1_id in claim_ids
    assert c2_id in claim_ids
    assert c3_id in claim_ids

    # Assert that get_sources_for_target returns both sources
    all_sources = pg.get_sources_for_target(hotel_target)
    assert len(all_sources) == 2
    source_ids = [s.source_id for s in all_sources]
    assert s1_id in source_ids
    assert s2_id in source_ids

    # Assert that get_claims_for_source_and_target returns only the claims from source 1
    src_1_claims = pg.get_claims_for_source_and_target(s1_id, hotel_target)
    assert len(src_1_claims) == 2
    assert all(c.source_id == s1_id for c in src_1_claims)
    
    # Assert it returns only the claim from source 2
    src_2_claims = pg.get_claims_for_source_and_target(s2_id, hotel_target)
    assert len(src_2_claims) == 1
    assert src_2_claims[0].claim_id == c3_id

if __name__ == "__main__":
    pytest.main([__file__])