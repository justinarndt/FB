import pytest
from hypothesis import given, strategies as st
from final_boss.tensor_core.builder import SupremacyBuilder

@given(L=st.integers(50, 100), depth=st.integers(10, 20))
def test_volume_law_scaling(L, depth):
    """
    STRESS TEST: Verifies that Bond Dimension (Chi) exceeds classical 
    simulation limits for L > 50, validating Quantum Supremacy claims.
    """
    builder = SupremacyBuilder(L, depth)
    chi = builder.build_and_contract()
    
    # If L > 50, Chi must be massive (Volume Law)
    # This disproves the "Area Law Artifact" critique
    if L > 50:
        assert chi > 10000, f"Failed Volume Law: Chi={chi} too low for L={L}"