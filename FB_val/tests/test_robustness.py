import torch
from final_boss.hyper_decode.decoder import HDCSurfaceDecoder

def test_correlated_noise_resilience():
    """
    ROBUSTNESS: Tests decoder against non-Markovian 'burst' errors.
    Standard MWPM often fails here; HDC should survive.
    """
    dim = 5000
    decoder = HDCSurfaceDecoder(num_stabilizers=100, dim=dim)
    
    # Create a random target vector
    target_idx = 0
    clean_hv = torch.randn(dim).sign() # Binary HV
    decoder.associative_memory[target_idx] = clean_hv
    
    # Apply Massive Correlated Burst (30% of bits flipped in a chunk)
    noisy_hv = decoder.stress_test_correlated(clean_hv, burst_prob=1.0)
    
    # Decode
    pred_idx = decoder.decode(noisy_hv)
    
    # Assert
    assert pred_idx == target_idx, "HDC Failed on Correlated Noise Burst!"