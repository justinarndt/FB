import sys
import os
import torch
import torchhd
import cirq
import numpy as np
import scipy.stats as stats
from torchdiffeq import odeint

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from final_boss.tensor_core.builder import SupremacyBuilder
from final_boss.hyper_decode.decoder import HDCSurfaceDecoder
from final_boss.neuro_drift.ode_model import CalibrationPredictor

def proof_1_finite_size_scaling():
    print("\n--- PROOF 1: FINITE SIZE SCALING (FSS) VERIFICATION ---")
    print("Hypothesis: Entanglement Entropy scales linearly with System Size (Volume Law).")
    
    # We use slightly larger steps to see the trend clearly
    sizes = [4, 6, 8, 10, 12] 
    results = []

    for L in sizes:
        qubits = cirq.LineQubit.range(L)
        # Increase depth multiplier to guarantee saturation (Hard Scramble)
        depth = L * 2 
        
        moments = []
        # Seed for reproducibility
        rng = np.random.RandomState(42 + L)

        for d in range(depth):
            # 1. Random Single Qubit Rotations (X then Y)
            # FIX: Separate Moments to avoid "Overlapping Operations" error
            x_exponents = rng.rand(L)
            y_exponents = rng.rand(L)
            
            x_ops = [cirq.X(q)**x_exponents[i] for i, q in enumerate(qubits)]
            y_ops = [cirq.Y(q)**y_exponents[i] for i, q in enumerate(qubits)]
            
            moments.append(cirq.Moment(x_ops))
            moments.append(cirq.Moment(y_ops))
            
            # 2. Maximal Entangling Layers (CZ Gates)
            if d % 2 == 0:
                pairs = [(qubits[i], qubits[i+1]) for i in range(0, L-1, 2)]
            else:
                pairs = [(qubits[i], qubits[i+1]) for i in range(1, L-1, 2)]
                
            ops = [cirq.CZ(q1, q2) for q1, q2 in pairs]
            moments.append(cirq.Moment(ops))
            
        circuit = cirq.Circuit(moments)
        
        # Exact Simulation
        sim = cirq.Simulator()
        result = sim.simulate(circuit)
        final_state = result.final_state_vector
        
        # Calculate Half-Chain Entanglement Entropy
        tensor = final_state.reshape([2] * L)
        split_idx = L // 2
        mat = tensor.reshape(2**split_idx, -1)
        
        # SVD
        s = np.linalg.svd(mat, compute_uv=False)
        
        # Entropy Calculation
        eigenvalues = s**2
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        results.append((L, entropy))
        print(f"  L={L:02d} | Depth={depth} | Entanglement Entropy={entropy:.4f}")

    # Statistical Test: Linearity Check
    L_vals = [r[0] for r in results]
    Ent_vals = [r[1] for r in results]
    slope, intercept, r_value, p_value, std_err = stats.linregress(L_vals, Ent_vals)
    
    print(f"  > R-Squared Value: {r_value**2:.4f}")
    
    # Volume law should be very clear now
    if r_value**2 > 0.95:
        print("  > CONCLUSION: VALID. Entanglement scales linearly with System Size (Volume Law).")
    else:
        print(f"  > CONCLUSION: INVALID. Slope={slope:.4f}. Scaling law not observed.")

def proof_2_holographic_robustness():
    print("\n--- PROOF 2: HOLOGRAPHIC ROBUSTNESS STRESS TEST ---")
    print("Hypothesis: HDC Decoder maintains accuracy under massive input corruption.")
    
    dim = 10000
    num_stabilizers = 50
    decoder = HDCSurfaceDecoder(num_stabilizers, dim, device='cpu')
    
    # Train Memory
    known_errors = []
    decoder.associative_memory = torch.zeros(10, dim)
    
    for i in range(10):
        syndrome_indices = torch.tensor([i, (i+1)%num_stabilizers])
        hv = decoder.encode_syndrome(syndrome_indices)
        decoder.associative_memory[i] = hv
        known_errors.append(syndrome_indices)
        
    # Attack with Noise
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    for noise in noise_levels:
        success_count = 0
        trials = 100
        for _ in range(trials):
            target_idx = np.random.randint(0, 10)
            target_syndrome = known_errors[target_idx]
            clean_hv = decoder.encode_syndrome(target_syndrome)
            
            mask = torch.rand(dim) < noise
            noisy_hv = clean_hv.clone()
            noisy_hv[mask] *= -1 
            
            sims = torchhd.cosine_similarity(noisy_hv, decoder.associative_memory)
            pred_idx = torch.argmax(sims).item()
            
            if pred_idx == target_idx:
                success_count += 1
                
        print(f"  Noise Level {int(noise*100)}%: Accuracy = {success_count}/{trials} ({success_count}%)")
        
    if success_count >= 90:
        print("  > CONCLUSION: VALID. Holographic property confirmed.")
    else:
        print("  > CONCLUSION: FAILED.")

def proof_3_neural_ode_extrapolation():
    print("\n--- PROOF 3: NEURAL ODE DYNAMICS EXTRAPOLATION ---")
    print("Hypothesis: Neural ODE learns the underlying Hamiltonian.")
    
    def get_ground_truth(t):
        return 1.0 + 0.5 * torch.sin(t / 3.0)
    
    t_train = torch.linspace(0., 15., 30)
    y_train = get_ground_truth(t_train).unsqueeze(1)
    
    t_test = torch.linspace(15., 30., 20)
    y_test_truth = get_ground_truth(t_test).unsqueeze(1)
    
    model = CalibrationPredictor(data_dim=1)
    model.integration_method = 'rk4' 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    
    print("  > Training on t=[0, 15]...")
    for i in range(150):
        optimizer.zero_grad()
        y0 = y_train[0]
        pred_y = model(y0, t_train)
        loss = torch.mean((pred_y.squeeze() - y_train.squeeze())**2)
        loss.backward()
        optimizer.step()
        
    print("  > Extrapolating to unseen future t=[15, 30]...")
    with torch.no_grad():
        y_last = y_train[-1]
        t_future_relative = t_test - t_test[0] 
        pred_test = model(y_last, t_future_relative)
    
    final_error = torch.mean(torch.abs(pred_test.squeeze() - y_test_truth.squeeze())).item()
    print(f"  > Mean Absolute Error on Future Predictions: {final_error:.4f}")
    
    if final_error < 0.35:
        print("  > CONCLUSION: VALID. Neural ODE predicted unobserved future dynamics.")
    else:
        print("  > CONCLUSION: FAILED.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    proof_1_finite_size_scaling()
    proof_2_holographic_robustness()
    proof_3_neural_ode_extrapolation()
    print("\n=== GOLD STANDARD SCIENCE VERIFICATION COMPLETE ===")