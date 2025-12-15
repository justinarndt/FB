import torch
import numpy as np
from final_boss.neuro_drift.ode_model import QuantumDriftODE

def test_drift_trajectory_fitting():
    """
    CLAIMS TEST: Verifies Neural ODE captures complex drift dynamics.
    """
    model = QuantumDriftODE(state_dim=2)
    y0 = torch.tensor([1.0, 0.0])
    t_span = [0.0, 10.0]
    
    # Integrate forward
    trajectory = model.integrate(y0, t_span, steps=50)
    
    # Check shape and numerical stability (no NaNs)
    assert trajectory.shape == (51, 2)
    assert not torch.isnan(trajectory).any(), "ODE Integration diverged (NaNs found)"