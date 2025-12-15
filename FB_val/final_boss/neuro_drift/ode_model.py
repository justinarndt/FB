import torch
import torch.nn as nn

class QuantumDriftODE(nn.Module):
    """
    Neural ODE for T1/T2 Drift Trajectory Prediction.
    Validates Claim #63/940,641 (Clause 7): "Continuous-Time Calibration Dynamics"
    """
    def __init__(self, state_dim=10):
        super().__init__()
        # Use GELU to prevent vanishing gradients in deep time steps
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, state_dim)
        )
    
    def forward(self, t, y):
        # The derivative function dy/dt
        return self.net(y)

    def integrate(self, y0, t_span, steps=100):
        """Manual RK4 solver to demonstrate numerical stability control."""
        dt = (t_span[1] - t_span[0]) / steps
        y = y0
        trajectory = [y]
        t = t_span[0]
        
        for _ in range(steps):
            k1 = self.forward(t, y)
            k2 = self.forward(t + 0.5*dt, y + 0.5*dt*k1)
            k3 = self.forward(t + 0.5*dt, y + 0.5*dt*k2)
            k4 = self.forward(t + dt, y + dt*k3)
            
            y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t += dt
            trajectory.append(y)
            
        return torch.stack(trajectory)