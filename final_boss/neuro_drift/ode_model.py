import torch
import torch.nn as nn
from torchdiffeq import odeint

class DriftFunc(nn.Module):
    """Approximates the time-derivative of calibration parameters[cite: 183]."""
    def __init__(self, data_dim):
        super(DriftFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, data_dim)
        )

    def forward(self, t, y):
        return self.net(y)

class CalibrationPredictor(nn.Module):
    def __init__(self, data_dim):
        super(CalibrationPredictor, self).__init__()
        self.func = DriftFunc(data_dim)
        self.integration_method = 'dopri5'

    def forward(self, y0, t_points):
        """Predicts state at future times t_points given initial state y0[cite: 185]."""
        return odeint(self.func, y0, t_points, method=self.integration_method)