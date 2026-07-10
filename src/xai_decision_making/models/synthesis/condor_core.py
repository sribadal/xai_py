"""
ConDOR model components for conditional ordinal diffusion.

Defines the core neural network modules: ConDORConfig (hyperparameters
and feature toggles), OrdinalRegression (cumulative-link ordinal
classifier), RDM (Rectified Diffusion Model for baseline QoL generation),
and TDM (Temporal Diffusion Model for longitudinal QoL trajectories).
"""
import torch
import torch.nn as nn

class ConDORConfig:

    def __init__(self, static_dim=0, use_static_rdm=False, use_static_tdm=False, use_treatment=False, use_mortality_risk=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_dim = 1
        self.cond_dim = 2
        self.label_dim = 5
        self.hidden_dim = 64
        self.rdm_steps = 50
        self.diffusion_steps = 10
        self.static_dim = static_dim
        self.treat_dim = 6 if use_treatment else 0
        self.mortality_risk_dim = 1 if use_mortality_risk else 0
        self.use_static_rdm = use_static_rdm
        self.use_static_tdm = use_static_tdm
        self.use_treatment = use_treatment
        self.use_mortality_risk = use_mortality_risk

class MLPBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)

class OrdinalRegression(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.x_dim + 1, 1)
        self.thresholds = nn.Parameter(torch.sort(torch.randn(config.label_dim - 1))[0])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, a):
        score = self.linear(torch.cat([x, a], dim=1))
        theta = self.thresholds.unsqueeze(0) - score
        cum_probs = torch.cat([
            torch.zeros(x.shape[0], 1).to(x.device),
            self.sigmoid(theta),
            torch.ones(x.shape[0], 1).to(x.device),
        ], dim=1)
        return cum_probs[:, 1:] - cum_probs[:, :-1] + 1e-8

class RDM(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config.x_dim + 3
        if config.use_static_rdm:
            input_dim += config.static_dim
        if config.use_mortality_risk:
            input_dim += config.mortality_risk_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            MLPBlock(config.hidden_dim),
            MLPBlock(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.x_dim),
        )
        self.use_static = config.use_static_rdm
        self.use_mortality_risk = config.use_mortality_risk

    def forward(self, x, t_step, label, time_val, static=None, mortality_risk=None):
        x_in = torch.cat([x, t_step.unsqueeze(1), label.float().unsqueeze(1), time_val.unsqueeze(1)], dim=1)
        if self.use_static and static is not None:
            x_in = torch.cat([x_in, static], dim=1)
        if self.use_mortality_risk and mortality_risk is not None:
            x_in = torch.cat([x_in, mortality_risk], dim=1)
        return self.net(x_in)

class TDM(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config.x_dim + 3
        if config.use_static_tdm:
            input_dim += config.static_dim
        if config.use_treatment:
            input_dim += config.treat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            MLPBlock(config.hidden_dim),
            MLPBlock(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.x_dim),
        )
        self.use_static = config.use_static_tdm
        self.use_treatment = config.use_treatment

    def forward(self, x, step, time_val, label, static=None, treat=None):
        x_in = torch.cat([x, step.float().unsqueeze(1), time_val.unsqueeze(1), label.float().unsqueeze(1)], dim=1)
        if self.use_static and static is not None:
            x_in = torch.cat([x_in, static], dim=1)
        if self.use_treatment and treat is not None:
            x_in = torch.cat([x_in, treat], dim=1)
        return self.net(x_in)