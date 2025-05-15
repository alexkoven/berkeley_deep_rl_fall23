import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Takes a single observation (as a numpy array) and returns a single action (as a numpy array).
        
        For discrete actions: Returns an integer action index
        For continuous actions: Returns a vector of action values
        
        Args:
            obs: numpy array of shape (ob_dim,)
        Returns:
            action: numpy array of shape (ac_dim,)
        """
        # Convert to tensor and add batch dimension
        obs = ptu.from_numpy(obs)[None]  # shape: (1, ob_dim)
        
        # Get action distribution
        dist = self.forward(obs)
        
        # Sample action
        action = dist.sample()  # shape: (1,) for discrete, (1, ac_dim) for continuous
        
        # Remove batch dimension and convert to numpy
        action = ptu.to_numpy(action.squeeze(0))
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        Defines the forward pass of the network.
        
        For discrete actions:
            Returns a Categorical distribution over action indices
            logits = f(obs) where f is the neural network
            P(a|s) = softmax(logits)
            
        For continuous actions:
            Returns a Normal distribution over action values
            μ = f(obs) where f is the neural network
            σ = exp(logstd) where logstd is a learned parameter
            P(a|s) = N(a|μ,σ²)
        
        Args:
            obs: tensor of shape (batch_size, ob_dim)
        Returns:
            dist: torch.distributions.Distribution object
        """
        if self.discrete:
            # Get logits from network
            # logits shape: (batch_size, ac_dim)
            # Each row contains raw logits for each possible action
            logits = self.logits_net(obs)
            
            # Create categorical distribution
            # Internally, Categorical applies softmax to convert logits to probabilities:
            # P(a|s) = exp(logits_a) / sum_a' exp(logits_a')
            # This is done in a numerically stable way using the log-sum-exp trick:
            # log P(a|s) = logits_a - log sum_a' exp(logits_a')
            dist = distributions.Categorical(logits=logits)
        else:
            # Get mean from network
            mean = self.mean_net(obs)  # shape: (batch_size, ac_dim)
            
            # Get standard deviation from learned parameter
            # σ = exp(logstd) to ensure positivity
            std = torch.exp(self.logstd)  # shape: (ac_dim,)
            
            # Create normal distribution
            # P(a|s) = N(a|μ,σ²)
            dist = distributions.Normal(mean, std)
            
            # Make distribution independent across action dimensions
            # log P(a|s) = Σᵢ log N(aᵢ|μᵢ,σᵢ²)
            dist = distributions.Independent(dist, 1)
        
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """
        Implements the policy gradient actor update.
        
        For the first part of the homework (3 Policy Gradient -> vanilla policy gradient without baselines):
        - The advantages are simply the returns (rewards)
        - For Case 1: advantages = sum_{t'=0}^T gamma^t' r_t'
        - For Case 2: advantages = sum_{t'=t}^T gamma^(t'-t) r_t'
        
        The policy gradient objective is:
        J(θ) = E[log π_θ(a|s) * A(s,a)]
        where A(s,a) is either the full trajectory return or reward-to-go
        
        We maximize this by gradient ascent:
        θ ← θ + α * ∇_θ J(θ)
        
        Args:
            obs: numpy array of shape (batch_size, ob_dim)
            actions: numpy array of shape (batch_size, ac_dim) for continuous actions
                    or (batch_size,) for discrete actions
            advantages: numpy array of shape (batch_size,) 
        Returns:
            dict containing the actor loss
        """
        # Convert inputs to tensors
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        
        # Get action distribution
        dist = self.forward(obs)
        
        # Compute log probabilities of actions
        # log π_θ(a|s)
        log_probs = dist.log_prob(actions)  # shape: (batch_size,)
        
        # Compute policy gradient loss
        # J(θ) = E[log π_θ(a|s) * A(s,a)]
        # where A(s,a) is either the full trajectory return or reward-to-go
        loss = -(log_probs * advantages).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
