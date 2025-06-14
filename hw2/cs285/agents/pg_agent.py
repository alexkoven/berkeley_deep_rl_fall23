from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """
        print("\nDEBUG: Input shapes:")
        print(f"obs: {[o.shape for o in obs]}")
        print(f"actions: {[a.shape for a in actions]}")
        print(f"rewards: {[r.shape for r in rewards]}")
        print(f"terminals: {[t.shape for t in terminals]}")

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
        print("\nDEBUG: After _calculate_q_vals:")
        print(f"q_values type: {type(q_values)}")
        print(f"q_values: {q_values}")
        if isinstance(q_values, list):
            print(f"q_values elements: {[type(q) for q in q_values]}")
            print(f"q_values shapes: {[q.shape if hasattr(q, 'shape') else len(q) for q in q_values]}")

        # flatten the lists of arrays into single arrays
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # Perform multiple gradient steps on the critic
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # If no baseline, advantages are just the Q-values (reward-to-go)
            advantages = q_values
        else:
            # Get value predictions from critic
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # If using a baseline but not GAE, advantages are Q-values minus value predictions
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    pass

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            # Calculate mean and standard deviation
            mean_advantage = np.mean(advantages)
            std_advantage = np.std(advantages)
            # Avoid division by zero by adding a small epsilon
            std_advantage = std_advantage + 1e-8
            # Normalize advantages
            advantages = (advantages - mean_advantage) / std_advantage

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> np.ndarray:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a numpy array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        rewards = np.array(rewards)
        T = len(rewards)
        q_values = np.zeros(T, dtype=np.float32)
        for t in range(T):
            # For each timestep t, calculate the discounted sum of all rewards
            discount_factors = np.power(self.gamma, np.arange(T))
            q_values[t] = np.sum(rewards * discount_factors)
        return q_values

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> np.ndarray:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a numpy array where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rewards = np.array(rewards)
        T = len(rewards)
        rtg = np.zeros(T, dtype=np.float32)
        for t in range(T):
            future_rewards = rewards[t:]
            discount_factors = np.power(self.gamma, np.arange(len(future_rewards)))
            rtg[t] = np.sum(future_rewards * discount_factors)
        return rtg
