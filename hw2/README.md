## Results
To be added.

## Call Stack for Policy Gradient Training

The following is the complete call stack for training a policy gradient agent:

```
run_hw2.py:main()
└── run_training_loop(args)
    ├── Initialize environment (gym.make)
    ├── Initialize PGAgent
    └── For each iteration:
        ├── utils.sample_trajectories(env, agent.actor, args.batch_size, max_ep_len)
        │   └── Returns list of trajectories containing:
        │       ├── observations
        │       ├── actions
        │       ├── rewards
        │       ├── next_observations
        │       └── terminals
        │
        ├── Convert trajectories to dictionary format
        │
        ├── agent.update(obs, actions, rewards, terminals)
        │   ├── _calculate_q_vals(rewards)
        │   │   ├── _discounted_return(rewards)  # if not using reward-to-go
        │   │   └── _discounted_reward_to_go(rewards)  # if using reward-to-go
        │   │
        │   ├── _estimate_advantage(obs, rewards, q_values, terminals)
        │   │   ├── Calculate baseline (if use_baseline=True)
        │   │   └── Normalize advantages (if normalize_advantages=True)
        │   │
        │   └── MLPPolicyPG.update(obs, actions, advantages)
        │       └── Update policy network parameters
        │
        ├── Evaluate (if scalar_log_freq)
        │   └── utils.sample_trajectories(env, agent.actor, args.eval_batch_size, max_ep_len)
        │
        └── Collect video rollouts (if video_log_freq)
            └── utils.sample_n_trajectories(env, agent.actor, MAX_NVIDEO, max_ep_len, render=True)
```

Key command line arguments for basic policy gradient:
```bash
python cs285/scripts/run_hw2.py \
    --env_name CartPole-v0 \
    --exp_name test_pg \
    --n_iter 100 \
    --batch_size 1000 \
    --learning_rate 5e-3 \
    --n_layers 2 \
    --layer_size 64 \
    --discount 1.0
```

Optional arguments that affect training:
- `--use_reward_to_go`: Use reward-to-go instead of full returns
- `--normalize_advantages`: Normalize advantages to mean 0, std 1
- `--use_baseline`: Use value function baseline
- `--gae_lambda`: Use Generalized Advantage Estimation


## 8. Homework Analysis

### 1. Policy-gradient calculations for the 2-action, single-state MDP (no discount)

We write every trajectory in terms of the **number of times the agent keeps pressing `a₁` before it finally terminates with `a₂`**.
Let

$$
n\;:=\;\text{“failures” before the first success}=\#\{\,a_{1}\text{ taken}\,\}\in\{0,1,2,\dots\}
$$


#### Action probabilities and trajectory statistics

| quantity                           | value for a trajectory with $n$ repeats of $a_{1}$                    |
| ---------------------------------- | --------------------------------------------------------------------- |
| probability of that trajectory     | $P_\theta(n)=\theta^{n}(1-\theta)$                                    |
| return (total undiscounted reward) | $R(n)=n$                                                              |
| log-probability                    | $\log P_\theta(n)= n\log\theta+\log(1-\theta)$                        |
| score function                     | $\nabla_\theta\log P_\theta(n)=\dfrac{n}{\theta}-\dfrac{1}{1-\theta}$ |

The random variable $n$ therefore follows a geometric distribution
$\mathcal G(p)$ with “success” probability $p=1-\theta$.


#### 1 (a) Policy-gradient theorem calculation

$$
\nabla_\theta J(\theta)\;=\;\mathbb E_{n\sim \mathcal G(1-\theta)}
\Bigl[\,R(n)\,\nabla_\theta\log P_\theta(n)\Bigr]
\;=\;\mathbb E\bigl[n\bigl(\tfrac{n}{\theta}-\tfrac{1}{1-\theta}\bigr)\bigr]
= \frac{\mathbb E[n^{2}]}{\theta}-\frac{\mathbb E[n]}{1-\theta}.
$$

For a geometric $\mathcal G(1-\theta)$:

$$
\mathbb E[n]=\frac{\theta}{1-\theta},\qquad
\operatorname{Var}[n]=\frac{\theta}{(1-\theta)^{2}},\qquad
\mathbb E[n^{2}]=\operatorname{Var}[n]+\mathbb E[n]^{2}
              =\frac{\theta(1+\theta)}{(1-\theta)^{2}}.
$$

Substituting,

$$
\nabla_\theta J(\theta)
      =\frac{1+\theta}{(1-\theta)^{2}}
       -\frac{\theta}{(1-\theta)^{2}}
      =\boxed{\frac{1}{(1-\theta)^{2}}}.
$$


#### 1 (b) Direct expectation of the return and its derivative

Because each $a_{1}$ yields reward 1 and the episode stops at the first $a_{2}$,

$$
J(\theta)=\mathbb E[R]
       =\sum_{n=0}^{\infty}(1-\theta)\theta^{n}\,n
       =(1-\theta)\sum_{n=0}^{\infty}n\theta^{n}
       =(1-\theta)\frac{\theta}{(1-\theta)^{2}}
       =\frac{\theta}{1-\theta}.
$$

Differentiate w\.r.t. $\theta$:

$$
\nabla_\theta J(\theta)
      =\frac{(1-\theta)-\theta(-1)}{(1-\theta)^{2}}
      =\boxed{\frac{1}{(1-\theta)^{2}}},
$$

in perfect agreement with the policy-gradient-theorem result in part (a).

Hence the likelihood-ratio policy-gradient estimator is **unbiased** and its analytic expectation equals the true gradient $1/(1-\theta)^{2}$.

---

## Difference to HW1 Policy in networks/policies.py
The main differences between `policies.py` (HW2) and `MLP_policy.py` (HW1) are:

1. **Architecture and Inheritance**:
   - HW1's `MLPPolicySL` inherits from both `BasePolicy` and `nn.Module`, and uses `abc.ABCMeta` for abstract base class functionality
   - HW2's `MLPPolicy` is a simpler implementation that only inherits from `nn.Module`
   - HW2 introduces a new subclass `MLPPolicyPG` specifically for policy gradient algorithms

2. **Action Space Handling**:
   - HW1's implementation only handles continuous action spaces (using Normal distribution)
   - HW2's implementation is more flexible, handling both discrete and continuous action spaces:
     - For discrete actions: Uses a `logits_net` to output action probabilities
     - For continuous actions: Uses a `mean_net` and `logstd` similar to HW1

3. **Forward Pass Implementation**:
   - HW1's forward pass is fully implemented, returning both actions and the distribution object
   - HW2's forward pass is currently incomplete (marked with TODOs), but is designed to be more flexible in its return type

4. **Update Method**:
   - HW1's update method is implemented for supervised learning, taking observations and actions as input
   - HW2's update method is abstract in the base class and implemented in `MLPPolicyPG` specifically for policy gradient, taking observations, actions, and advantages as input

5. **Action Selection**:
   - HW1 doesn't have a separate `get_action` method
   - HW2 introduces a new `get_action` method (currently incomplete) that will be used for inference, taking a single observation and returning a single action

The key new aspects in HW2's implementation are:

1. **Policy Gradient Specifics**:
   - The introduction of advantages in the update method, which is crucial for policy gradient algorithms
   - A separate `MLPPolicyPG` class that implements the policy gradient update rule

2. **More Flexible Architecture**:
   - Support for both discrete and continuous action spaces
   - A cleaner separation between the base policy and specific implementations
   - A dedicated inference method (`get_action`) separate from the forward pass

3. **Simplified Base Structure**:
   - Removed the abstract base class complexity
   - More focused implementation specific to policy gradient methods

These changes reflect the transition from supervised learning (behavior cloning) in HW1 to reinforcement learning (policy gradient) in HW2. The new implementation is designed to handle the specific requirements of policy gradient algorithms while maintaining flexibility for different action spaces and use cases.



## Are we using replay buffer?

Let me search through the codebase to understand how data handling is implemented in hw2.


Ran tool
Based on my analysis of the codebase, hw2 does not use a traditional replay buffer or data buffer. Instead, it uses an on-policy approach where:

1. Data Collection:
- The code collects trajectories (episodes) on-the-fly using the current policy
- This is done in `utils.py` through the `sample_trajectory()` and `sample_trajectories()` functions
- Each trajectory contains observations, actions, rewards, next observations, and terminal flags

2. Training Process:
- The data is collected in batches (specified by `batch_size` parameter)
- Each batch contains multiple trajectories that sum up to at least `min_timesteps_per_batch` steps
- The data is used immediately for training and then discarded
- This is evident in `run_hw2.py` where trajectories are sampled and then immediately used to update the policy

3. Key Implementation Details:
- The `PGAgent` class in `pg_agent.py` processes the collected trajectories to:
  - Calculate Q-values (returns)
  - Estimate advantages
  - Update the policy network
  - Optionally update a value baseline network

This is different from off-policy algorithms (like DQN) that use replay buffers to store and reuse past experiences. The policy gradient approach used here is on-policy, meaning it only uses data collected from the current policy to make updates.

The batch size parameter (default 1000) controls how many environment steps are collected before each policy update, but this is not a persistent buffer - it's just the amount of data collected for each training iteration.

## How are the trajectories fed to the PG algorithm and how dense are the rewards?

Let me analyze how the trajectories are processed and used for training, and examine the reward structure.


Ran tool
Let me explain how the trajectories are processed and used for training, and address the reward structure:

1. Trajectory Processing:
- The trajectories are collected in temporal order during sampling (in `sample_trajectory()`)
- When used for training, they are flattened into single arrays (in `PGAgent.update()`):
```python
# flatten the lists of arrays into single arrays
obs = np.concatenate(obs)
actions = np.concatenate(actions)
rewards = np.concatenate(rewards)
terminals = np.concatenate(terminals)
q_values = np.concatenate(q_values)
```
- The temporal order is preserved during flattening, but the policy gradient update itself doesn't depend on the order since it treats each (state, action) pair independently

2. Training Process:
- The policy gradient update uses all datapoints in the batch to compute the loss:
```python
# Compute policy loss as negative mean of (log_probs * advantages)
loss = -(log_probs * advantages).mean()
```
- This means each (state, action) pair contributes equally to the gradient, regardless of its position in the trajectory
- The advantages/returns are computed in temporal order (either full returns or reward-to-go) but the actual gradient update treats each sample independently

3. Reward Structure:
Looking at the environments used in hw2:
- CartPole-v0: Dense rewards (every action gets a reward)
- HalfCheetah: Dense rewards (continuous rewards for movement)
- LunarLander: Semi-dense rewards (rewards for various actions like firing engines, landing)
- InvertedPendulum: Dense rewards (continuous rewards for keeping the pole upright)
- Humanoid: Dense rewards (continuous rewards for movement and staying upright)

None of these environments are truly sparse in their reward structure. They all provide feedback (rewards) for most actions, which is why policy gradient methods work well on them. The rewards are designed to guide the agent toward the desired behavior through continuous feedback rather than sparse, delayed rewards.

The temporal ordering of the trajectories is important for computing the returns/advantages (especially for reward-to-go calculations), but the actual policy update treats each sample independently. This is why the policy gradient method can work with both dense and sparse rewards, though it generally performs better with dense rewards as they provide more immediate feedback for learning.


## Does the cartpole video only show one episode?

Yes, that's correct! In the current implementation, each episode runs independently and the video window resets when a new episode starts. Each episode starts with env.reset() which:
- Resets the cart position to center
- Resets the pole to vertical position
- Clears the previous visualization
- Starts a new episode


## Explanation of Actor Loss

Can you analyze the @q2_pg_cartpole_lb_no_rtg_no_na_CartPole-v0_26-05-2025_11-45-02  folder's in@hw2 data (json) and explain what the different metrics show and what their ideal curve should look like? For example, for a loss should it be monotonically increasing, etc. 

---

**Cursor**

Let me search for and analyze the data folder you mentioned.


Let me look at the contents of the specific folder you mentioned.


Let me read the training logs to analyze the metrics.


Let me analyze the key metrics from the training logs and explain what they mean and their ideal behavior:

1. **Eval_AverageReturn**
- This is the average return (total reward) achieved during evaluation episodes
- Ideal behavior:
  - Should generally increase over time as the policy improves
  - For CartPole-v0, the maximum possible return is 200 (when the pole stays upright for 200 steps)
  - We can see the agent quickly learns to achieve the maximum return (200) around iteration 18
  - Some fluctuations are normal due to the stochastic nature of policy gradient methods
  - The slight decrease at the end (from 200 to around 115) might indicate some instability in learning

2. **Train_AverageReturn**
- Similar to Eval_AverageReturn but measured during training episodes
- Ideal behavior:
  - Should generally increase over time
  - May be more volatile than eval returns due to exploration during training
  - Should eventually converge to a high value (close to 200 for CartPole)

3. **Train_StdReturn**
- Standard deviation of returns during training
- Ideal behavior:
  - Should generally decrease over time as the policy becomes more consistent
  - Some variation is normal due to the stochastic nature of the environment and policy
  - A very low std might indicate the policy is too deterministic and not exploring enough

4. **Train_AverageEpLen**
- Average episode length during training
- Ideal behavior:
  - For CartPole, should increase and eventually reach close to 200 steps
  - Longer episodes indicate the pole stays upright longer
  - Should correlate positively with the average return

5. **Train_Entropy**
- Entropy of the policy distribution
- Ideal behavior:
  - Should generally decrease over time as the policy becomes more deterministic
  - A very low entropy might indicate the policy is too deterministic
  - A very high entropy might indicate the policy is too random
  - Some entropy is good for maintaining exploration

6. **Train_Loss**
- The policy gradient loss
- Ideal behavior:
  - Should generally decrease over time
  - May be more volatile than other metrics
  - The exact value is less important than the trend
  - Fluctuations are normal due to the stochastic nature of policy gradient updates

7. **Time**
- Training time per iteration
- Ideal behavior:
  - Should be relatively constant
  - Any significant increases might indicate performance issues

8. **Train_BaselineLoss** (if using a baseline)
- Loss of the value function baseline
- Ideal behavior:
  - Should generally decrease over time
  - Should be lower than the policy loss
  - Helps reduce variance in policy gradient updates

9. **Train_ExplainedVariance** (if using a baseline)
- How well the baseline predicts the returns
- Ideal behavior:
  - Should generally increase over time
  - Values closer to 1.0 indicate better predictions
  - Helps assess the quality of the baseline

For CartPole-v0 specifically:
- The agent should learn to balance the pole for 200 steps (maximum episode length)
- The learning curve should show rapid improvement in the first few iterations
- The policy should become relatively stable after reaching good performance
- Some fluctuations in performance are normal due to the stochastic nature of policy gradient methods
- The agent should maintain good performance (close to 200 steps) consistently after learning

The data shows that your agent learned quickly (reaching 200 steps by iteration 18) but showed some instability later in training. This is not uncommon in policy gradient methods, and you might want to experiment with:
1. Learning rate scheduling
2. Different batch sizes
3. Advantage normalization
4. Reward-to-go calculations
to achieve more stable performance.

---

**User**

What about the actor_loss? 

---

**Cursor**

Let me analyze the actor loss in the training logs.


Let me try another search pattern since the loss might be logged under a different name.


Let me read more of the training logs to find the loss metrics.


Let me look at the code to understand how the actor loss is calculated and logged.


Based on the code and training logs, let me explain the actor loss in policy gradient:

1. **Actor Loss Calculation**:
From `MLPPolicyPG.update()` in `policies.py`, the actor loss is calculated as:
```python
# Calculate log probabilities of taken actions
if self.discrete:
    log_probs = action_dist.log_prob(actions)
else:
    log_probs = action_dist.log_prob(actions).sum(dim=-1)

# Compute policy loss as negative mean of (log_probs * advantages)
loss = -(log_probs * advantages).mean()
```

2. **What the Actor Loss Represents**:
- The actor loss is the negative expected advantage-weighted log probability of actions
- It represents how well the policy is performing relative to the baseline (or just the returns if no baseline is used)
- A lower loss means the policy is better at selecting actions that lead to higher returns

3. **Ideal Behavior**:
- The actor loss should generally decrease over time as the policy improves
- However, it may not be monotonically decreasing due to:
  - The stochastic nature of policy gradient updates
  - Changes in the advantage estimates
  - Exploration vs exploitation trade-off
  - The fact that we're maximizing expected returns, not minimizing the loss directly

4. **Interpretation**:
- A decreasing trend indicates the policy is learning to select better actions
- Fluctuations are normal and expected
- The absolute value of the loss is less important than its trend
- The loss might increase temporarily when the policy explores new actions

5. **Relationship with Other Metrics**:
- The actor loss should correlate with the average return
- When the average return increases, the actor loss should generally decrease
- The loss might increase when the policy is exploring new behaviors that could lead to better performance

6. **Normalization Effects**:
- If `normalize_advantages` is True, the advantages are normalized to have zero mean and unit variance
- This can make the loss values more stable and easier to interpret
- The loss values will be in a more consistent range across different environments

7. **Baseline Effects**:
- If using a baseline (value function), the advantages will be smaller
- This can lead to smaller loss values but doesn't necessarily mean better performance
- The baseline helps reduce variance in the updates

8. **Warning Signs**:
- If the loss becomes extremely large or small, it might indicate numerical instability
- If the loss doesn't decrease at all, the learning rate might be too small
- If the loss oscillates wildly, the learning rate might be too large

The actor loss is just one metric to monitor during training. It should be considered alongside other metrics like average return, entropy, and standard deviation of returns to get a complete picture of the training progress.

```python
# Calculate log probabilities of taken actions
if self.discrete:
    log_probs = action_dist.log_prob(actions)
else:
    log_probs = action_dist.log_prob(actions).sum(dim=-1)

# Compute policy loss as negative mean of (log_probs * advantages)
loss = -(log_probs * advantages).mean()
```

---

**User**

How come the loss starts low, then goes high by iteration 30 and then continuously decreases back down to 20?

---

**Cursor**

Let me analyze this behavior by looking at the training logs and the policy gradient implementation.


Let me explain why the loss shows this pattern:

1. **Initial Low Loss (Iterations 1-30)**:
- At the start, the policy is random or close to random
- The log probabilities of actions are relatively uniform
- The advantages (returns) are also relatively uniform since the policy hasn't learned to distinguish good from bad actions
- This leads to a low initial loss because: `loss = -(log_probs * advantages).mean()`
- When both log_probs and advantages are small, their product is small

2. **High Loss Peak (Around Iteration 30)**:
- By this point, the policy has started to learn which actions are good
- The advantages start to show clear differences between good and bad actions
- The policy starts to assign higher probabilities to good actions
- This leads to:
  - Higher log probabilities for good actions
  - Higher advantages for good actions
  - Their product becomes larger, leading to a higher loss
- This is actually a good sign - it means the policy is starting to differentiate between good and bad actions

3. **Decreasing Loss (After Iteration 30)**:
- As the policy continues to improve:
  - It becomes more confident in its good actions (higher log probabilities)
  - The advantages become more refined
  - The policy learns to avoid bad actions (lower log probabilities)
  - The product of log probabilities and advantages becomes more balanced
- The loss decreases as the policy becomes more stable and efficient

4. **Why This Pattern is Normal**:
- This pattern is common in policy gradient methods because:
  - The loss is not a direct measure of performance
  - It's a measure of how much the policy is being pushed to change
  - High loss can indicate active learning
  - Low loss can indicate either:
    - A random policy (early training)
    - A well-learned policy (late training)

5. **Correlation with Returns**:
Looking at the `Eval_AverageReturn` values:
- The returns increase significantly around iteration 10-20
- They reach 200 (maximum) by iteration 18
- This shows that the high loss period corresponds to the policy learning to achieve good performance
- The decreasing loss later corresponds to the policy refining its behavior

6. **Why It's Not a Problem**:
- The loss pattern doesn't indicate a problem because:
  - The returns are high and stable
  - The policy is achieving the maximum possible return (200)
  - The loss is just a training signal, not a performance metric
  - The final low loss indicates the policy has converged to a good solution

This pattern is actually a good sign of learning:
1. Start with random policy (low loss)
2. Learn to distinguish good from bad actions (high loss)
3. Refine the policy to be more efficient (decreasing loss)
4. Converge to a stable solution (low loss)

The key is to focus on the returns rather than the loss - the returns show that the policy is performing well, which is what really matters.

---

**User**

Great thanks. Now, why is there are dip in the rewards at the very end of the iterations (around 90). In the video, it seems the cart reaches the borders of the environment. Is that negatively penalized? If so, why does it or does it not learn how to deal with that?

---

**Cursor**

Let me analyze this behavior by looking at the CartPole environment and the training data.

1. **CartPole Environment Boundaries**:
- In CartPole-v0, the cart can move left and right
- The episode terminates if:
  1. The pole falls over (angle > 12 degrees)
  2. The cart moves too far from center (position > 2.4 units)
  3. The episode reaches 200 steps (maximum length)

2. **Looking at the Training Data**:
From the logs, we can see the returns drop from 200 to around 115 at the end:
```
200.0,  # iteration 85
200.0,  # iteration 86
200.0,  # iteration 87
200.0,  # iteration 88
200.0,  # iteration 89
151.6666717529297,  # iteration 90
200.0,  # iteration 91
136.6666717529297,  # iteration 92
132.5,  # iteration 93
118.25,  # iteration 94
107.5,  # iteration 95
113.5,  # iteration 96
114.75,  # iteration 97
107.25,  # iteration 98
110.0,  # iteration 99
112.0,  # iteration 100
107.75,  # iteration 101
107.25,  # iteration 102
113.75,  # iteration 103
115.25  # iteration 104
```

3. **Why This Happens**:
- The policy has learned to balance the pole, but it's not explicitly learning to keep the cart centered
- The reward function in CartPole only gives +1 for each step the pole stays upright
- There's no explicit penalty for being near the boundaries
- The policy is optimizing for maximum steps (maximum reward) without considering cart position

4. **Why It Doesn't Learn to Stay Centered**:
- The policy gradient algorithm is optimizing for maximum expected return
- Since there's no explicit reward/penalty for cart position:
  - The policy doesn't have a signal to learn to stay centered
  - It only learns to keep the pole upright
  - It can achieve this by moving the cart back and forth
- This is a common issue in policy gradient methods where the policy optimizes for the exact reward function given, not for what we might intuitively want

5. **Why It's Not a Problem**:
- The policy is still achieving its primary goal (keeping the pole upright)
- The returns are still good (around 110-115 steps)
- The policy is still stable and consistent
- The cart hitting boundaries doesn't affect the main objective

6. **How to Fix This**:
If you wanted the cart to stay centered, you would need to:
1. Modify the reward function to include a penalty for distance from center
2. Or use a different environment version (like CartPole-v1) that might have different dynamics
3. Or implement a custom reward function that encourages centering behavior

7. **Why It's Actually Good**:
- The policy is doing exactly what it was trained to do
- It's maximizing the given reward function
- The behavior is consistent and stable
- The policy is robust enough to handle boundary conditions

This is a good example of how reinforcement learning algorithms will optimize for exactly what you tell them to optimize for, not what you might intuitively want. The policy is working as designed - it's maximizing the number of steps the pole stays upright, which it can do by moving the cart back and forth near the boundaries. If you wanted different behavior, you would need to modify the reward function to explicitly encourage that behavior.

The dip in rewards at the end is not a failure of learning, but rather a demonstration that the policy has learned to optimize the exact reward function it was given, even if that leads to behavior that might not be what we intuitively expect.

```plaintext
200.0,  # iteration 85
200.0,  # iteration 86
200.0,  # iteration 87
200.0,  # iteration 88
200.0,  # iteration 89
151.6666717529297,  # iteration 90
200.0,  # iteration 91
136.6666717529297,  # iteration 92
132.5,  # iteration 93
118.25,  # iteration 94
107.5,  # iteration 95
113.5,  # iteration 96
114.75,  # iteration 97
107.25,  # iteration 98
110.0,  # iteration 99
112.0,  # iteration 100
107.75,  # iteration 101
107.25,  # iteration 102
113.75,  # iteration 103
115.25  # iteration 104
```

---

**User**

Ok, but why does the reward dip all the sudden in the end?

---

**Cursor**

Let me analyze why the reward suddenly dips at the end. This is an interesting observation that could have several causes:

1. **Policy Instability**:
- Looking at the returns pattern:
```
200.0,  # iteration 85
200.0,  # iteration 86
200.0,  # iteration 87
200.0,  # iteration 88
200.0,  # iteration 89
151.6666717529297,  # iteration 90  <- Sudden drop
200.0,  # iteration 91
136.6666717529297,  # iteration 92
132.5,  # iteration 93
118.25,  # iteration 94
107.5,  # iteration 95
113.5,  # iteration 96
114.75,  # iteration 97
107.25,  # iteration 98
110.0,  # iteration 99
112.0,  # iteration 100
107.75,  # iteration 101
107.25,  # iteration 102
113.75,  # iteration 103
115.25  # iteration 104
```

2. **Possible Causes**:

a) **Learning Rate Effects**:
- The policy might be overshooting due to the fixed learning rate
- As the policy gets better, the same learning rate might cause larger updates
- This could lead to the policy "jumping" to a worse region of the parameter space

b) **Exploration vs Exploitation**:
- The policy might have become too deterministic
- Without enough exploration, it might get stuck in a local optimum
- The sudden drop could be the policy discovering a new behavior that's less optimal

c) **Numerical Instability**:
- The policy might be experiencing numerical issues
- This could happen if the log probabilities or advantages become too extreme
- The policy might be making updates that are too large

3. **Why It Stays Low**:
- Once the policy moves to this new region, it might be in a local optimum
- The gradient updates might not be strong enough to push it back to the better region
- The policy might have found a new stable behavior that's less optimal but still viable

4. **Why It's Not Recovering**:
- The policy gradient method might not have enough exploration to find its way back
- The learning rate might be too small to make significant changes
- The policy might be in a region where the gradients are small

5. **Potential Solutions**:
- Use a learning rate schedule that decreases over time
- Add entropy regularization to encourage exploration
- Implement early stopping when performance is good
- Use a larger batch size to get more stable updates
- Add gradient clipping to prevent large updates

6. **Why This is Common in Policy Gradient**:
- Policy gradient methods can be sensitive to parameter updates
- They don't have built-in mechanisms to prevent performance degradation
- They can get stuck in local optima
- The optimization landscape can be non-convex and have multiple stable points

The sudden dip and subsequent stability at a lower performance level is a classic example of policy gradient methods' sensitivity to parameter updates and their tendency to get stuck in local optima. It's not necessarily a failure of the algorithm, but rather a demonstration of its limitations in maintaining optimal performance over long training periods.

To prevent this in future runs, you might want to:
1. Implement early stopping when performance is good
2. Use a learning rate schedule
3. Add entropy regularization
4. Use a larger batch size
5. Implement gradient clipping

These modifications could help maintain the high performance achieved earlier in training.

```plaintext
200.0,  # iteration 85
200.0,  # iteration 86
200.0,  # iteration 87
200.0,  # iteration 88
200.0,  # iteration 89
151.6666717529297,  # iteration 90  <- Sudden drop
200.0,  # iteration 91
136.6666717529297,  # iteration 92
132.5,  # iteration 93
118.25,  # iteration 94
107.5,  # iteration 95
113.5,  # iteration 96
114.75,  # iteration 97
107.25,  # iteration 98
110.0,  # iteration 99
112.0,  # iteration 100
107.75,  # iteration 101
107.25,  # iteration 102
113.75,  # iteration 103
115.25  # iteration 104
```

## More on Actor Loss

In reinforcement learning (RL), particularly in actor-critic methods, the **actor loss curve** exhibits characteristic patterns due to the stochastic nature of the environment, the optimization landscape, and the interaction between the actor and the critic. Understanding the shape and behavior of this curve requires a detailed understanding of what actor loss is and how it is optimized.

### Actor Loss Definition

In an actor-critic setup, the **actor** is a policy $\pi_\theta(a|s)$, parameterized by $\theta$, that maps states to action distributions. The **actor loss** is typically formulated as:

$$
\mathcal{L}_{\text{actor}}(\theta) = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ \hat{A}^{\pi}(s, a) \log \pi_\theta(a|s) \right]
$$

Here, $\hat{A}^{\pi}(s, a)$ is the **advantage estimate**, which measures how good action $a$ is compared to the average action at state $s$. This loss encourages the policy to increase the probability of actions with positive advantage and decrease it for negative ones.

In practice, implementations may vary. For instance, **PPO (Proximal Policy Optimization)** uses a clipped surrogate objective, which modifies this loss to ensure stable updates.

### Typical Shape of the Actor Loss Curve

Empirically, the **actor loss curve over training episodes** often exhibits the following pattern:

1. **Initial Increase**: At the beginning of training, the actor loss usually **increases**. This might seem counterintuitive but arises because the critic’s estimate of the advantage function is initially inaccurate. The actor, taking large policy steps based on noisy gradients, may increase the loss even as performance improves.

2. **High Variance Region**: The curve becomes **noisy** and fluctuates due to:
   - Stochasticity in policy sampling.
   - High variance in advantage estimates.
   - Imperfect critic approximations.
   Despite these fluctuations, the **policy performance (e.g., reward)** may still be improving.

3. **Gradual Decrease or Plateau**: As the policy improves and the critic stabilizes, the advantage estimates become more accurate. The actor starts making finer updates. Consequently, the loss begins to **stabilize** or **gradually decrease**. This phase corresponds to **policy refinement**.

4. **Convergence or Divergence**:
   - If learning converges successfully, the actor loss flattens and shows low variance.
   - If the policy overfits or the learning rate is high, the curve might **diverge** again due to **policy collapse** or **destabilization** of the critic.

### Why the Curve Behaves This Way

The behavior arises due to the nature of **policy gradient optimization**:

#### 1. Non-stationarity

The target (advantage estimate) is changing throughout training, so the loss surface itself is non-stationary.

#### 2. High variance gradient estimation

Policy gradient methods estimate gradients via Monte Carlo sampling, leading to noisy updates:

$$
\nabla_\theta \mathcal{L}_{\text{actor}}(\theta) = - \mathbb{E} \left[ \hat{A}(s,a) \nabla_\theta \log \pi_\theta(a|s) \right]
$$

Even if the policy improves, the log-probabilities can vary significantly, affecting the sign and magnitude of the loss.

#### 3. Clipping in PPO

In PPO, the clipped objective constrains the update:

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E} \left[ \min\left(r(\theta) \hat{A}, \text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A} \right) \right]
$$

This leads to oscillations in loss, especially if advantage estimates straddle zero. Clipping also means the loss doesn't necessarily correlate monotonically with policy performance.

#### 4. Critic's Influence

If the critic provides poor estimates (e.g., wrong sign on advantages), the actor is misdirected, temporarily increasing loss or degrading performance.

### Conclusion

The actor loss curve is **noisy, non-monotonic**, and often **not a good proxy** for policy performance. It typically rises early in training, oscillates with high variance, and may flatten or drop as learning stabilizes. This reflects the complex interplay between stochastic sampling, non-stationary targets, and the inherent variance of policy gradient methods.

Monitoring **reward curves** and **KL divergence** between old and new policies is often more informative than observing actor loss alone.

---

### Why Improved Policy Does Not Necessarily Reduce Actor Loss

This is a subtle and crucial point in reinforcement learning: **an improved policy does not necessarily imply a decrease in actor loss**. The reason lies in the *nature of the actor loss function* and how it interacts with the **policy gradient formulation**, particularly in stochastic settings.

Let’s unpack this in depth.

#### 1. Actor Loss is Not a True Objective Function

In supervised learning, the loss function (e.g., MSE or cross-entropy) is directly minimized to improve predictive accuracy. However, in **policy gradient RL**, the "loss" is a *proxy for a directional update*. The actor loss is designed such that:

$$
\nabla_\theta \mathcal{L}_{\text{actor}}(\theta) \propto -\mathbb{E} \left[ \hat{A}(s, a) \nabla_\theta \log \pi_\theta(a|s) \right]
$$

This means we are *maximizing expected return* by ascending the gradient, but the scalar value of the loss itself (i.e., $\mathcal{L}_{\text{actor}}(\theta)$) has **no intrinsic meaning as a performance metric**. Its numerical value is affected by advantage magnitudes, policy entropy, and sampling variability.

#### 2. Improved Policy May Have Zero or Near-Zero Advantage

As training proceeds and the policy converges toward optimal behavior, the estimated advantage $\hat{A}(s, a) \approx 0$ for most actions, because:

$$
\hat{A}(s, a) = Q(s, a) - V(s)
$$

If the policy becomes nearly deterministic and well-matched to the critic’s value estimates, then $Q(s,a) \approx V(s)$. As a result, the gradient signal $\hat{A}(s,a) \log \pi_\theta(a|s)$ goes to zero, and so does the actor loss—even without any further policy improvement.

But here's the twist: the actor loss can also **remain high or oscillate**, even with a good policy, depending on how **policy entropy** and **advantage magnitudes** behave.

#### 3. High Actor Loss Can Persist With a Good Policy

Let’s consider an illustrative example.

Suppose the policy has improved substantially—it now consistently selects near-optimal actions. But if the advantage estimates $\hat{A}(s,a)$ have high magnitude (perhaps due to large returns or poor value normalization), and the log-probabilities $\log \pi_\theta(a|s)$ are small (due to high entropy or stochasticity), then:

$$
\mathcal{L}_{\text{actor}}(\theta) = - \mathbb{E}[ \hat{A}(s,a) \log \pi_\theta(a|s) ]
$$

can still yield a **large negative value**, i.e., a large positive loss.

Thus, high loss does not mean the policy is poor. It just means the **update direction has high magnitude**.

#### 4. Clipped Surrogate Loss in PPO Obfuscates the Meaning of the Loss Value

In PPO, the actor loss is defined as:

$$
\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E} \left[ \min\left(r(\theta) \hat{A}, \text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A} \right) \right]
$$

Here $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$. The clipping restricts the update magnitude to prevent divergence. As the policy improves and the updates become conservative (e.g., KL divergence < threshold), the clipped loss becomes **constant**, regardless of how optimal the policy is. Again, this means loss value **does not track policy quality**.

#### 5. Entropy Regularization and Exploration Terms Add Noise

If entropy regularization is used (common in actor-critic methods like A3C, SAC), then the actor loss also includes a term like:

$$
\mathcal{L}_{\text{actor}}(\theta) = -\mathbb{E} \left[ \hat{A}(s,a) \log \pi_\theta(a|s) + \beta \mathcal{H}[\pi_\theta(\cdot|s)] \right]
$$

This entropy term encourages exploration and can **increase the loss** even as the policy improves. As entropy decays (in entropy-annealing schedules), the actor loss may decrease, but this is due to **reduced exploration**, not better performance.

### Conclusion

An improved policy leads to better reward and often more stable behavior—but **not necessarily to lower actor loss**. This is because the actor loss is not a performance measure but a stochastic estimate of the gradient signal guiding policy improvement. It depends on fluctuating advantage estimates, policy entropy, and implementation details (e.g., clipping in PPO).

In short: **the actor loss is a tool for optimization, not a scorecard**.
