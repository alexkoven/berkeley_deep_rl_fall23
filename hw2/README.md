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

## Relevant Side Information

### Difference to HW1 Policy in networks/policies.py
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



## Setup

See [installation.md](installation.md). It's worth going through this again since some dependencies have changed since homework 1. You also need to make sure to run `pip install -e .` in the hw2 folder.

## Running on Google Cloud
Starting with HW2, we will be providing some infrastructure to run experiments on Google Cloud compute. There are some very important caveats:

- **Do not leave your instance running.** The provided infrastructure tries to prevent this, but it will still be easy to accidentally leave your instance running and burn through all of your credits. You are responsible for making sure you use your credits wisely.
- **Only use this for big hyperparameter sweeps.** Definitely don't use Google Cloud for debugging; only launch a job once you are 100% sure your code works. Even then, single jobs will probably run faster on your local machine (yes, even if you don't have a GPU). The only reason to use Google Cloud is if you want to run multiple jobs in parallel.

For more instructions, see [google_cloud/README.md](google_cloud/README.md).

## Assignment

There are TODOs in these files:

- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/infrastructure/utils.py`

See the [Assignment PDF](hw2.pdf) for more info.