## Table of Contents
- [Results](#results)
  - [Behavior Cloning (BC)](#behavior-cloning-bc)
  - [Imitation Learning (DAgger)](#imitation-learning-dagger)
- [Algorithm Workflow and Insights](#algorithm-workflow-and-insights)
  - [Behavior Cloning](#behavior-cloning)
  - [Imitation Learning](#imitation-learning)
- [Setup](#setup)
- [Complete the code](#complete-the-code)
- [Run the code](#run-the-code)
  - [Section 1 (Behavior Cloning)](#section-1-behavior-cloning)
  - [Section 2 (DAgger)](#section-2-dagger)
- [More Details](#more-details)
  - [1. Why is BC only trained for one iteration?](#1-why-is-bc-only-trained-for-one-iteration)
  - [2. How does the replay buffer work?](#2-how-does-the-replay-buffer-work)
  - [3. What kind of gradient update is used and how does it work?](#3-what-kind-of-gradient-update-is-used-and-how-does-it-work)
  - [4. How exactly does the replay buffer work?](#4-how-exactly-does-the-replay-buffer-work)
  - [5. Are all trajectories relabeled by DAGGER?](#5-are-all-trajectories-relabeled-by-dagger)
  - [Understanding mean and variance for MLP](#understanding-mean-and-variance-for-mlp)
  - [Would a neural network output two parameters, the mean and the variance?](#would-a-neural-network-output-two-parameters-the-mean-and-the-variance)
  - [How does the update function work?](#how-does-the-update-function-work)


## Results

### Behavior Cloning (BC)
| Metric | Value |
|--------|--------|
| Eval_AverageReturn | 4172.27 |
| Eval_StdReturn | 0.00 |
| Eval_MaxReturn | 4172.27 |
| Eval_MinReturn | 4172.27 |
| Eval_AverageEpLen | 1000.00 |
| Train_AverageReturn | 4681.89 |
| Train_StdReturn | 30.71 |
| Train_MaxReturn | 4712.60 |
| Train_MinReturn | 4651.18 |
| Train_AverageEpLen | 1000.00 |
| Training Loss | -1.92 |
| Train_EnvstepsSoFar | 0.00 |
| TimeSinceStart | 4.80 |
| Initial_DataCollection_AverageReturn | 4681.89 |


Before Training:

![Random Policy](videos/untrained_spider.gif)

After Training:

![BC Ant Policy Trained](videos/spider_bc_trained.gif)

### Imitation Learning (DAgger)
| Metric | Value |
|--------|--------|
| Eval_AverageReturn | 4770.26 |
| Eval_StdReturn | 121.56 |
| Eval_MaxReturn | 4941.97 |
| Eval_MinReturn | 4611.63 |
| Eval_AverageEpLen | 1000.00 |
| Train_AverageReturn | 4723.17 |
| Train_StdReturn | 0.00 |
| Train_MaxReturn | 4723.17 |
| Train_MinReturn | 4723.17 |
| Train_AverageEpLen | 1000.00 |
| Training Loss | -2.55 |
| Train_EnvstepsSoFar | 9000.00 |
| TimeSinceStart | 58.27 |


After Training:

![IL Ant Policy Trained](videos/spider_il_trained.gif)

## Algorithm Workflow and Insights

### Behavior Cloning
1. Entry Point: `main()` in `run_hw1.py`
   ```
   main()
   ├── parse command line arguments
   ├── setup logging directory
   └── run_training_loop(params)
   ```

2. Training Loop Setup in `run_training_loop(params)`:
   ```
   run_training_loop(params)
   ├── Initialize Logger
   ├── Setup Environment
   │   └── gym.make(params['env_name'])
   ├── Create Actor
   │   └── MLPPolicySL(ac_dim, ob_dim, n_layers, size, learning_rate)
   ├── Create Replay Buffer
   │   └── ReplayBuffer(max_size)
   └── Load Expert Policy
       └── LoadedGaussianPolicy(expert_policy_file)
   ```

3. Training Process:
   ```
   Training Loop
   ├── Record initial performance
   │   └── utils.sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)
   │
   ├── Load expert data
   │   └── pickle.load(expert_data)
   │
   ├── Add expert data to replay buffer
   │   └── replay_buffer.add_rollouts(expert_data)
   │
   └── Training iterations (num_agent_train_steps_per_iter times)
       ├── Sample batch from replay buffer
       │   ├── ob_batch = replay_buffer.obs[rand_indices]
       │   └── ac_batch = replay_buffer.acs[rand_indices]
       │
       └── Update policy
           └── actor.update(ob_batch, ac_batch)
               ├── forward(observations)
               │   ├── mean_net(observation) -> mean
               │   ├── exp(logstd) -> std
               │   └── distributions.Normal(mean, std)
               │
               └── Compute & apply gradients
                   ├── -distribution.log_prob(actions).mean() -> loss
                   ├── loss.backward()
                   └── optimizer.step()
   ```

4. Evaluation & Logging:
   ```
   Logging
   ├── Collect evaluation rollouts
   │   └── utils.sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)
   │
   └── Log metrics & videos
       └── logger.log_paths_as_videos(eval_video_paths)
   ```

The key points about this flow:

1. BC is a single iteration process (n_iter=1) because it only needs the initial expert data
2. Within that single iteration, it performs multiple gradient updates (num_agent_train_steps_per_iter times)
3. The policy is a Gaussian policy that outputs a mean and learned standard deviation
4. During training, it uses the reparameterization trick for sampling actions
5. During evaluation/inference, it just uses the mean action
6. The loss function is the negative log likelihood of the expert actions under the policy's distribution

This explains why we don't use the sampled actions (`pred_actions`) in the loss - we're training the policy to maximize the likelihood of the expert actions under its distribution, not to minimize the difference between sampled actions and expert actions.

### Imitation Learning

1. Entry Point: `main()` in `run_hw1.py`
   ```
   main()
   ├── parse command line arguments
   │   └── assert n_iter > 1 for DAgger
   ├── setup logging directory
   └── run_training_loop(params)
   ```

2. Training Loop Setup (same as BC):
   ```
   run_training_loop(params)
   ├── Initialize Logger
   ├── Setup Environment
   │   └── gym.make(params['env_name'])
   ├── Create Actor
   │   └── MLPPolicySL(ac_dim, ob_dim, n_layers, size, learning_rate)
   ├── Create Replay Buffer
   │   └── ReplayBuffer(max_size)
   └── Load Expert Policy
       └── LoadedGaussianPolicy(expert_policy_file)
   ```

3. DAgger Training Process (for each iteration):
   ```
   Training Loop (n_iter times)
   ├── Iteration 0 (Behavior Cloning)
   │   ├── Load expert data
   │   │   └── pickle.load(expert_data)
   │   └── Add to replay buffer
   │       └── replay_buffer.add_rollouts(expert_data)
   │
   └── Iterations 1 to n-1 (DAgger)
       ├── Collect new trajectories with current policy
       │   └── utils.sample_trajectories(env, actor, batch_size, ep_len)
       │       ├── Reset environment
       │       └── For each step:
       │           ├── Get action from current policy
       │           └── Step environment
       │
       ├── Relabel trajectories with expert
       │   └── For each trajectory:
       │       ├── Keep states from student policy
       │       └── Replace actions with expert actions
       │           └── expert_policy.get_action(observations)
       │
       ├── Add to replay buffer
       │   └── replay_buffer.add_rollouts(relabeled_paths)
       │
       └── Train policy
           └── For num_agent_train_steps_per_iter times:
               ├── Sample batch from replay buffer
               │   ├── ob_batch = replay_buffer.obs[rand_indices]
               │   └── ac_batch = replay_buffer.acs[rand_indices]
               │
               └── Update policy
                   └── actor.update(ob_batch, ac_batch)
                       ├── forward(observations)
                       │   ├── mean_net(observation) -> mean
                       │   ├── exp(logstd) -> std
                       │   └── distributions.Normal(mean, std)
                       │
                       └── Compute & apply gradients
                           ├── -distribution.log_prob(actions).mean() -> loss
                           ├── loss.backward()
                           └── optimizer.step()
   ```

4. Evaluation & Logging (same as BC):
   ```
   Logging
   ├── Collect evaluation rollouts
   │   └── utils.sample_trajectories(env, actor, eval_batch_size, ep_len)
   │
   ├── Compute metrics
   │   └── utils.compute_metrics(paths, eval_paths)
   │
   └── Log metrics & videos
       └── logger.log_paths_as_videos(eval_video_paths)
   ```

Key Differences from BC:
1. Multiple iterations (n_iter > 1) instead of just one
2. After iteration 0, collects new data using the student policy
3. Relabels all collected trajectories with expert actions (discards student actions, which may seem wasteful. See later part of HW1 `SwitchDAgger`)
4. Aggregates data over time in the replay buffer
5. Each iteration adds more data to the training set
6. Training happens on the growing dataset that includes:
   - Original expert demonstrations
   - Student-collected states with expert-relabeled actions

This approach helps address covariate shift by:
- Learning from states the student policy actually visits
- Getting expert feedback on those states
- Building a dataset that covers both expert and student state distributions
- Teaching the policy how to recover from its own mistakes


## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2023/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, edit
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py)

You have the option of running locally or on Colab using
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

---

## More Details

### 1. Why is BC only trained for one iteration?

Behavior Cloning (BC) is trained for only one iteration because it's a simple supervised learning approach that learns directly from a fixed dataset of expert demonstrations. Here's the detailed explanation:

- BC works by learning a direct mapping from states to actions by mimicking the expert's behavior
- It only needs one iteration because:
  1. The expert dataset is fixed and doesn't change
  2. Within this one iteration, it performs multiple gradient updates (controlled by `num_agent_train_steps_per_iter`)
  3. Unlike DAgger, it doesn't collect new data or interact with the environment
  4. It's a pure supervised learning problem - just trying to match the expert's actions

This is enforced in the code:
```python
if args.do_dagger:
    assert args.n_iter>1, ('DAGGER needs more than 1 iteration...')
else:
    assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')
```

### 2. How does the replay buffer work?

The replay buffer has two main components:

#### Storage Structure:
```python
def __init__(self, max_size=1000000):
    self.max_size = max_size
    self.paths = []  # store each rollout
    # store (concatenated) component arrays
    self.obs = None
    self.acs = None
    self.rews = None
    self.next_obs = None
    self.terminals = None
```

#### Data Organization:
- Trajectories are stored both as complete paths and as decomposed components
- When adding data:
```python
def add_rollouts(self, paths, concat_rew=True):
    # add new rollouts into list of rollouts
    for path in paths:
        self.paths.append(path)
        
    # convert rollouts into component arrays
    observations, actions, rewards, next_observations, terminals = convert_listofrollouts(paths)
```
- The data is stored in both formats:
  1. Complete trajectories in `self.paths`
  2. Flattened arrays in `self.obs`, `self.acs`, etc.

This dual storage allows for:
- Easy access to complete trajectories when needed
- Efficient sampling of individual transitions for training
- The buffer maintains a maximum size and overwrites old data when full using modulo operations

### 3. What kind of gradient update is used and how does it work?

The gradient update used in the `MLPPolicySL` class is a simple supervised learning update using Mean Squared Error (MSE) loss. Here's how it works:

```python
def update(self, observations, actions):
    # Convert to tensors
    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(actions)
    
    # Get predicted actions from policy network
    predicted_actions = self.mean_net(observations)
    
    # Calculate MSE loss
    loss = F.mse_loss(predicted_actions, actions)
    
    # Gradient descent step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

The process:
1. The policy network predicts actions for given observations
2. MSE loss is computed between predicted and expert actions
3. Gradients are computed through backpropagation
4. The Adam optimizer updates the network parameters to minimize the loss
5. The network architecture is an MLP (Multi-Layer Perceptron) with:
   - Input size: observation dimension
   - Output size: action dimension
   - Configurable hidden layers and sizes
   - Tanh activation functions between layers

This is a standard supervised learning setup, treating the expert's actions as the ground truth labels to be imitated.


### 4. How exactly does the replay buffer work?

#### Basic Structure and Purpose
The replay buffer is a crucial component in many reinforcement learning algorithms, particularly in off-policy methods. It maintains two types of storage:
- Raw rollouts (`self.paths`): Stores complete trajectory data
- Processed arrays: Stores concatenated components of the experiences (`obs`, `acs`, `rews`, `next_obs`, `terminals`)

#### Key Components
```python
def __init__(self, max_size=1000000):
    self.max_size = max_size  # Maximum number of transitions to store
    self.paths = []  # List to store complete trajectories
    # Component arrays (initialized as None)
    self.obs = None        # Observations
    self.acs = None        # Actions
    self.rews = None       # Rewards
    self.next_obs = None   # Next observations
    self.terminals = None  # Terminal flags
```

#### Storage Methods

##### Complete Trajectories vs Chunked Storage
The buffer uses a dual storage system:
1. **Complete Trajectories** (`self.paths`):
   - Used for visualization/debugging (creating videos)
   - Special algorithms needing full trajectory information (like DAgger's relabeling)
   - Preserves episode boundaries and temporal relationships

2. **Chunked Storage** (Component Arrays):
   - Used for efficient random sampling during training
   - Allows easy batch creation
   - Better memory management through `max_size` parameter

##### Usage of Complete Trajectories
Complete trajectories are primarily used for:
1. **Video Logging/Visualization**
   - Creating videos of agent behavior for TensorBoard
   - Used across different algorithms (DQN, SAC, BC/DAgger)

2. **Expert Data Loading/DAgger**
   - Loading expert demonstrations
   - Relabeling trajectories with expert actions in DAgger

#### Data Processing (convert_listofrollouts function)

The `convert_listofrollouts` function transforms trajectory data into a flat format:

##### Parameters
1. `paths`: List of dictionaries (each dictionary is a trajectory)
2. `concat_rew`: Boolean flag for reward processing

##### Path Dictionary Structure
Each path contains:
- `"observation"`: States/observations
- `"action"`: Actions taken
- `"reward"`: Rewards received
- `"next_observation"`: Next states
- `"terminal"`: Terminal flags

##### Processing Steps
1. **Observations**: Concatenates all observation arrays
2. **Actions**: Concatenates all action arrays
3. **Rewards**: 
   - If `concat_rew=True`: One concatenated array
   - If `concat_rew=False`: List of separate arrays
4. **Next Observations**: Concatenates all next state arrays
5. **Terminals**: Concatenates all terminal flags

##### Example
```python
# Input paths:
paths = [
    {
        "observation": np.array([[1,1], [2,2]]),      # 2 timesteps
        "action": np.array([[0], [1]]),
        "reward": np.array([0.5, 1.0]),
        "next_observation": np.array([[2,2], [3,3]]),
        "terminal": np.array([0, 1])
    },
    {
        "observation": np.array([[4,4]]),             # 1 timestep
        "action": np.array([[0]]),
        "reward": np.array([0.7]),
        "next_observation": np.array([[5,5]]),
        "terminal": np.array([1])
    }
]

# After conversion:
observations = np.array([[1,1], [2,2], [4,4]])        # 3 timesteps total
actions = np.array([[0], [1], [0]])
rewards = np.array([0.5, 1.0, 0.7])
next_observations = np.array([[2,2], [3,3], [5,5]])
terminals = np.array([0, 1, 1])
```

The function converts trajectory-based data into a flat format for easier random sampling during training, particularly useful for off-policy algorithms that sample individual transitions rather than complete trajectories.

### 5. Are all trajectories relabeled by DAGGER?

1. **Initial Training Phase**:
- DAGGER starts with pure Behavior Cloning (BC) in the first iteration (itr=0)
- It uses expert demonstrations loaded from a pickle file:
```python
if itr == 0:
    # BC training from expert data.
    paths = pickle.load(open(params['expert_data'], 'rb'))
    envsteps_this_batch = 0
```

2. **Data Collection Phase** (itr > 0):
- For each subsequent iteration, DAGGER:
  1. Collects new trajectories using the current policy
  2. Uses `utils.sample_trajectories()` to gather `batch_size` transitions
  3. Each trajectory contains:
     - observations
     - actions (from current policy)
     - rewards
     - next observations
     - terminal flags

3. **Relabeling Process**:
```python
if params['do_dagger']:
    print("\nRelabelling collected observations with labels from an expert policy...")
    for path in paths:
        expert_actions = expert_policy.get_action(path["observation"])
        path["action"] = expert_actions
```

The key aspects of the relabeling process are:

a) **What gets relabeled**:
- Only the actions are relabeled
- The states/observations remain unchanged
- The original trajectories' structure is preserved

b) **How relabeling works**:
1. Takes the observations from the collected trajectories (using student policy)
2. Feeds each observation through the expert policy
3. Replaces the student's actions with the expert's actions
4. Maintains the original state-next_state relationships

4. **Data Aggregation**:
```python
total_envsteps += envsteps_this_batch
replay_buffer.add_rollouts(paths)
```
- The relabeled trajectories are added to a replay buffer
- This aggregates data across iterations, combining:
  - Original expert demonstrations
  - New trajectories with expert-relabeled actions

5. **Training Update**:
```python
print('\nTraining agent using sampled data from replay buffer...')
```
- The policy is then trained on the aggregated dataset
- This includes both:
  - Original expert demonstrations
  - Newly collected states with expert-relabeled actions

The key insight is that DAGGER:
1. Collects states/trajectories using the student's current policy
2. Relabels only the actions using the expert policy
3. Aggregates this data with previous iterations
4. Trains on the growing dataset

This approach helps address the covariate shift problem in Behavior Cloning by:
- Collecting states that the student policy actually visits
- Getting expert feedback on those states
- Building a dataset that covers the state distribution the student encounters
- Maintaining the expert's optimal actions for those states

This creates a more robust policy that learns how to recover from its own mistakes, as it sees expert actions for states it encounters, rather than just states the expert encounters.


Yes, all trajectories collected after iteration 0 (the initial BC training) are relabeled. Here's the detailed breakdown:

1. **Iteration 0 (BC Phase)**:
   - Uses pure expert demonstrations
   - No relabeling occurs
   - Data comes directly from expert_data file

2. **All Subsequent Iterations (DAGGER Phase)**:
   - Every single trajectory collected by the student policy is relabeled
   - The relabeling loop iterates through all paths: `for path in paths`
   - For each path, all observations are passed through the expert policy to get new actions
   - The original actions are completely replaced with expert actions: `path["action"] = expert_actions`

This is actually a key feature of DAGGER's design:
- It wants to learn the expert's policy for ALL states visited by the student
- There's no selective relabeling or partial relabeling
- The goal is to get expert feedback on every single state encountered by the student policy
- This comprehensive relabeling helps create a robust policy that can recover from mistakes

This is different from some other imitation learning algorithms that might selectively relabel or only relabel certain states/trajectories. DAGGER takes the approach of getting expert feedback on everything the student encounters to build a comprehensive training dataset.


### Understanding mean and variance for MLP


Let me explain the role and significance of the `logstd` parameter in this policy network implementation.

In this MLP policy, we're implementing what's known as a Gaussian policy for continuous action spaces. The policy outputs a probability distribution over actions rather than deterministic actions. Specifically, it outputs a diagonal Gaussian distribution characterized by two parameters:

1. The mean (μ) - computed by the neural network (`mean_net`)
2. The standard deviation (σ) - parameterized through `logstd`

The `logstd` parameter is defined as:
```python
self.logstd = nn.Parameter(
    torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
)
```

Here's why this is important:

1. **Why nn.Parameter?**
   - `nn.Parameter` is a special kind of Tensor in PyTorch that gets automatically registered as a parameter of the model
   - When you wrap a tensor with `nn.Parameter`, it tells PyTorch that this tensor should be considered a trainable parameter
   - This means during backpropagation, gradients will be computed with respect to this parameter, and it will be updated during optimization

2. **Why logarithm of standard deviation?**
   - We use log(σ) instead of σ directly because:
     - It ensures the actual standard deviation is always positive (since σ = exp(log(σ)))
     - It's numerically more stable for optimization
     - It allows for more efficient exploration of different scales of variance

3. **Shape and Initialization**
   - The `logstd` has shape `[ac_dim]`, meaning we learn a separate standard deviation for each dimension of the action space
   - It's initialized to zeros, meaning initial σ = exp(0) = 1 for all dimensions
   - This is a common practice as it starts with moderate exploration

4. **Usage in the Policy**
   - During forward pass (though not implemented in this code yet), typically:
     1. The `mean_net` computes the mean action μ
     2. The `logstd` parameter gives log(σ)
     3. A Gaussian distribution is created with these parameters: N(μ, σ)
     4. Actions are sampled from this distribution

5. **Learning Process**
   - The optimizer is set up to update both the network parameters and logstd:
   ```python
   self.optimizer = optim.Adam(
       itertools.chain([self.logstd], self.mean_net.parameters()),
       self.learning_rate
   )
   ```
   - During training, both the mean (through network weights) and standard deviation (through logstd) are learned
   - This allows the policy to learn not just the best action (mean) but also how uncertain it should be about those actions (variance)

This dual parameterization (mean + learned standard deviation) is crucial for:
- Exploration: Higher σ means more exploration in that action dimension
- Uncertainty representation: The policy can learn to be more certain about some actions than others
- Stable learning: The log parameterization ensures stable training dynamics

This is a standard approach in policy gradient methods for continuous control, where having a stochastic policy is important for both exploration and robustness.


### Would a neural network output two parameters, the mean and the variance?

In this specific implementation, the neural network (`mean_net`) only outputs the mean, while `logstd` is a separate learnable parameter that exists outside the neural network architecture. Let me explain this design choice and contrast it with alternatives:

**Current Implementation (Separate Parameter)**:
```python
# From the code:
self.mean_net = build_mlp(
    input_size=self.ob_dim,
    output_size=self.ac_dim,
    n_layers=self.n_layers, size=self.size,
)
self.logstd = nn.Parameter(torch.zeros(self.ac_dim))
```

In this design:
1. The neural network outputs only the mean (shape: `[batch_size, ac_dim]`)
2. The `logstd` is a single vector (shape: `[ac_dim]`) shared across all inputs
3. The same standard deviation is used for all observations, but it's still learned during training

**Alternative Design (Network Outputs Both)**:
You could alternatively have the network output both μ and log(σ). It would look something like:

```python
# Alternative implementation (not in current code)
self.network = build_mlp(
    input_size=self.ob_dim,
    output_size=2 * self.ac_dim,  # Double the output size
    n_layers=self.n_layers,
    size=self.size,
)
```

In this case:
1. The network would output `2 * ac_dim` values
2. First `ac_dim` values would be the mean
3. Second `ac_dim` values would be the logstd
4. Each observation would get its own mean AND standard deviation

**Why Use Separate Parameter?**
The current implementation (separate `logstd`) is chosen because:
1. **Simplicity**: Having a global learnable variance is often sufficient for many tasks
2. **Stability**: Learning state-dependent variances can be more unstable
3. **Sample Efficiency**: Fewer parameters to learn, which can be important in RL
4. **Controlled Exploration**: Global variance parameters make it easier to control exploration behavior

Both approaches are valid and used in practice. The choice often depends on:
- Whether you want state-dependent uncertainty
- The complexity of the task
- How much data you have
- How stable you need the training to be

For this homework implementation, using a separate `logstd` parameter provides a good balance of simplicity and effectiveness for learning a basic policy.

### How does the update function work?

Yes, exactly! Let me break down the exact mathematical process happening in the `update` function of `MLPPolicySL`:


1. First, we get a distribution over actions from our policy network:
```python
_, distribution = self.forward(observations)
```
This gives us a Gaussian distribution where:
- The mean (μ) comes from the neural network: `mean = self.mean_net(observation)`
- The standard deviation (σ) comes from the learnable parameter: `std = torch.exp(self.logstd)`

2. Then we compute the loss using negative log likelihood:
```python
loss = -distribution.log_prob(actions).mean()
```

This is mathematically equivalent to:
$$L(\theta) = -\frac{1}{N}\sum_{i=1}^N \log \pi_\theta(a_i|s_i)$$

where for a Gaussian policy:
$$\log \pi_\theta(a|s) = -\frac{(a - \mu_\theta(s))^2}{2\sigma^2} - \log(\sigma) - \frac{1}{2}\log(2\pi)$$

3. The gradient descent step then updates:
- The mean network parameters to make the mean closer to the expert actions
- The logstd parameter to adjust the policy's uncertainty

This approach:
- Maximizes the likelihood of expert actions under our policy's distribution
- Learns both the optimal action (mean) and the appropriate variance
- Uses the reparameterization trick during training for stable gradients
- During inference, just uses the mean action (most likely action)

So you're exactly right - we're trying to make our policy's distribution match the expert's actions by maximizing the likelihood of those actions under our policy's distribution. The mean of our policy should converge toward the expert's actions, while the standard deviation represents our policy's uncertainty.
