# Deep RL HW1 - Behavior Cloning and DAgger Implementation Details

## 1. Why is BC only trained for one iteration?

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

## 2. How does the replay buffer work?

The replay buffer has two main components:

### Storage Structure:
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

### Data Organization:
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

## 3. What kind of gradient update is used and how does it work?

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


## 4. How exactly does the replay buffer work?

### Basic Structure and Purpose
The replay buffer is a crucial component in many reinforcement learning algorithms, particularly in off-policy methods. It maintains two types of storage:
- Raw rollouts (`self.paths`): Stores complete trajectory data
- Processed arrays: Stores concatenated components of the experiences (`obs`, `acs`, `rews`, `next_obs`, `terminals`)

### Key Components
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

### Storage Methods

#### Complete Trajectories vs Chunked Storage
The buffer uses a dual storage system:
1. **Complete Trajectories** (`self.paths`):
   - Used for visualization/debugging (creating videos)
   - Special algorithms needing full trajectory information (like DAgger's relabeling)
   - Preserves episode boundaries and temporal relationships

2. **Chunked Storage** (Component Arrays):
   - Used for efficient random sampling during training
   - Allows easy batch creation
   - Better memory management through `max_size` parameter

#### Usage of Complete Trajectories
Complete trajectories are primarily used for:
1. **Video Logging/Visualization**
   - Creating videos of agent behavior for TensorBoard
   - Used across different algorithms (DQN, SAC, BC/DAgger)

2. **Expert Data Loading/DAgger**
   - Loading expert demonstrations
   - Relabeling trajectories with expert actions in DAgger

### Data Processing (convert_listofrollouts function)

The `convert_listofrollouts` function transforms trajectory data into flat arrays:

#### Parameters
1. `paths`: List of dictionaries (each dictionary is a trajectory)
2. `concat_rew`: Boolean flag for reward processing

#### Path Dictionary Structure
Each path contains:
- `"observation"`: States/observations
- `"action"`: Actions taken
- `"reward"`: Rewards received
- `"next_observation"`: Next states
- `"terminal"`: Terminal flags

#### Processing Steps
1. **Observations**: Concatenates all observation arrays
2. **Actions**: Concatenates all action arrays
3. **Rewards**: 
   - If `concat_rew=True`: One concatenated array
   - If `concat_rew=False`: List of separate arrays
4. **Next Observations**: Concatenates all next state arrays
5. **Terminals**: Concatenates all terminal flags

#### Example
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

## 5. Are all trajectories relabeled by DAGGER?

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
