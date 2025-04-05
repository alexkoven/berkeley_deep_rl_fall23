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


## More Info on the Replay Buffer

## Basic Structure and Purpose
The replay buffer is a crucial component in many reinforcement learning algorithms, particularly in off-policy methods. It maintains two types of storage:
- Raw rollouts (`self.paths`): Stores complete trajectory data
- Processed arrays: Stores concatenated components of the experiences (`obs`, `acs`, `rews`, `next_obs`, `terminals`)

## Key Components
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

## Storage Methods

### Complete Trajectories vs Chunked Storage
The buffer uses a dual storage system:
1. **Complete Trajectories** (`self.paths`):
   - Used for visualization/debugging (creating videos)
   - Special algorithms needing full trajectory information (like DAgger's relabeling)
   - Preserves episode boundaries and temporal relationships

2. **Chunked Storage** (Component Arrays):
   - Used for efficient random sampling during training
   - Allows easy batch creation
   - Better memory management through `max_size` parameter

### Usage of Complete Trajectories
Complete trajectories are primarily used for:
1. **Video Logging/Visualization**
   - Creating videos of agent behavior for TensorBoard
   - Used across different algorithms (DQN, SAC, BC/DAgger)

2. **Expert Data Loading/DAgger**
   - Loading expert demonstrations
   - Relabeling trajectories with expert actions in DAgger

## Data Processing (convert_listofrollouts function)

The `convert_listofrollouts` function transforms trajectory data into flat arrays:

### Parameters
1. `paths`: List of dictionaries (each dictionary is a trajectory)
2. `concat_rew`: Boolean flag for reward processing

### Path Dictionary Structure
Each path contains:
- `"observation"`: States/observations
- `"action"`: Actions taken
- `"reward"`: Rewards received
- `"next_observation"`: Next states
- `"terminal"`: Terminal flags

### Processing Steps
1. **Observations**: Concatenates all observation arrays
2. **Actions**: Concatenates all action arrays
3. **Rewards**: 
   - If `concat_rew=True`: One concatenated array
   - If `concat_rew=False`: List of separate arrays
4. **Next Observations**: Concatenates all next state arrays
5. **Terminals**: Concatenates all terminal flags

### Example
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