## Results
To be added.


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