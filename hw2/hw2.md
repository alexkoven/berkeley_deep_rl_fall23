## CS 285 HW2: Policy Gradients – To-Do List

### I Core Implementation

#### 1. `pg_agent.py`
- [x] Implement **return estimators**:
  - [x] **Case 1:** Full-trajectory discounted returns 
  - [x] **Case 2:** Reward-to-go discounted returns
- [ ] Skip baseline code sections for now

#### 2. `estimate_advantage` (in `pg_agent.py`)
- [x] Complete the `estimate_advantage` method:
  - [x] Implement vanilla advantage = reward-to-go
  - [ ] Later: subtract NN baseline
  - [ ] Later: implement GAE

#### 3. `MLPPolicyPG:update` method
- [ ] Finish the `update` method for:
  - [ ] Policy parameter update
  - [ ] Neural net baseline update (if `nn_baseline=True`)

---

### II Experiments and Deliverables

#### Experiment 1: CartPole (Discrete)
- [ ] Run 8 combinations (small + large batch; with/without reward-to-go; with/without advantage normalization):
  bash
  python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b <1000|4000> [-rtg] [-na] --exp_name <name>
`

- [ ] Plot learning curves:

  - [ ] Small batch configs (prefix: `cartpole`)
  - [ ] Large batch configs (prefix: `cartpole_lb`)
- [ ] Answer:

  - [ ] Which value estimator performs better without normalization?
  - [ ] Does advantage normalization help?
  - [ ] Impact of batch size?
- [ ] Include all command-line configs used

---

#### Experiment 2: HalfCheetah (NN Baseline)

- [ ] Implement neural network baseline support (update baseline, subtract prediction)
- [ ] Run:

  bash
  % No baseline
  python ... --use_reward_to_go

  % With baseline
  python ... --use_baseline -blr 0.01 -bgs 5
  
- [ ] Plot:

  - [ ] Baseline loss
  - [ ] Evaluation return
- [ ] Experiment with:

  - [ ] Decreased `-bgs` and/or `-blr`
  - [ ] Optionally, add `-na` and video logging (`--video_log_freq 10`)

---

#### Experiment 3: LunarLander (GAE)

- [ ] Implement GAE (`estimate_advantage` with GAE)
- [ ] Run with:

  bash
  --gae_lambda <0, 0.95, 0.98, 0.99, 1>
  
- [ ] Plot all learning curves in one plot
- [ ] Describe:

  - [ ] How λ affects performance
  - [ ] Interpret λ=0 and λ=1 in terms of bias-variance tradeoff

---

#### Experiment 4: InvertedPendulum (Hyperparameter Tuning)

- [ ] Run default settings with 5 seeds:

  bash
  for seed in $(seq 1 5); do
    python ... --seed $seed
  done
  
- [ ] Tune hyperparameters to reach return of **1000** in **fewest env steps**
- [ ] Plot learning curves for:

  - [ ] Default settings
  - [ ] Tuned settings

---

#### Experiment 5: Humanoid (Extra Credit)

- [ ] Only run after verifying earlier experiments
- [ ] Run:

  bash
  python ... --env_name Humanoid-v4 ... --gae_lambda 0.97 --video_log_freq 5
  
- [ ] Watch for early convergence (≥300 return in 20 iterations)

---

### III Analysis Section

#### Problem 1: Analytical Policy Gradients

- [x] (a) Derive policy gradient ∇J(θ) manually
- [x] (b) Compute E\[R(τ)] directly and verify gradient matches (a)

#### Problem 2: Variance of Gradient

- [ ] Derive closed-form expression for variance of policy gradient
- [ ] Analyze θ values minimizing/maximizing variance

#### Problem 3: Return-to-Go

- [ ] (a) Show unbiasedness of modified gradient
- [ ] (b) Compute and **plot** variance of return-to-go estimator vs. original

#### Problem 4: Importance Sampling in Sparse-Reward MDP

- [ ] (a) Derive policy gradient with importance sampling
- [ ] (b) Derive and analyze variance as horizon H increases

---

### IV Final Submission

#### Report PDF

- [ ] Include:

  - [ ] Graphs + brief answers for Experiments 1–4
  - [ ] All command-line configurations
  - [ ] (Optional) Extra credit results
  - [ ] Full solutions to Section 8 (Analysis)

#### Code & Logs (submit.zip)

- [ ] `run_logs/` with all experiment folders (from `cs285/data/`)
- [ ] `cs285/` with original structure (exclude data/)
- [ ] `README.md` with instructions to reproduce results
- [ ] Use `zip -vr submit.zip submit -x "*.DS_Store"` to package



You can paste this directly into a `README.md` or any Markdown-supporting editor without formatting issues. Let me know if you'd like the list as a rendered HTML or downloadable `.md` file.

