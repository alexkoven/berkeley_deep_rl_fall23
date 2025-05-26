#!/usr/bin/env python3

import os
import subprocess
import argparse
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

def run_experiment(exp_type, video_log_freq=None):
    """
    Run CartPole experiment with specified configuration.
    
    Args:
        exp_type: One of ['sb_no_rtg_no_na', 'sb_rtg_no_na', 'sb_no_rtg_na', 'sb_rtg_na',
                         'lb_no_rtg_no_na', 'lb_rtg_no_na', 'lb_no_rtg_na', 'lb_rtg_na']
        video_log_freq: Optional[int], frequency to log videos
    """
    # Base command
    base_cmd = [
        "python", "cs285/scripts/run_hw2.py",
        "--env_name", "CartPole-v0",
        "-n", "100"
    ]
    
    # Configuration mapping
    configs = {
        # Small batch (1000) configurations
        'sb_no_rtg_no_na': {
            'batch_size': 1000,
            'rtg': False,
            'na': False,
            'exp_name': 'cartpole_sb_no_rtg_no_na'
        },
        'sb_rtg_no_na': {
            'batch_size': 1000,
            'rtg': True,
            'na': False,
            'exp_name': 'cartpole_sb_rtg_no_na'
        },
        'sb_no_rtg_na': {
            'batch_size': 1000,
            'rtg': False,
            'na': True,
            'exp_name': 'cartpole_sb_no_rtg_na'
        },
        'sb_rtg_na': {
            'batch_size': 1000,
            'rtg': True,
            'na': True,
            'exp_name': 'cartpole_sb_rtg_na'
        },
        # Large batch (4000) configurations
        'lb_no_rtg_no_na': {
            'batch_size': 4000,
            'rtg': False,
            'na': False,
            'exp_name': 'cartpole_lb_no_rtg_no_na'
        },
        'lb_rtg_no_na': {
            'batch_size': 4000,
            'rtg': True,
            'na': False,
            'exp_name': 'cartpole_lb_rtg_no_na'
        },
        'lb_no_rtg_na': {
            'batch_size': 4000,
            'rtg': False,
            'na': True,
            'exp_name': 'cartpole_lb_no_rtg_na'
        },
        'lb_rtg_na': {
            'batch_size': 4000,
            'rtg': True,
            'na': True,
            'exp_name': 'cartpole_lb_rtg_na'
        }
    }
    
    if exp_type not in configs:
        raise ValueError(f"Invalid experiment type. Choose from: {list(configs.keys())}")
    
    # Get configuration
    config = configs[exp_type]
    
    # Build command
    cmd = base_cmd + [
        "-b", str(config['batch_size']),
        "--exp_name", config['exp_name']
    ]
    
    if config['rtg']:
        cmd.append("-rtg")
    if config['na']:
        cmd.append("-na")
    if video_log_freq is not None:
        cmd += ["--video_log_freq", str(video_log_freq)]
    
    # Print and run command
    print(f"Running experiment: {exp_type}")
    print(f"Command: {' '.join(cmd)}")
    
    # Set PYTHONPATH environment variable for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    subprocess.run(cmd, env=env)

def main():
    parser = argparse.ArgumentParser(description='Run CartPole experiments')
    parser.add_argument('--exp_type', type=str, required=True,
                      help='Experiment type to run. Options: ' + 
                           'sb_no_rtg_no_na, sb_rtg_no_na, sb_no_rtg_na, sb_rtg_na, ' +
                           'lb_no_rtg_no_na, lb_rtg_no_na, lb_no_rtg_na, lb_rtg_na')
    parser.add_argument('--video_log_freq', type=int, default=None,
                      help='Frequency (in iterations) to log videos. If not set, no videos are logged.')
    args = parser.parse_args()
    run_experiment(args.exp_type, video_log_freq=args.video_log_freq)

if __name__ == "__main__":
    main() 