# BC
# python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1


# cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --num_agent_train_steps_per_iter 1000 --eval_batch_size 5000 --ep_len 1000 --video_log_freq 1

# Dagger
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--num_agent_train_steps_per_iter 1000 --eval_batch_size 5000 --ep_len 1000 \
--video_log_freq -1


# tensorboard --logdir data