# Humanoid Walking Training Log

2026-09-30 15:40:52 | training_session_started | target=humanoid_walking | total_timesteps=180000
2026-09-30 15:40:52 | environment=brax_humanoid_mjx | backend=cpu | target=walk_and_run
2026-09-30 15:40:52 | finding: cpu-only JAX is slower, so training is set to a moderate but real PPO curriculum
2026-09-30 15:40:52 | milestone: balance phase expected before walking and running
2026-09-30 15:40:52 | launching_ppo_training | num_timesteps=180000 num_envs=6 num_eval_envs=1
2026-09-30 15:41:18 | | steps: 0          | reward: 110.350      | avg_episode_length: 42.00    | milestone: start_balance
2026-09-30 15:43:05 | | steps: 60120      | reward: 349.807      | avg_episode_length: 138.00   | milestone: walking_stable
2026-09-30 15:44:51 | | steps: 120240     | reward: 458.917      | avg_episode_length: 176.00   | milestone: walking_stable
2026-09-30 15:46:28 | | steps: 180360     | reward: 517.514      | avg_episode_length: 200.00   | milestone: running_stride
2026-09-30 15:46:28 | training_finished | elapsed_seconds=335.9
2026-09-30 15:46:28 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl
2026-09-30 15:48:34 | sanity_check | validation_reward=99.71 over 20 steps
2026-09-30 15:48:34 | milestone: policy has been validated and saved for viewer playback
2026-10-01 08:29:15 | warm_start | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl | prior_session_steps=180360 | critic=restored
2026-10-01 08:29:15 | session_started | env=brax_humanoid (non-MJX, deprecated) backend=cpu | timesteps=30000 num_envs=16 evals=3
2026-10-01 08:29:15 | reward_shaping | forward_reward_weight=2.5 healthy_reward=1.0 ctrl_cost_weight=0.1 | rationale=alive bonus cut so locomotion pays more than standing still
2026-10-01 08:29:15 | tracking | fwd_speed=eval/episode_x_velocity divided by eval/avg_episode_length (m/s)
2026-10-01 08:29:38 | | steps: 0         | reward: 144.2     | ep_len: 199.5  | fwd_speed: 0.105   m/s | alive/step: 0.50  | dist: 82.04   | sps: nan     | stage: balance_holding
2026-10-01 08:30:14 | | steps: 15360     | reward: 136.7     | ep_len: 158.5  | fwd_speed: 0.161   m/s | alive/step: 0.50  | dist: 65.07   | sps: 705.8   | stage: balance_holding
2026-10-01 08:30:47 | | steps: 30720     | reward: 156.6     | ep_len: 161.0  | fwd_speed: 0.205   m/s | alive/step: 0.50  | dist: 67.05   | sps: 748.4   | stage: balance_holding
2026-10-01 08:30:47 | session_finished | elapsed_seconds=92.1 interrupted=False | session_steps=30000 cumulative_steps=210360
2026-10-01 08:30:47 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl
2026-10-01 08:39:27 | deterministic_rollout | ep_len=87 fwd_speed=0.429 m/s x_distance=37.30 reward=178.9 | stage=falling_early
2026-10-01 08:39:27 | verdict | falling_early at 0.429 m/s over 87 steps
2026-10-01 09:04:55 | warm_start | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl | recorded_prior_steps=0 | critic=restored
2026-10-01 09:04:55 | session_started | env=brax_humanoid (non-MJX, deprecated) backend=cpu | timesteps=20000 num_envs=16 evals=3
2026-10-01 09:04:55 | reward_shaping | forward_reward_weight=2.5 healthy_reward=1.0 ctrl_cost_weight=0.1 | rationale=alive bonus cut so locomotion pays more than standing still
2026-10-01 09:04:55 | tracking | fwd_speed = mean(eval/episode_x_velocity) in m/s; up_steps recovered from eval/episode_reward_alive because brax reports eval/avg_episode_length at 2x the real step count
2026-10-01 09:05:31 | | steps: 0         | reward: 144.2    | up_steps: 99.8   | fwd_speed: 0.210   m/s | x_travel: 0.31   m | x_pos: 10.92   | sps: nan     | stage: falling_early
2026-10-01 09:06:19 | | steps: 10240     | reward: 144.0    | up_steps: 87.0   | fwd_speed: 0.295   m/s | x_travel: 0.39   m | x_pos: 10.87   | sps: 377.2   | stage: falling_early
2026-10-01 09:07:04 | | steps: 20480     | reward: 171.2    | up_steps: 90.0   | fwd_speed: 0.392   m/s | x_travel: 0.53   m | x_pos: 14.14   | sps: 401.0   | stage: falling_early
2026-10-01 09:07:05 | session_finished | elapsed_seconds=130.2 interrupted=False | session_steps=20480 cumulative_steps=20480 | avg_sps=157.3
2026-10-01 09:07:05 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl cumulative_steps=20480
2026-10-01 09:19:24 | deterministic_rollout | up_steps=108 fwd_speed=0.192 m/s x_travel=0.31 m reward=158.4 | stage=falling_early
2026-10-01 09:19:24 | verdict | falling_early at 0.192 m/s over 108 upright steps
2026-10-01 09:20:26 | warm_start | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl | recorded_prior_steps=180360 | critic=restored
2026-10-01 09:20:26 | session_started | env=brax_humanoid (non-MJX, deprecated) backend=cpu | timesteps=5000000 num_envs=16 evals=20
2026-10-01 09:20:26 | reward_shaping | forward_reward_weight=2.5 healthy_reward=1.0 ctrl_cost_weight=0.1 | rationale=alive bonus cut so locomotion pays more than standing still
2026-10-01 09:20:26 | tracking | fwd_speed = mean(eval/episode_x_velocity) in m/s; up_steps recovered from eval/episode_reward_alive because brax reports eval/avg_episode_length at 2x the real step count
2026-10-01 09:21:06 | | steps: 0         | reward: 144.2    | up_steps: 99.8   | fwd_speed: 0.210   m/s | x_travel: 0.31   m | x_pos: 10.92   | sps: nan     | stage: falling_early
2026-10-01 09:26:55 | | steps: 263680    | reward: 172.6    | up_steps: 72.8   | fwd_speed: 0.578   m/s | x_travel: 0.63   m | x_pos: 16.91   | sps: 787.4   | stage: locomotion_emerging
2026-10-01 09:31:15 | | steps: 527360    | reward: 194.0    | up_steps: 69.5   | fwd_speed: 0.749   m/s | x_travel: 0.78   m | x_pos: 20.89   | sps: 1084.2  | stage: locomotion_emerging
2026-10-01 09:37:19 | | steps: 791040    | reward: 318.3    | up_steps: 96.2   | fwd_speed: 0.957   m/s | x_travel: 1.38   m | x_pos: 53.88   | sps: 765.4   | stage: locomotion_emerging
2026-10-01 09:43:13 | | steps: 1054720   | reward: 329.1    | up_steps: 88.8   | fwd_speed: 1.120   m/s | x_travel: 1.49   m | x_pos: 49.40   | sps: 754.9   | stage: locomotion_emerging
2026-10-01 09:49:05 | | steps: 1318400   | reward: 493.9    | up_steps: 115.2  | fwd_speed: 1.351   m/s | x_travel: 2.34   m | x_pos: 110.15  | sps: 760.8   | stage: locomotion_emerging
2026-10-01 09:55:04 | | steps: 1582080   | reward: 569.4    | up_steps: 114.8  | fwd_speed: 1.624   m/s | x_travel: 2.79   m | x_pos: 118.93  | sps: 745.7   | stage: locomotion_emerging
2026-10-01 10:00:52 | | steps: 1845760   | reward: 642.2    | up_steps: 143.2  | fwd_speed: 1.427   m/s | x_travel: 3.07   m | x_pos: 190.10  | sps: 769.5   | stage: locomotion_emerging
2026-10-01 10:06:16 | | steps: 2109440   | reward: 614.3    | up_steps: 127.0  | fwd_speed: 1.572   m/s | x_travel: 2.99   m | x_pos: 151.06  | sps: 827.3   | stage: locomotion_emerging
2026-10-01 10:11:35 | | steps: 2373120   | reward: 735.3    | up_steps: 139.0  | fwd_speed: 1.754   m/s | x_travel: 3.66   m | x_pos: 242.24  | sps: 837.9   | stage: locomotion_emerging
2026-10-01 10:16:56 | | steps: 2636800   | reward: 783.4    | up_steps: 142.8  | fwd_speed: 1.833   m/s | x_travel: 3.93   m | x_pos: 259.87  | sps: 832.9   | stage: locomotion_emerging
2026-10-01 10:22:57 | | steps: 2900480   | reward: 862.6    | up_steps: 156.2  | fwd_speed: 1.846   m/s | x_travel: 4.33   m | x_pos: 294.20  | sps: 741.9   | stage: locomotion_emerging
2026-10-01 10:28:31 | | steps: 3164160   | reward: 970.5    | up_steps: 159.2  | fwd_speed: 2.078   m/s | x_travel: 4.96   m | x_pos: 338.44  | sps: 801.0   | stage: locomotion_emerging
2026-10-01 10:34:48 | | steps: 3427840   | reward: 1149.0   | up_steps: 177.5  | fwd_speed: 2.230   m/s | x_travel: 5.94   m | x_pos: 502.91  | sps: 710.1   | stage: locomotion_emerging
2026-10-01 10:41:01 | | steps: 3691520   | reward: 1376.8   | up_steps: 198.0  | fwd_speed: 2.423   m/s | x_travel: 7.19   m | x_pos: 650.18  | sps: 717.4   | stage: locomotion_emerging
2026-10-01 10:47:16 | | steps: 3955200   | reward: 2158.6   | up_steps: 289.2  | fwd_speed: 2.627   m/s | x_travel: 11.40  m | x_pos: 1566.08 | sps: 714.2   | stage: locomotion_emerging
2026-10-01 10:53:31 | | steps: 4218880   | reward: 1340.8   | up_steps: 184.0  | fwd_speed: 2.559   m/s | x_travel: 7.06   m | x_pos: 739.43  | sps: 713.2   | stage: locomotion_emerging
2026-10-01 10:59:42 | | steps: 4482560   | reward: 2501.9   | up_steps: 299.5  | fwd_speed: 2.985   m/s | x_travel: 13.41  m | x_pos: 2208.95 | sps: 721.6   | stage: locomotion_emerging
2026-10-01 11:05:52 | | steps: 4746240   | reward: 2612.9   | up_steps: 303.0  | fwd_speed: 3.095   m/s | x_travel: 14.07  m | x_pos: 2178.12 | sps: 722.5   | stage: locomotion_emerging
2026-10-01 11:12:02 | | steps: 5009920   | reward: 1872.0   | up_steps: 229.2  | fwd_speed: 2.911   m/s | x_travel: 10.01  m | x_pos: 1377.40 | sps: 723.1   | stage: locomotion_emerging
2026-10-01 11:12:03 | session_finished | elapsed_seconds=6696.3 interrupted=False | session_steps=5009920 cumulative_steps=5190280 | avg_sps=748.2
2026-10-01 11:12:03 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl cumulative_steps=5190280
2026-10-01 11:38:11 | deterministic_rollout | up_steps=214 fwd_speed=2.832 m/s x_travel=9.09 m reward=1708.5 | stage=locomotion_emerging
2026-10-01 11:38:11 | verdict | locomotion_emerging at 2.832 m/s over 214 upright steps
2026-10-01 13:17:11 | warm_start | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl | recorded_prior_steps=5190280 | critic=restored
2026-10-01 13:17:11 | session_started | env=brax_humanoid (non-MJX, deprecated) backend=cpu | timesteps=5000000 num_envs=16 evals=20
2026-10-01 13:17:11 | reward_shaping | forward_reward_weight=2.5 healthy_reward=1.0 ctrl_cost_weight=0.1 | rationale=alive bonus cut so locomotion pays more than standing still
2026-10-01 13:17:11 | optimizer | learning_rate=0.00025 schedule=adaptive_kl desired_kl=0.01 | resume_from_step=5190280
2026-10-01 13:17:11 | tracking | fwd_speed = mean(eval/episode_x_velocity) in m/s; up_steps recovered from eval/episode_reward_alive because brax reports eval/avg_episode_length at 2x the real step count
2026-10-01 13:17:36 | | steps: 0         | reward: 2237.3   | up_steps: 265.0  | fwd_speed: 3.022   m/s | x_travel: 12.01  m | x_pos: 2019.60 | sps: nan     | stage: locomotion_emerging
2026-10-01 13:22:35 | | steps: 263680    | reward: 2585.5   | up_steps: 301.0  | fwd_speed: 3.079   m/s | x_travel: 13.90  m | x_pos: 1877.56 | sps: 931.8   | stage: locomotion_emerging
2026-10-01 13:27:47 | | steps: 527360    | reward: 4255.3   | up_steps: 452.5  | fwd_speed: 3.407   m/s | x_travel: 23.12  m | x_pos: 4747.85 | sps: 902.8   | stage: locomotion_emerging
2026-10-01 13:34:39 | | steps: 791040    | reward: 4362.9   | up_steps: 458.2  | fwd_speed: 3.453   m/s | x_travel: 23.74  m | x_pos: 4961.49 | sps: 682.3   | stage: locomotion_emerging
2026-10-01 13:41:22 | | steps: 1054720   | reward: 4486.3   | up_steps: 465.5  | fwd_speed: 3.500   m/s | x_travel: 24.44  m | x_pos: 5176.77 | sps: 664.1   | stage: locomotion_emerging
2026-10-01 13:48:04 | | steps: 1318400   | reward: 4382.8   | up_steps: 450.0  | fwd_speed: 3.542   m/s | x_travel: 23.91  m | x_pos: 5052.58 | sps: 666.2   | stage: locomotion_emerging
2026-10-01 13:54:59 | | steps: 1582080   | reward: 2397.9   | up_steps: 266.2  | fwd_speed: 3.248   m/s | x_travel: 12.97  m | x_pos: 2034.53 | sps: 643.7   | stage: locomotion_emerging
2026-10-01 14:01:44 | | steps: 1845760   | reward: 2956.1   | up_steps: 309.0  | fwd_speed: 3.473   m/s | x_travel: 16.10  m | x_pos: 3243.27 | sps: 661.2   | stage: locomotion_emerging
2026-10-01 14:08:39 | | steps: 2109440   | reward: 3111.9   | up_steps: 324.0  | fwd_speed: 3.489   m/s | x_travel: 16.96  m | x_pos: 3338.43 | sps: 645.7   | stage: locomotion_emerging
2026-10-01 14:15:38 | | steps: 2373120   | reward: 4437.6   | up_steps: 444.2  | fwd_speed: 3.643   m/s | x_travel: 24.28  m | x_pos: 5080.16 | sps: 638.9   | stage: locomotion_emerging
2026-10-01 14:22:33 | | steps: 2636800   | reward: 5168.2   | up_steps: 500.0  | fwd_speed: 3.781   m/s | x_travel: 28.36  m | x_pos: 6375.62 | sps: 644.9   | stage: locomotion_emerging
2026-10-01 14:29:27 | | steps: 2900480   | reward: 2642.7   | up_steps: 271.5  | fwd_speed: 3.541   m/s | x_travel: 14.42  m | x_pos: 3178.03 | sps: 645.3   | stage: locomotion_emerging
2026-10-01 14:36:20 | | steps: 3164160   | reward: 4167.6   | up_steps: 414.0  | fwd_speed: 3.674   m/s | x_travel: 22.82  m | x_pos: 4858.39 | sps: 649.1   | stage: locomotion_emerging
2026-10-01 14:43:12 | | steps: 3427840   | reward: 5237.7   | up_steps: 500.0  | fwd_speed: 3.838   m/s | x_travel: 28.78  m | x_pos: 6495.79 | sps: 649.6   | stage: locomotion_emerging
2026-10-01 14:50:06 | | steps: 3691520   | reward: 4301.3   | up_steps: 418.8  | fwd_speed: 3.757   m/s | x_travel: 23.60  m | x_pos: 4584.07 | sps: 646.0   | stage: locomotion_emerging
2026-10-01 14:56:01 | | steps: 3955200   | reward: 4240.0   | up_steps: 410.0  | fwd_speed: 3.785   m/s | x_travel: 23.28  m | x_pos: 4998.37 | sps: 753.4   | stage: locomotion_emerging
2026-10-01 15:01:48 | | steps: 4218880   | reward: 4019.9   | up_steps: 392.0  | fwd_speed: 3.750   m/s | x_travel: 22.05  m | x_pos: 4868.80 | sps: 771.9   | stage: locomotion_emerging
2026-10-01 15:06:34 | | steps: 4482560   | reward: 4502.5   | up_steps: 437.2  | fwd_speed: 3.767   m/s | x_travel: 24.71  m | x_pos: 5166.44 | sps: 938.4   | stage: locomotion_emerging
2026-10-01 15:11:22 | | steps: 4746240   | reward: 5306.9   | up_steps: 500.0  | fwd_speed: 3.895   m/s | x_travel: 29.21  m | x_pos: 6569.69 | sps: 928.4   | stage: locomotion_emerging
2026-10-01 15:15:28 | | steps: 5009920   | reward: 4970.5   | up_steps: 473.0  | fwd_speed: 3.852   m/s | x_travel: 27.33  m | x_pos: 5834.87 | sps: 1085.1  | stage: locomotion_emerging
2026-10-01 15:15:28 | session_finished | elapsed_seconds=7097.1 interrupted=False | session_steps=5009920 cumulative_steps=10200200 | avg_sps=705.9
2026-10-01 15:15:28 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl cumulative_steps=10200200
2026-10-01 18:48:45 | warm_start | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl | recorded_prior_steps=10200200 | critic=restored
2026-10-01 18:48:45 | session_started | env=brax_humanoid (non-MJX, deprecated) backend=cpu | timesteps=5000000 num_envs=16 evals=20
2026-10-01 18:48:45 | reward_shaping | forward_reward_weight=2.5 healthy_reward=1.0 ctrl_cost_weight=0.1 | rationale=alive bonus cut so locomotion pays more than standing still
2026-10-01 18:48:45 | optimizer | learning_rate=0.00025 schedule=adaptive_kl desired_kl=0.01 entropy_cost=0.0001 | resume_from_step=10200200
2026-10-01 18:48:45 | tracking | fwd_speed = mean(eval/episode_x_velocity) in m/s; up_steps recovered from eval/episode_reward_alive because brax reports eval/avg_episode_length at 2x the real step count
2026-10-01 18:49:07 | | steps: 0         | reward: 5352.2   | up_steps: 500.0  | fwd_speed: 3.931   m/s | x_travel: 29.48  m | x_pos: 6612.56 | sps: nan     | stage: locomotion_emerging
2026-10-01 18:53:19 | | steps: 263680    | reward: 4101.5   | up_steps: 396.5  | fwd_speed: 3.787   m/s | x_travel: 22.52  m | x_pos: 4392.93 | sps: 1108.8  | stage: locomotion_emerging
2026-10-01 18:57:39 | | steps: 527360    | reward: 5338.9   | up_steps: 500.0  | fwd_speed: 3.921   m/s | x_travel: 29.41  m | x_pos: 6599.85 | sps: 1064.4  | stage: locomotion_emerging
2026-10-01 19:01:56 | | steps: 791040    | reward: 5372.9   | up_steps: 500.0  | fwd_speed: 3.948   m/s | x_travel: 29.61  m | x_pos: 6650.87 | sps: 1084.1  | stage: locomotion_emerging
2026-10-01 19:06:02 | | steps: 1054720   | reward: 5359.8   | up_steps: 500.0  | fwd_speed: 3.938   m/s | x_travel: 29.53  m | x_pos: 6628.16 | sps: 1088.0  | stage: locomotion_emerging
2026-10-01 19:10:09 | | steps: 1318400   | reward: 4935.7   | up_steps: 463.5  | fwd_speed: 3.909   m/s | x_travel: 27.18  m | x_pos: 5717.32 | sps: 1081.9  | stage: locomotion_emerging
2026-10-01 19:14:15 | | steps: 1582080   | reward: 3637.7   | up_steps: 352.0  | fwd_speed: 3.783   m/s | x_travel: 19.98  m | x_pos: 4063.30 | sps: 1089.8  | stage: locomotion_emerging
2026-10-01 19:18:22 | | steps: 1845760   | reward: 5430.3   | up_steps: 500.0  | fwd_speed: 3.994   m/s | x_travel: 29.96  m | x_pos: 6695.16 | sps: 1082.8  | stage: locomotion_emerging
2026-10-01 19:22:27 | | steps: 2109440   | reward: 4091.8   | up_steps: 385.2  | fwd_speed: 3.899   m/s | x_travel: 22.53  m | x_pos: 5069.46 | sps: 1091.3  | stage: locomotion_emerging
2026-10-01 19:26:32 | | steps: 2373120   | reward: 5103.4   | up_steps: 479.2  | fwd_speed: 3.910   m/s | x_travel: 28.11  m | x_pos: 6100.31 | sps: 1092.6  | stage: locomotion_emerging
2026-10-01 19:30:36 | | steps: 2636800   | reward: 3738.8   | up_steps: 360.5  | fwd_speed: 3.799   m/s | x_travel: 20.54  m | x_pos: 3979.48 | sps: 1096.5  | stage: locomotion_emerging
2026-10-01 19:34:41 | | steps: 2900480   | reward: 5375.1   | up_steps: 500.0  | fwd_speed: 3.950   m/s | x_travel: 29.63  m | x_pos: 6648.34 | sps: 1095.0  | stage: locomotion_emerging
2026-10-01 19:38:47 | | steps: 3164160   | reward: 4253.9   | up_steps: 403.2  | fwd_speed: 3.870   m/s | x_travel: 23.41  m | x_pos: 5081.27 | sps: 1087.0  | stage: locomotion_emerging
2026-10-01 19:42:53 | | steps: 3427840   | reward: 4289.3   | up_steps: 407.2  | fwd_speed: 3.863   m/s | x_travel: 23.60  m | x_pos: 5091.85 | sps: 1090.1  | stage: locomotion_emerging
2026-10-01 19:47:00 | | steps: 3691520   | reward: 3471.6   | up_steps: 341.2  | fwd_speed: 3.719   m/s | x_travel: 19.03  m | x_pos: 3635.94 | sps: 1085.2  | stage: locomotion_emerging
2026-10-01 19:51:24 | | steps: 3955200   | reward: 2215.8   | up_steps: 228.8  | fwd_speed: 3.525   m/s | x_travel: 12.10  m | x_pos: 2067.21 | sps: 1012.2  | stage: locomotion_emerging
2026-10-01 19:55:33 | | steps: 4218880   | reward: 5371.7   | up_steps: 500.0  | fwd_speed: 3.948   m/s | x_travel: 29.61  m | x_pos: 6630.13 | sps: 1071.6  | stage: locomotion_emerging
2026-10-01 19:59:39 | | steps: 4482560   | reward: 4705.4   | up_steps: 447.0  | fwd_speed: 3.861   m/s | x_travel: 25.89  m | x_pos: 5367.71 | sps: 1092.3  | stage: locomotion_emerging
2026-10-01 20:03:44 | | steps: 4746240   | reward: 4139.3   | up_steps: 393.2  | fwd_speed: 3.861   m/s | x_travel: 22.78  m | x_pos: 4757.77 | sps: 1090.1  | stage: locomotion_emerging
2026-10-01 20:07:49 | | steps: 5009920   | reward: 3431.7   | up_steps: 343.2  | fwd_speed: 3.649   m/s | x_travel: 18.79  m | x_pos: 3591.37 | sps: 1089.2  | stage: locomotion_emerging
2026-10-01 20:07:50 | session_finished | elapsed_seconds=4744.9 interrupted=False | session_steps=5009920 cumulative_steps=15210120 | avg_sps=1055.8
2026-10-01 20:07:50 | checkpoint_saved | path=C:\Users\ayush\projects\train_walking\checkpoints\humanoid_latest.pkl cumulative_steps=15210120
2026-10-01 20:07:57 | deterministic_rollout | up_steps=999 fwd_speed=4.237 m/s x_travel=63.49 m reward=11460.5 | stage=gait_sustained
2026-10-01 20:07:57 | verdict | gait_sustained at 4.237 m/s over 999 upright steps
