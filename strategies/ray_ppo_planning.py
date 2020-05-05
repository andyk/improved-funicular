import random
import numpy as np
import mlflow
import ray
from pprint import pprint
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.rollout_worker import _validate_env
from strategies.utility import do_trials
from envs.nick_gym_adapter import Nick2048Gym



class PPOTrainerWithReset(PPOTrainer):
   def reset_config(self, new_config):
       assert "env_config" in new_config
       def update_worker_conf(w):
           context = EnvContext(new_config["env_config"], w.worker_index)
           env = w.env_creator(context)
           w.env = _validate_env(env)
       self.workers.foreach_worker(update_worker_conf)
       return True

def get_strategy_function(cls, ray_agent):
    def strategy_fn(board):
        print(np.asarray(board).reshape((4, 4)))
        conf = ray_agent.config
        conf["env_config"] = {"init_board": board}
        print(f"reset config returned {ray_agent.reset_config(conf)}")
        print(pretty_print(ray_agent.train()))
        action = ray_agent.compute_action(board)
        tmp_env = Nick2048Gym()
        a_s = tmp_env.get_afterstate(tuple(board), action)[0]
        if np.array_equal(np.asarray(a_s), board):
            new_action = cls.action_space.sample()
            print(f"picked a random action {new_action} to avoid duplicate action {action}")
            while new_action == action:
                new_action = cls.action_space.sample()
                print(f"picked a random action {new_action} to avoid duplicate action {action}")
            action = new_action
        print(f"taking action {action}")
        return action
    return strategy_fn


def try_ray_ppo_planning(cls, trial_count):
    with mlflow.start_run():
        ray.init()
        config = {
            "env": cls,
            "num_workers": 3,
            "num_gpus": 0,
            "horizon": 5,
            "train_batch_size": 12000,  # val of 128 leads to ~1s per training iteration.
        }
        full_config = DEFAULT_CONFIG.copy()
        for k, v in config.items():
            full_config[k] = v
        pprint(full_config)
        agent = PPOTrainerWithReset(full_config)
        strategy_fn = get_strategy_function(cls, agent)
        strategy_fn.info = "Ray PPO Planning strategy"

        trial_result = do_trials(cls, trial_count, strategy_fn, max_steps_per_episode=10000, always_print=True)
        checkpoint = agent.save()
        print(f"checkpoint saved at {checkpoint}")
        mlflow.log_metrics(trial_result)
