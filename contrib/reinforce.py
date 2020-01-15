# RL Agent that plays 2048 using REINFORCE. Coded by Andy.
from envs.andy_2048 import BoardEnv
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import mlflow
import numpy as np
from copy import deepcopy

params = {"board_width": 4,
          "num_iters": 2,
          "num_episodes_per_iter": 10,
          "max_steps_per_episode": 500,
          "discount_rate": 0.95,
          "epsilon_base": 0.2,
          "learning_rate": 0.01
          }

with mlflow.start_run() as run:
    mlflow.log_params(params)

    # model takes a board state (a 5x5 array of ints) and ouputs an action in [0,1,2,3]
    model = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(4, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(lr=params["learning_rate"])
    loss_fn = keras.losses.CategoricalCrossentropy()

    b = BoardEnv()
    all_actions = []
    game_scores = []
    max_tile = 0
    for iter_num in range(params["num_iters"]):
        rewards_lists = []
        grads_lists = []
        ep_prob = min(0.7, (params["epsilon_base"] + 100) / (iter_num + 1))
        if iter_num % 20 == 0:
            print("epsilon prob: %s" % ep_prob)
        mlflow.log_metric("epsilon_adjusted", ep_prob, step=iter_num)
        done = False
        for episode_num in range(params["num_episodes_per_iter"]):
            state = b.reset()
            rewards_lists.append([])
            grads_lists.append([])
            max_tile_this_ep = 0
            for step_num in range(params["max_steps_per_episode"]):
                with tf.GradientTape() as tape:
                    action_probs = tf.squeeze(model(state[np.newaxis]), axis=0)
                    if np.random.random() > ep_prob:
                        dice_roll = tfp.distributions.Multinomial(
                            total_count=1, probs=action_probs
                        ).sample(1)
                    else:
                        dice_roll = tf.one_hot(np.random.randint(4), 4)
                    loss = loss_fn(dice_roll, action_probs)
                grads = tape.gradient(loss, model.trainable_variables)
                grads_lists[-1].append(grads)
                action = b.action_space[np.argmax(dice_roll)]
                all_actions.append(action)
                # print(action_probs, dice_roll, action)
                new_state, reward, done, _ = b.step(action)
                # if np.array_equal(new_state, state):  # don't keep trying dud moves
                #     break
                state = new_state

                rewards_lists[-1].append(reward)
                mlflow.log_metric(
                    "rewards in iter %s episode %s" % (iter_num, episode_num),
                    reward,
                    step=step_num,
                )
                max_tile_this_ep = max(max_tile_this_ep, reward)
                if done:
                    break
                max_tile = max(max_tile, reward)
            game_scores.append(np.sum(rewards_lists[-1]))
            mlflow.log_metric("game scores in iter %s" % (iter_num), game_scores[-1])
            print("ep num %s: %s points" % (episode_num, game_scores[-1]))
        print(game_scores[-episode_num - 1:])
        print(
            "%.1f avg score, %s max points per turn, in game iter %s, episode %s"
            % (
                np.array(game_scores[-episode_num - 1:]).mean(),
                max_tile_this_ep,
                iter_num,
                episode_num,
            )
        )
        mlflow.log_metric(
            "avg game score per_iter",
            np.array(game_scores[-episode_num - 1:]).mean(),
            step=iter_num,
        )
        mlflow.log_metric(
            "max single_turn points per_iter", max_tile_this_ep, step=iter_num
        )

        # Update the policy based on these rollouts
        # rewards_lists is a list of lists, one list per episode
        discounted_rewards = deepcopy(rewards_lists)
        for ep_num, rew_list in enumerate(rewards_lists):
            for rew_num in range(len(rew_list) - 2, -1, -1):
                discounted_rewards[ep_num][rew_num] += (
                    params["discount_rate"] * discounted_rewards[ep_num][rew_num + 1]
                )

        # Standardize the discounted rewards
        reward_list_np = np.array(rewards_lists)
        ragged_rewards = tf.ragged.constant(rewards_lists, dtype=tf.float32)
        rewards_mean = tf.reduce_mean(ragged_rewards)
        rewards_std = tf.math.sqrt(
            tf.reduce_mean(tf.square(ragged_rewards - rewards_mean))
        )
        # print(rewards_mean, rewards_std, tf.reduce_max(ragged_rewards))
        standardized_rewards = (ragged_rewards - rewards_mean) / rewards_std

        # for each step, trainable_variable: calculate expectection of reward * grads and use
        # the optimizer to apply them to take a gradient ascent step in the model parameters.
        weighted_grads = []
        for var_num in range(len(model.trainable_variables)):
            per_var_grads = [
                grads_lists[ep_num][st_num][var_num]
                for ep_num, ep_rewards in enumerate(standardized_rewards)
                for st_num, step_reward in enumerate(ep_rewards)
            ]
            weighted_grads.append(tf.reduce_mean(per_var_grads, axis=0))
        optimizer.apply_gradients(zip(weighted_grads, model.trainable_variables))

    if game_scores:
        print(game_scores)
        ten_pct = int(0.1 * len(game_scores))
        first_pt = np.array(game_scores[:ten_pct])
        last_pt = np.array(game_scores[-ten_pct:])
        print(
            "mean reward (first %s, last %s): %.1f, %.1f"
            % (ten_pct, ten_pct, first_pt.mean(), last_pt.mean())
        )
        print("std reward: %.1f, %.1f" % (first_pt.std(), last_pt.std()))
        print("max points in a single turn: %s" % max_tile)

        # Plot game_scores over time
        import matplotlib.pyplot as plt
        from scipy.ndimage.filters import gaussian_filter1d

        plt.rcParams["figure.figsize"] = [10, 5]

        plt.plot(game_scores)
        plt.savefig("raw_game_scores.png")
        mlflow.log_artifact("raw_game_scores.png")

        ysmoothed = gaussian_filter1d(game_scores, sigma=4)
        plt.plot(ysmoothed)
        plt.savefig("smoothed_game_scores.png")
        mlflow.log_artifact("smoothed_game_scores.png")
        plt.show()
