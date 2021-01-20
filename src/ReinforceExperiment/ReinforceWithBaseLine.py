import numpy as np
import sys
import tensorflow as tf
sys.path.append("../")
from ReinforceExperiment.ReinforceEnvironment import ReinforceEvironment as environment
from ReinforceExperiment.ReinforceWithBaseLineNetwork import Network

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    # reinforce + baseline parameters
    parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=20_000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--maximal_action_space", default=10, type=int, help="maximum number of proposals per step")

    # environment and mask-rcnn parameters
    parser.add_argument("--detector_model", default="logs_detector/detector01/mask_rcnn_detector_of_geom_primitives_0140.h5", type=str,
                        help="path for trained detector!")
    parser.add_argument("--hint", default=0, type=int,
                        help="Get hint which tool should be used each time. 1 use hint, 0 do not")
    parser.add_argument("--additional_moves", default=2, type=int,
                        help="How much more moves evaluation gets on top of minimal construction length.")
    parser.add_argument("--history_size", default=1, type=int,
                        help="history size")
    parser.add_argument("--generate_levels", default="02.*01", type=str,
                        help="regex that matches lvl names")

    args = parser.parse_args()

    env = environment(args)
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = {"image_and_mask": [], "class_ids": [], "tool_mask": []}, {"tool": [], "click_one": [], "click_two": [], "click_three": []}, []
        for _ in range(args.batch_size):
            # Perform episode
            states = {"image_and_mask": [], "class_ids": [], "tool_mask": []}
            actions = {"tool": [], "click_one": [], "click_two": [], "click_three": []}
            rewards = []
            state, done = env.reset(), False
            while not done:

                # probabilities for each action
                pred = network.predict(state)
                tool_probabilities = pred[0][0]
                click_one_probabilities = pred[1][0]
                click_two_probabilities = pred[2][0]
                click_three_probabilities = pred[3][0]

                # choose uniformly based on probabilities
                action_tool = np.random.choice(len(tool_probabilities), p=tool_probabilities)
                action_click_one = np.random.choice(len(click_one_probabilities), p=click_one_probabilities)
                action_click_two = np.random.choice(len(click_two_probabilities), p=click_two_probabilities)
                action_click_three = np.random.choice(len(click_three_probabilities), p=click_three_probabilities)

                next_state, reward, done = env.step(action_tool, [action_click_one, action_click_two, action_click_three])

                for k in state.keys():
                    states[k].append(state[k][0])

                actions["tool"].append(action_tool)
                actions["click_one"].append(action_click_one)
                actions["click_two"].append(action_click_two)
                actions["click_three"].append(action_click_three)
                rewards.append(reward)

                state = next_state

            # Compute returns by summing rewards (with discounting)
            sum = 0
            returns = [0] * len(rewards)
            for i in range(len(rewards)):
                sum = args.gamma * sum + rewards[-i - 1]
                returns[-i - 1] = sum

            for k in batch_states.keys():
                batch_states[k] += states[k]
            for k in batch_actions.keys():
                batch_actions[k] += actions[k]
            batch_returns += returns

        # Train using the generated batch
        network.train(batch_states, batch_actions, batch_returns)
