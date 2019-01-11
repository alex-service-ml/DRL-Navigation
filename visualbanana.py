import numpy as np
import torch
import argparse
import os
import time
import platform

from unityagents import UnityEnvironment

from agent import VisualBananAgent, ReplayBuffer
from memory import Memory

action_strings = {
    0: '^ FORWARD ',
    1: 'v BACKWARD',
    2: '< LEFT    ',
    3: '> RIGHT   '
}

host_os_banana = {
    'linux': 'VisualBanana_Linux/Banana.x86_64',
    'mac': 'VisualBanana.app'
}


def grayscale(rgb):
    """
    Convert a 3-channel RGB image to grayscale
    :param rgb: ndarray of shape (x, y, 3)
    :return: ndarray of shape (x, y, 1)
    """
    g = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]  # From opencv's RGB2GRAY formula
    return g  # np.expand_dims(g, 2)  # Maintain (x, y, c) shape


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Checkpoint to load', default=None, required=False)
    parser.add_argument('--save-as', help='Checkpoint to save as', default='checkpoint.pth', required=False)
    parser.add_argument('--batch-size', type=int, help='Minibatch size', default=64, required=False)
    parser.add_argument('--memory', type=int, help='Size of the memory buffer', default=10000, required=False)
    parser.add_argument('--episodes', type=int, help='Number of episodes', default=1000, required=False)
    parser.add_argument('--gamma', type=float, help='Discount rate', default=0.99, required=False)
    parser.add_argument('--eps', type=float, help='Initial epsilon value', default=1.0, required=False)
    parser.add_argument('--eps-decay', type=float, help='Decay rate of epsilon', default=0.995, required=False)
    parser.add_argument('--eps-min', type=float, help='Minimum epsilon value', default=0.01, required=False)
    parser.add_argument('--seed', type=int, help='RNG Seed', required=False)
    parser.add_argument('--evaluate', action='store_true', help='Run trained model in eval mode', default=False)
    parser.add_argument('--per-b', type=float, help='Initial value of PER hyperparameter b', default=0.4)
    parser.add_argument('--slow', action='store_false', help='Run the game at normal speed', default=True)
    parser.add_argument('--no-render', action='store_true', help='Disable Unity rendering', default=False)
    parser.add_argument('--stack-frames', type=int, help='Number of frames to pass to the DQN', default=4)
    return parser.parse_args()


def save_scores(filename, scores):
    with open(filename + '_scores.txt', 'w') as f:  # Not appending. Just replace it 'cause whatever
        for score in scores:
            f.write("%i\n" % score)


if __name__ == '__main__':
    args = parse_arguments()

    # Setup RNG
    if args.seed:
        print('torch RNG using seed:', args.seed)
        seed = torch.manual_seed(args.seed)

    # Setup Environment
    env = UnityEnvironment(file_name=host_os_banana['linux'] if not platform.mac_ver()[0] else host_os_banana['mac'],
                           no_graphics=args.no_render)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Setup Agent and Experience Replay Buffer
    state_size = (args.stack_frames, 84, 84)  # TODO: Programmatically determine
    action_size = brain.vector_action_space_size
    memory = Memory(args.memory, args.batch_size, n=args.memory, b=args.per_b)
    agent = VisualBananAgent(state_size, action_size, memory=memory, checkpoint_filename=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None)
    print('Number of actions:', action_size)
    print('State Features: ', state_size)

    # Retrieve hyperparameters
    epsilon = args.eps
    epsilon_decay = args.eps_decay
    epsilon_mininum = args.eps_min
    per_b_increment = (1. - args.per_b) / args.episodes

    evaluate = args.evaluate
    if evaluate:
        print('Running in evaluation mode!')

    max_score = 0
    scores = 100 * [0]
    for i_episode in range(args.episodes):
        env_info = env.reset(train_mode=args.slow)[brain_name]  # reset the environment
        state = env_info.visual_observations[0]  # get the current state
        # Convert and Stack up initial state
        state_shape = np.squeeze(state).shape  # typically (84, 84, 3)
        state = grayscale(np.squeeze(state))  # (84, 84) NOTE: unlike the DQN paper, we don't take max over mult. frames
        # state = np.expand_dims(np.stack((state for _ in range(args.stack_frames)), 0), 0) # repeat the initial observed state, e.g. (84, 84, 4)
        state = np.stack((state for _ in range(args.stack_frames)), 0)
        # print('INITIAL STATE SHAPE:', state.shape)
        score = 0  # initialize the score

        # ======================
        """
        next_state = np.zeros((args.stack_frames, 84, 84))
        while True:
            # print('state shape:', state.shape)
            action = agent.act(state, epsilon, evaluate=evaluate)  # select an action

            # accumulate the next few frames
            # TODO: Move to its own function
            reward = 0
            
            done = False
            cumulative_reward = 0
            for i in range(args.stack_frames):
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state[i, :, :] = grayscale(np.squeeze(env_info.visual_observations[0]))  # get the next state
                reward = env_info.rewards[0]  # accumulate the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                cumulative_reward += reward
                if done and i < args.stack_frames - 1:
                    # End of episode and not enough frames to fill the buffer.
                    # TODO: See if appending zeros is just as good
                    # extra_frames = args.stack_frames - (i + 1)  # e.g. done on 3rd frame, 4 - (2 + 1) = 1 frame needed
                    break

        """
        next_state = state.copy()
        reward = 4 * [0]
        while True:
            # print('state shape:', state.shape)
            action = agent.act(state, epsilon, evaluate=evaluate)  # select an action

            # accumulate the next few frames
            # TODO: Move to its own function

            done = False
            cumulative_reward = 0
            # for i in range(args.stack_frames):
            env_info = env.step(action)[brain_name]  # send the action to the environment

            next_state = np.concatenate((next_state[1:, :, :], np.expand_dims(grayscale(np.squeeze(env_info.visual_observations[0])), 0)))
            # print(next_state.shape)
            # next_state[i, :, :] = grayscale(np.squeeze(env_info.visual_observations[0]))  # get the next state
            reward = reward[1:] + [env_info.rewards[0]]  # accumulate the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward[-1]  # update the score
            # cumulative_reward += reward
            # if done and i < args.stack_frames - 1:
                # End of episode and not enough frames to fill the buffer.
                # TODO: See if appending zeros is just as good
                # extra_frames = args.stack_frames - (i + 1)  # e.g. done on 3rd frame, 4 - (2 + 1) = 1 frame needed
            #    break

        # =====================

            # Show action
            if evaluate:
                print('\rScore: {}\tAction: {}'.format(score, action_strings[action]),  end="")
            # Learnin' time (sometimes)
            if not evaluate:
                agent.step(state, action, sum(reward), next_state, done)

            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        epsilon = max(epsilon_mininum, epsilon*epsilon_decay)
        memory.b = min(1., memory.b + per_b_increment)  # TODO: Integrate better with agent
        
        scores += [score]
        if np.mean(scores[-100:]) > max_score and not evaluate:
            max_score = np.mean(scores[-100:])
            print("\nNew max score! {:.1f}".format(max_score))
            torch.save(agent.qnetwork_local.state_dict(), args.save_as)
            save_scores(args.save_as[:-4], scores)
        print("\rEpisode {} Score: {:.1f} Avg: {:.2f} Eps: {:.2f}".format(i_episode, score, np.mean(scores[-100:]), epsilon)) #, end="")

        if evaluate:
            time.sleep(1)

    if not evaluate:
        save_scores(args.save_as[:-4], scores)
    else:
        save_scores(args.save_as[:-4] + '_eval', scores)
    env.close()
    print('Done!')

