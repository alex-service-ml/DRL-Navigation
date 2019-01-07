import numpy as np
import torch
import argparse
import os
import time

from unityagents import UnityEnvironment

from agent import BananAgent, ReplayBuffer

action_strings = {
    0: '^ FORWARD ',
    1: 'v BACKWARD',
    2: '< LEFT    ',
    3: '> RIGHT   '
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Checkpoint to load', default=None, required=False)
    parser.add_argument('--save-as', help='Checkpoint to save as', default='checkpoint.pth', required=False)
    parser.add_argument('--batch-size', type=int, help='Minibatch size', default=64, required=False)
    parser.add_argument('--memory', type=int, help='Size of the memory buffer', default=10000, required=False)
    parser.add_argument('--episodes', type=int, help='Number of episodes', default=10, required=False)
    parser.add_argument('--gamma', type=float, help='Discount rate', default=0.99, required=False)
    parser.add_argument('--eps', type=float, help='Initial epsilon value', default=1.0, required=False)
    parser.add_argument('--eps-decay', type=float, help='Decay rate of epsilon', default=0.995, required=False)
    parser.add_argument('--eps-min', type=float, help='Minimum epsilon value', default=0.01, required=False)
    parser.add_argument('--seed', type=int, help='RNG Seed', required=False)
    parser.add_argument('--evaluate', action='store_true', help='Run trained model in eval mode', default=False)
    parser.add_argument('--slow', action='store_false', help='Run the game at normal speed', default=True)
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
    env = UnityEnvironment(file_name="Banana.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Setup Agent and Experience Replay Buffer
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    memory = ReplayBuffer(action_size, args.memory, args.batch_size)
    agent = BananAgent(state_size, action_size, memory=memory, checkpoint_filename=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None)
    print('Number of actions:', action_size)
    print('State Features: ', state_size)

    # Retrieve hyperparameters
    epsilon = args.eps
    epsilon_decay = args.eps_decay
    epsilon_mininum = args.eps_min
    evaluate = args.evaluate
    if evaluate:
        print('Running in evaluation mode!')

    max_score = 0
    scores = []
    for i_episode in range(args.episodes):
        env_info = env.reset(train_mode=args.slow)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score

        while True:

            action = agent.act(state, epsilon, evaluate=evaluate)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score

            # Show action
            print('\rScore: {}\tAction: {}'.format(score, action_strings[action]),  end="")
            # Learnin' time (sometimes)
            if not evaluate:
                agent.step(state, action, reward, next_state, done)

            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        epsilon = max(epsilon_mininum, epsilon*epsilon_decay)

        scores += [score]
        if np.mean(scores[-100:]) > max_score and not evaluate:
            max_score = np.mean(scores[-100:])
            print("\nNew max score! {:.1f}".format(max_score))
            torch.save(agent.qnetwork_local.state_dict(), args.save_as)
            save_scores(args.save_as[:-4], scores)
        print("\rEpisode {} Score: {:.1f} Avg: {:.2f} Eps: {:.2f}".format(i_episode, score, np.mean(scores[-100:]), epsilon)) #, end="")
        time.sleep(1)
    if not evaluate:
        save_scores(args.save_as[:-4], scores)
    else:
        save_scores(args.save_as[:-4] + '_eval', scores)
    env.close()
    print('Done!')

