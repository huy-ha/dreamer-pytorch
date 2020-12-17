import pickle
import numpy as np


def compute_mean_episode_rewards(experience):
    episode_rewards = [0.0]
    for done, reward in zip(experience['done'], experience['reward']):
        episode_rewards[-1] += reward
        if done:
            episode_rewards.append(0.0)
    print('Mean Episode Rewards:', np.mean(episode_rewards))


def compute_mutual_info(experience):

    latent_states = [
        rssm_state.prev_state.stoch
        for rssm_state in experience['agent_infos']]
    ground_truth_states = [
        step_info.internal_state
        for step_info in experience['info']
    ]
    obs = experience['obs']

    # Note: since latent states is prev state, not curr state
    # you might want to offset the two arrays by one
    print(len(latent_states), len(ground_truth_states), len(obs))
    # @Will TODO


if __name__ == "__main__":
    experience = pickle.load(open('experience.pkl', 'rb'))
    compute_mean_episode_rewards(experience)
    compute_mutual_info(experience)
