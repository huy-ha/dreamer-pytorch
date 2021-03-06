import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
import os
from tqdm import tqdm


def compute_mean_episode_rewards(experience):
    episode_rewards = [0.0]
    for done, reward in zip(experience['done'], experience['reward']):
        episode_rewards[-1] += reward
        if done:
            episode_rewards.append(0.0)
    print('Mean Episode Rewards:', np.mean(episode_rewards))


def order_distance(a, b):

    pairwise_a = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    pairwise_b = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)

    a_args = np.argsort(pairwise_a, axis=1)
    b_args = np.argsort(pairwise_b, axis=1)

    shape, shape = a_args.shape
    loss = 0
    for i in range(shape):
        a_line = a_args[i]
        b_line = b_args[i]
        shape = len(a_line)
        output = []

        for i in range(shape):
            elt = a_line[i]
            new_pos = np.arange(shape)[b_line == elt][0]
            output.append(new_pos)

        distances = abs(np.arange(shape)-np.array(output))
        # prob_dist = 1/(np.arange(shape)+1)+1
        # loss += distances @ prob_dist
        loss += distances

    return -loss


def compute_mutual_info(experience, n_pts=100):

    latent_states = [
        rssm_state.prev_state.stoch
        for rssm_state in experience['agent_infos']]
    ground_truth_states = [
        step_info.internal_state
        for step_info in experience['info']
    ]

    # Note: since latent states is prev state, not curr state
    # you might want to offset the two arrays by one
    aligned_latent = np.stack(latent_states[1:])
    aligned_ground = np.stack(ground_truth_states[:-1])
    np.random.seed(0)
    perm = np.random.permutation(aligned_ground.shape[0])[:n_pts]
    return order_distance(aligned_latent[perm], aligned_ground[perm])


our_pickles = [
    (10999, 'pickles/ours_10999_eval.pkl'),
    (100999, 'pickles/ours_100999_eval.pkl'),
    (150999, 'pickles/ours_150999_eval.pkl'),
    (200999, 'pickles/ours_200999_eval.pkl'),
    (250999, 'pickles/ours_250999_eval.pkl'),
    (300999, 'pickles/ours_300999_eval.pkl'),
    (350999, 'pickles/ours_350999_eval.pkl'),
    (400999, 'pickles/ours_400999_eval.pkl'),
    (450999, 'pickles/ours_450999_eval.pkl'),
    (500999, 'pickles/ours_500999_eval.pkl'),
    (550999, 'pickles/ours_550999_eval.pkl'),
    (600999, 'pickles/ours_600999_eval.pkl'),
    (650999, 'pickles/ours_650999_eval.pkl'),
]
baseline_pickles = [
    (10999, 'pickles/baseline_10999_eval.pkl'),
    (100999, 'pickles/baseline_100999_eval.pkl'),
    (150999, 'pickles/baseline_150999_eval.pkl'),
    (200999, 'pickles/baseline_200999_eval.pkl'),
    (250999, 'pickles/baseline_250999_eval.pkl'),
    (300999, 'pickles/baseline_300999_eval.pkl'),
    (350999, 'pickles/baseline_350999_eval.pkl'),
    (400999, 'pickles/baseline_400999_eval.pkl'),
    (450999, 'pickles/baseline_450999_eval.pkl'),
    (500999, 'pickles/baseline_500999_eval.pkl'),
    (550999, 'pickles/baseline_550999_eval.pkl'),
    (600999, 'pickles/baseline_600999_eval.pkl'),
    (650999, 'pickles/baseline_650999_eval.pkl'),
]


if __name__ == "__main__":

    x = []
    y = []
    algo = []
    dataframe_path = 'rank_dataframe.pkl'
    if os.path.exists(dataframe_path):
        df = pd.read_pickle(dataframe_path)
    else:
        df = pd.DataFrame()
        for i, pkl_file in tqdm(our_pickles):
            experience = pickle.load(open(pkl_file, 'rb'))
            x.append(i)
            y.append(compute_mutual_info(experience))
            algo.append('Ours')
        for i, pkl_file in tqdm(baseline_pickles):
            experience = pickle.load(open(pkl_file, 'rb'))
            x.append(i)
            y.append(compute_mutual_info(experience))
            algo.append('Dreamer')
        df['Training Iterations'] = np.array(x)
        df['Y'] = np.array(y)
        df['Algorithm'] = algo
        df.to_pickle(dataframe_path)
    sns.lineplot(
        data=df,
        x="Training Iterations",
        y="Y",
        hue="Algorithm"
    )
    plt.grid()
    plt.show()
