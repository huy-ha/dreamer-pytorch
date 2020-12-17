import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
from pandas import DataFrame
import os


def compute_mean_episode_rewards(experience):
    episode_rewards = [0.0]
    for done, reward in zip(experience['done'], experience['reward']):
        episode_rewards[-1] += reward
        if done:
            episode_rewards.append(0.0)
    print('Mean Episode Rewards:', np.mean(episode_rewards))
    return episode_rewards


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
    dataframe_path = 'dataframe.pkl'
    if os.path.exists(dataframe_path):
        df = pd.read_pickle(dataframe_path)
    else:
        df = pd.DataFrame()
        for i, pkl_file in our_pickles:
            experience = pickle.load(open(pkl_file, 'rb'))
            for episode_reward in compute_mean_episode_rewards(experience):
                x.append(i)
                y.append(episode_reward)
                algo.append('Ours')
        for i, pkl_file in baseline_pickles:
            experience = pickle.load(open(pkl_file, 'rb'))
            for episode_reward in compute_mean_episode_rewards(experience):
                x.append(i)
                y.append(episode_reward)
                algo.append('Dreamer')
        df['Training Iterations'] = np.array(x)
        df['Episode Rewards'] = np.array(y)
        df['Algorithm'] = algo
        df.to_pickle(dataframe_path)
    sns.lineplot(
        data=df,
        x="Training Iterations",
        y="Episode Rewards",
        hue="Algorithm"
    )
    plt.grid()
    plt.show()
