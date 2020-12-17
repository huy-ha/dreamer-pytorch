import pickle
import numpy as np
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from time import time


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


def plot_tsne(experience):
    latent_states = np.array([
        list(rssm_state.prev_state.stoch)
        for rssm_state in experience['agent_infos']])
    rewards = np.array(experience['reward'])
    feature_cols = ['axis_'+str(i) for i in range(latent_states.shape[1])]
    df = DataFrame(latent_states, columns=feature_cols)
    df['y'] = rewards
    print(df.shape)
    plt.title('Latent States Embedding')
    time_start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=10000)
    tsne_results = tsne.fit_transform(
        df[feature_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("flare", as_cmap=True),
        data=df,
        alpha=0.6
    )
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    experience = pickle.load(open('ours-experience.pkl', 'rb'))
    plot_tsne(experience)
    compute_mean_episode_rewards(experience)
    compute_mutual_info(experience)
