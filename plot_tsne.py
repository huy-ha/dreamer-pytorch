import pickle
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
from dreamer.models.rnns import stack_states
import os
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm
from pprint import pprint


def plot_tsne(experience=None, latent_states=None, rewards=None):
    if latent_states is None or rewards is None:
        latent_states = np.array([
            list(rssm_state.prev_state.stoch)
            for rssm_state in experience['agent_infos']])
        rewards = np.array(experience['reward'])
    np.random.seed(0)
    perm = np.random.permutation(10000)
    latent_states = latent_states[perm]
    rewards = rewards[perm]
    feature_cols = ['axis_'+str(i) for i in range(latent_states.shape[1])]
    df = DataFrame(latent_states, columns=feature_cols)
    df['y'] = rewards
    time_start = time()
    tsne = TSNE(n_components=2,
                verbose=1,
                perplexity=1000,
                n_iter=1000,
                n_jobs=16)
    tsne_results = tsne.fit_transform(
        df[feature_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))
    pickle.dump(tsne_results, open('tsne_results.pkl', 'wb'))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("flare", as_cmap=True),
        data=df,
        alpha=0.6,
        s=5
    )
    plt.show()


def show_img(img, ax):
    img = np.swapaxes(img, 0, -1)
    img = np.swapaxes(img, 0, 1)
    ax.axis('off')
    ax.imshow(img)


if __name__ == "__main__":
    if os.path.exists('tsne_results.pkl'):
        # use precomputed tsne results
        tsne_results = pickle.load(open('tsne_results.pkl', 'rb'))
        latent_states = pickle.load(open('latents.pkl', 'rb'))
        rewards = pickle.load(open('rewards.pkl', 'rb'))
        experience = pickle.load(open('ours-experience.pkl', 'rb'))
        np.random.seed(0)
        perm = np.random.permutation(10000)
        obs = np.stack(experience['obs'])[perm]
        latent_states = latent_states[perm]
        rewards = rewards[perm]
        # create figure
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 5)

        # points near 1, 1
        indices = [7885, 6319,  2010]
        for idx in indices:
            rewards[idx] = 5

        show_img(
            obs[7885],
            fig.add_subplot(gs[0, 4]))
        show_img(
            obs[6319],
            fig.add_subplot(gs[1, 4]))
        show_img(
            obs[2010],
            fig.add_subplot(gs[2, 4]))

        # points near -1.3, -1.3
        indices = [521, 1541, 6701]
        for idx in indices:
            rewards[idx] = 5

        show_img(
            obs[521],
            fig.add_subplot(gs[0, 0]))
        show_img(
            obs[1541],
            fig.add_subplot(gs[1, 0]))
        show_img(
            obs[6701],
            fig.add_subplot(gs[2, 0]))

        # plot tsne
        ax = fig.add_subplot(gs[:, 1:4])
        df = DataFrame()
        df['y'] = rewards
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("flare", as_cmap=True),
            data=df,
            alpha=0.6,
            s=5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.get_legend().remove()

        plt.tight_layout()
        plt.show()
        # distances = [
        #     (i, np.linalg.norm(tsne_pnt-pnt))
        #     for i, tsne_pnt in tqdm(enumerate(tsne_results))]
        # distances.sort(key=lambda x: x[1])
        # indices = [x[0] for x in distances[:5]]
        # imgs = obs[indices]
        # print(indices)
        # fig, axes = plt.subplots(5, 1)
        # for img, ax in zip(imgs, axes):
        #     img = np.swapaxes(img, 0, -1)
        #     img = np.swapaxes(img, 0, 1)
        #     ax.imshow(img)
        #     ax.axis('off')
        # plt.show()
    elif os.path.exists('latents.pkl') and os.path.exists('rewards.pkl'):
        # use cached latents and rewards
        latent_states = pickle.load(open('latents.pkl', 'rb'))
        rewards = pickle.load(open('rewards.pkl', 'rb'))
        plot_tsne(latent_states=latent_states, rewards=rewards)
    else:
        # do from scratch
        experience = pickle.load(open('ours-experience.pkl', 'rb'))
        latent_states = np.array([
            list(rssm_state.prev_state.stoch)
            for rssm_state in experience['agent_infos']])
        rewards = np.array(experience['reward'])
        pickle.dump(latent_states, open('latents.pkl', 'wb'))
        pickle.dump(rewards, open('rewards.pkl', 'wb'))
        plot_tsne(latent_states=latent_states, rewards=rewards)
