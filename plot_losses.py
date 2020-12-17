from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_tf_event_file(path):
    data = {}
    try:
        for e in summary_iterator(path):
            for v in e.summary.value:
                if v.tag not in data:
                    data[v.tag] = []
                data[v.tag].append((
                    e.step,
                    v.simple_value))
    except Exception as e:
        print(e)
        pass
    return data


if __name__ == "__main__":
    ours_data = parse_tf_event_file(
        'logs/dec16-distractor-envs/run_1/events.out.tfevents.1608113549.workstation.17230.0')
    baseline_data = parse_tf_event_file(
        'logs/dec16-distractor-envs/crvlab/dec16-distractor-envs/run_0/events.out.tfevents.1608109092.crvdesktop00.22278.0')
    df = pd.DataFrame()
    x = []
    rewards = []
    algo = []
    for step, reward in ours_data['reward_loss/Average']:
        x.append(step)
        rewards.append(reward)
        algo.append('Ours')
    for step, reward in baseline_data['reward_loss/Average']:
        x.append(step)
        rewards.append(reward)
        algo.append('Dreamer')
    df['Training Iterations'] = x
    df['Reward Loss'] = rewards
    df['Algorithm'] = algo
    sns.lineplot(
        data=df,
        x="Training Iterations",
        y="Reward Loss",
        hue="Algorithm"
    )
    plt.grid()
    plt.show()

    x = []
    rewards = []
    algo = []
    for step, reward in ours_data['model_loss/Average']:
        x.append(step)
        rewards.append(reward)
        algo.append('Ours')
    for step, reward in baseline_data['model_loss/Average']:
        x.append(step)
        rewards.append(reward)
        algo.append('Dreamer')
    df['Training Iterations'] = x
    df['Model Loss'] = rewards
    df['Algorithm'] = algo
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="Training Iterations",
        y="Model Loss",
        hue="Algorithm",
        ax=ax
    )
    ax.set(yscale="log")
    plt.grid()
    plt.show()
