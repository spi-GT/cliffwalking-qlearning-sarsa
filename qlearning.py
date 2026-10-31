"""
Q-learning for CliffWalking-v1
训练、测试与参数分析
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

SEED = 42
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


class QLearningAgent:
    """Q-learning agent（ε-greedy 探索，off-policy 更新）"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_init=1.0, epsilon_min=0.001, epsilon_decay=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, terminated):
        # Q(S,A) ← Q(S,A) + α[R + γ·maxₐQ(S',a) - Q(S,A)]
        best_next = 0 if terminated else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def moving_average(data, window=20):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def train(env, agent, episodes=500, max_steps=1000, verbose=True):
    rewards_history = np.zeros(episodes)
    steps_history = np.zeros(episodes)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step_count in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        rewards_history[ep] = total_reward
        steps_history[ep] = step_count + 1

        if verbose and (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[max(0, ep - 99):ep + 1])
            print(f"Episode {ep + 1:3d}/{episodes} | ε={agent.epsilon:.4f} | "
                  f"Avg Reward(last 100): {avg_reward:.1f}")

    return rewards_history, steps_history


def test(env, agent, episodes=100):
    """ε=0 纯贪婪测试，返回奖励、步数、成功率、坠崖率"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0
    rewards = np.zeros(episodes)
    steps = np.zeros(episodes)
    successes = 0
    fall_episodes = 0

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        fell = False

        for step_count in range(10000):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if reward == -100:
                fell = True
            if terminated:
                successes += 1
            if terminated or truncated:
                break

        rewards[ep] = total_reward
        steps[ep] = step_count + 1
        if fell:
            fall_episodes += 1

    success_rate = successes / episodes
    fall_rate = fall_episodes / episodes

    print(f"\n{'=' * 45}")
    print(f"  测试结果 ({episodes} episodes, ε=0)")
    print(f"{'=' * 45}")
    print(f"  平均累积奖励:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  平均步数:      {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  成功率:        {success_rate * 100:.1f}%")
    print(f"  坠崖率:        {fall_rate * 100:.1f}%")
    print(f"{'=' * 45}\n")

    agent.epsilon = old_epsilon
    return rewards, steps, success_rate, fall_rate


def plot_learning_curve(rewards_history, title='Q-learning Learning Curve',
                        save_path='qlearning_curve.png'):
    window = 20
    smooth = moving_average(rewards_history, window)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.3, color='steelblue', linewidth=0.5, label='Raw reward')
    plt.plot(np.arange(window - 1, len(rewards_history)), smooth, color='crimson',
             linewidth=1.5, label=f'Moving average (window={window})')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"学习曲线已保存至 figures/{Path(save_path).name}")


def run_experiment(env, alpha=0.1, gamma=0.99, epsilon_decay=0.99, episodes=500):
    np.random.seed(SEED)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=alpha, gamma=gamma,
        epsilon_init=1.0, epsilon_min=0.001,
        epsilon_decay=epsilon_decay
    )
    rewards, _ = train(env, agent, episodes=episodes, verbose=False)
    return rewards


def plot_parameter_analysis(param_name, param_values, fixed_params, run_fn,
                            title, label, save_path, episodes=500):
    """固定其他参数，对比单一参数不同取值的学习曲线"""
    window = 20
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    plt.figure(figsize=(10, 5))
    for value, color in zip(param_values, colors):
        env = gym.make('CliffWalking-v1')
        rewards = run_fn(env, episodes=episodes, **fixed_params, **{param_name: value})
        smooth = moving_average(rewards, window)
        plt.plot(np.arange(window - 1, len(rewards)), smooth, color=color, linewidth=1.5,
                 label=f'{label} = {value}')
        env.close()

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward (moving average)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"参数分析已保存至 figures/{Path(save_path).name}")


def main():
    np.random.seed(SEED)
    print("=" * 50)
    print("  Q-learning for CliffWalking-v1")
    print("=" * 50)

    env = gym.make('CliffWalking-v1')
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1, gamma=0.99,
        epsilon_init=1.0, epsilon_min=0.001,
        epsilon_decay=0.99
    )

    print("\n[1/5] 训练 Q-learning agent...")
    rewards, _ = train(env, agent, episodes=500, max_steps=1000)
    print(f"训练完成，最终 ε = {agent.epsilon:.4f}")

    print("\n[2/5] 绘制学习曲线...")
    plot_learning_curve(rewards, save_path=FIG_DIR / 'qlearning_curve.png')

    print("\n[3/5] 测试 agent (ε=0)...")
    test(env, agent, episodes=100)
    env.close()

    print("\n[4/5] 参数分析：学习率 α 的影响...")
    plot_parameter_analysis('alpha', [0.1, 0.5, 0.9], {'gamma': 0.99}, run_experiment,
                            'Q-learning: Effect of Learning Rate α (γ=0.99)',
                            'α', FIG_DIR / 'qlearning_alpha_analysis.png')

    print("\n[5/5] 参数分析：折扣因子 γ 的影响...")
    plot_parameter_analysis('gamma', [0.9, 0.99, 0.999], {'alpha': 0.1}, run_experiment,
                            'Q-learning: Effect of Discount Factor γ (α=0.1)',
                            'γ', FIG_DIR / 'qlearning_gamma_analysis.png')

    print("\n" + "=" * 50)
    print("  全部完成！结果已保存为 PNG 图片文件。")
    print("=" * 50)


if __name__ == '__main__':
    main()
