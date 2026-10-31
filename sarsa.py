"""
SARSA for CliffWalking-v1
训练、测试、参数分析及与 Q-learning 的对比
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from qlearning import (FIG_DIR, QLearningAgent, moving_average, plot_learning_curve,
                       plot_parameter_analysis, test, train as train_qlearning)

SEED = 42


class SARSAAgent:
    """SARSA agent（ε-greedy 探索，on-policy 更新）"""

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

    def update(self, state, action, reward, next_state, next_action, terminated):
        # Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
        q_next = 0 if terminated else self.q_table[next_state, next_action]
        td_target = reward + self.gamma * q_next
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_sarsa(env, agent, episodes=500, max_steps=1000, verbose=True):
    rewards_history = np.zeros(episodes)
    steps_history = np.zeros(episodes)

    for ep in range(episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0

        for step_count in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated:
                agent.update(state, action, reward, next_state, None, terminated=True)
            else:
                # on-policy：用当前策略为下一状态选动作
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, terminated=False)

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        agent.decay_epsilon()
        rewards_history[ep] = total_reward
        steps_history[ep] = step_count + 1

        if verbose and (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[max(0, ep - 99):ep + 1])
            print(f"Episode {ep + 1:3d}/{episodes} | ε={agent.epsilon:.4f} | "
                  f"Avg Reward(last 100): {avg_reward:.1f}")

    return rewards_history, steps_history


def run_sarsa_experiment(env, alpha=0.1, gamma=0.99, epsilon_decay=0.99, episodes=500):
    np.random.seed(SEED)
    agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=alpha, gamma=gamma,
        epsilon_init=1.0, epsilon_min=0.001,
        epsilon_decay=epsilon_decay
    )
    rewards, _ = train_sarsa(env, agent, episodes=episodes, verbose=False)
    return rewards


def compare_algorithms(episodes=500):
    """相同环境与超参数下对比 SARSA 与 Q-learning"""
    env = gym.make('CliffWalking-v1')
    params = dict(alpha=0.1, gamma=0.99, epsilon_init=1.0,
                  epsilon_min=0.001, epsilon_decay=0.99)

    print("训练 SARSA...")
    np.random.seed(SEED)
    sarsa_agent = SARSAAgent(env.observation_space.n, env.action_space.n, **params)
    sarsa_rewards, _ = train_sarsa(env, sarsa_agent, episodes=episodes, verbose=False)

    print("训练 Q-learning...")
    np.random.seed(SEED)
    q_agent = QLearningAgent(env.observation_space.n, env.action_space.n, **params)
    q_rewards, _ = train_qlearning(env, q_agent, episodes=episodes, verbose=False)

    window = 20
    sarsa_smooth = moving_average(sarsa_rewards, window)
    q_smooth = moving_average(q_rewards, window)
    x_sarsa = np.arange(window - 1, len(sarsa_rewards))
    x_q = np.arange(window - 1, len(q_rewards))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sarsa_rewards, alpha=0.2, color='forestgreen', linewidth=0.5)
    plt.plot(q_rewards, alpha=0.2, color='steelblue', linewidth=0.5)
    plt.plot(x_sarsa, sarsa_smooth, color='forestgreen', linewidth=1.5, label='SARSA')
    plt.plot(x_q, q_smooth, color='steelblue', linewidth=1.5, label='Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_sarsa, sarsa_smooth, color='forestgreen', linewidth=1.5, label='SARSA')
    plt.plot(x_q, q_smooth, color='steelblue', linewidth=1.5, label='Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward (smoothed)')
    plt.title(f'Smoothed Comparison (window={window})')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sarsa_vs_qlearning_curve.png', dpi=150)
    plt.show()
    print("对比学习曲线已保存至 figures/sarsa_vs_qlearning_curve.png")

    print("\n" + "=" * 45)
    print("  SARSA 测试")
    print("=" * 45)
    _, _, sarsa_sr, sarsa_fr = test(env, sarsa_agent, episodes=100)

    print("=" * 45)
    print("  Q-learning 测试")
    print("=" * 45)
    _, _, q_sr, q_fr = test(env, q_agent, episodes=100)

    print("\n" + "=" * 60)
    print("  SARSA vs Q-learning 对比总结")
    print("=" * 60)
    print(f"  {'指标':<20} {'SARSA (on-policy)':<20} {'Q-learning (off-policy)':<20}")
    print(f"  {'-'*58}")
    print(f"  {'平均累积奖励':<20} {np.mean(sarsa_rewards[-100:]):>+8.1f} (last 100)    {np.mean(q_rewards[-100:]):>+8.1f} (last 100)")
    print(f"  {'成功率':<20} {sarsa_sr * 100:<19.1f}% {q_sr * 100:<19.1f}%")
    print(f"  {'坠崖率':<20} {sarsa_fr * 100:<19.1f}% {q_fr * 100:<19.1f}%")
    print(f"  {'策略类型':<20} {'On-policy (保守)':<20} {'Off-policy (激进)':<20}")
    print("=" * 60)
    print("预期结论：")
    print("  - SARSA 更保守：学习过程中会考虑自身探索带来的风险，因此")
    print("    倾向于学习更安全的路径，坠崖率更低。")
    print("  - Q-learning 更激进：使用 max 算子的 off-policy 更新，")
    print("    倾向于找到理论最优（可能更短但更危险）的路径。")
    print("  - 在 CliffWalking 环境中，这种差异尤为明显，因为悬崖边缘")
    print("    的「一步之差」可能导致截然不同的结果。")
    print("=" * 60)

    env.close()


def main():
    np.random.seed(SEED)
    print("=" * 50)
    print("  SARSA for CliffWalking-v1")
    print("=" * 50)

    env = gym.make('CliffWalking-v1')
    agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1, gamma=0.99,
        epsilon_init=1.0, epsilon_min=0.001,
        epsilon_decay=0.99
    )

    print("\n[1/5] 训练 SARSA agent...")
    rewards, _ = train_sarsa(env, agent, episodes=500, max_steps=1000)
    print(f"训练完成，最终 ε = {agent.epsilon:.4f}")
    env.close()

    print("\n[2/5] 绘制 SARSA 学习曲线...")
    plot_learning_curve(rewards, title='SARSA Learning Curve',
                        save_path=FIG_DIR / 'sarsa_curve.png')

    print("\n[3/5] 测试 SARSA agent (ε=0)...")
    env = gym.make('CliffWalking-v1')
    test(env, agent, episodes=100)
    env.close()

    print("\n[4/5] SARSA 参数分析...")
    plot_parameter_analysis('alpha', [0.1, 0.5, 0.9], {'gamma': 0.99}, run_sarsa_experiment,
                            'SARSA: Effect of Learning Rate α (γ=0.99)',
                            'α', FIG_DIR / 'sarsa_alpha_analysis.png')
    plot_parameter_analysis('gamma', [0.9, 0.99, 0.999], {'alpha': 0.1}, run_sarsa_experiment,
                            'SARSA: Effect of Discount Factor γ (α=0.1)',
                            'γ', FIG_DIR / 'sarsa_gamma_analysis.png')

    print("\n[5/5] SARSA vs Q-learning 系统对比...")
    compare_algorithms(episodes=500)

    print("\n" + "=" * 50)
    print("  全部完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
