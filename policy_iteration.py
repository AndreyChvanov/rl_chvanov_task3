import numpy as np
import gym


class PolicyIteration:
    def __init__(self, env, gamma, eval_policy_th):
        self.env = env
        self.count_action = len(self.env.action_set)
        self.gamma = gamma
        self.eval_th = eval_policy_th
        self.count_states = len(self.env.transition_matrix)

    def get_mdp(self):
        P = np.zeros((self.count_states, self.count_action, self.count_states))
        R = np.zeros((self.count_states, self.count_action, self.count_states))
        for state in self.env.transition_matrix.keys():
            for action_ind, cond_probs in self.env.transition_matrix[state].items():
                for prob in cond_probs:
                    next_state_prob, next_state, reward = prob[0], prob[1], prob[2]
                    P[state, action_ind, next_state] += next_state_prob
                    R[state, action_ind, next_state] += reward
        return P, R

    def iterative_evaluation(self, policy, th=0.001):
        p, r = self.get_mdp()
        V = np.zeros(self.count_states)
        while True:
            delta = 0
            prev_v = V.copy()
            for s in range(self.count_states):
                v = prev_v[s]
                v2_ = np.sum(policy[s] @ (p[s] * (r[s] + self.gamma * prev_v)))
                V[s] = v2_
                delta = max(delta, abs(v - V[s]))
            if delta < th:
                break
        return V

    def policy_iteration(self):
        policy = np.random.uniform(0, 1, (self.count_states, self.count_action))
        policy = policy / np.sum(policy, axis=1)[:, None]
        i = 0
        while True:
            i += 1
            V = self.iterative_evaluation(policy, self.eval_th)
            p, r = self.get_mdp()
            policy_stable = True
            for s in range(self.count_states):
                old_action = policy[s].argmax()
                values = []
                for a in range(self.count_action):
                    values.append(np.sum(p[s, a] * (r[s, a] + self.gamma * V)))
                t = np.argmax(values)
                policy[s] = np.zeros_like(policy[s])
                policy[s, t] = 1
                if old_action != t:
                    policy_stable = False
            if policy_stable:
                break
        return policy, V, i

    def value_iteration(self, th):
        V = np.random.uniform(0, 1, self.count_states)
        p, r = self.get_mdp()
        i = 0
        while True:
            delta = 0
            i += 1
            prev_v = V.copy()
            for s in range(self.count_states):
                v = V[s]
                values = []
                for a in range(self.count_action):
                    values.append(np.sum(p[s, a] * (r[s, a] + self.gamma * prev_v)))
                V[s] = np.max(values)
                delta = max(delta, abs(v - V[s]))
            if delta < th:
                break
        policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.count_states):
            values = []
            for a in range(self.count_action):
                values.append(np.sum(p[s, a] * (r[s, a] + self.gamma * V)))
            t = np.argmax(values)
            policy[s, t] = 1
        return policy, V, i


def show_policy_game_board(policy, env_shape):
    p = policy.argmax(axis=1)
    return p.reshape(env_shape)


def show_V_game_board(V, env_shape):
    return V.reshape(env_shape)


if __name__ == "__main__":
    env = gym.make('frozen_lake:default-v0', map_name='small', action_set_name='slippery')
    env.reset()
    alg = PolicyIteration(env, gamma=0.5, eval_policy_th=0.0001)
    p, v, i = alg.policy_iteration()
    print(i)

    p, v, i = alg.value_iteration(th=0.00001)
    print(i)
