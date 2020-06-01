import numpy as np
import random
import scipy.signal
import tensorflow as tf


class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 queries,
                 gru_unit_size=16,
                 num_step=10,
                 num_layers=1,
                 max_step=2000,
                 batch_size=10000,
                 discount=1.00,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.queries = queries
        self.gru_unit_size = gru_unit_size
        self.num_step = num_step
        self.num_layers = num_layers
        self.max_step = max_step
        self.batch_size = batch_size
        self.discount = discount
        self.summary_writer = summary_writer

    def flush_summary(self, value, tag="rewards"):
        global_step = self.policy.session.run(self.policy.global_step)
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def compute_monte_carlo_returns(self, rewards):
        return scipy.signal.lfilter([1.], [1, -self.discount], rewards[::-1])[::-1]

    def collect_one_episode(self, query_entity=None, query_relation=None, targets=[]):
        observations, available_actions, actions, rewards = [], [], [], []
        init_states = tuple([] for _ in range(self.num_layers))

        if query_entity is None or query_relation is None:
            query_entity = random.choice(list(self.queries.keys()))
            query_relation = random.choice(list(self.queries[query_entity].keys()))
            targets = self.queries[query_entity][query_relation]

        observation, available_actions_ep, query_relation = self.env.reset(query_entity, query_relation, targets)
        init_state = tuple([np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])

        for t in range(self.max_step):
            action, final_state = self.policy.sampleAction(observation[np.newaxis, np.newaxis, :], available_actions_ep[np.newaxis, np.newaxis, :], np.array([[query_relation]]), init_state)
            next_observation, next_available_actions_ep, reward, done, _ = self.env.step(action)

            # appending the experience
            observations.append(observation)
            available_actions.append(available_actions_ep)
            actions.append(action)
            rewards.append(reward)
            for i in range(self.num_layers):
                init_states[i].append(init_state[i][0])

            # going to next state
            observation = next_observation
            available_actions_ep = next_available_actions_ep
            init_state = final_state
            if done:
                break

        self.flush_summary(np.sum(rewards))
        returns = self.compute_monte_carlo_returns(rewards)
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)  # moved to batch following MINERVA!
        available_actions = self.expand_available_actions(available_actions, axis=0)
        episode = dict(
                    observations = np.array(observations),
                    available_actions = np.array(available_actions),
                    actions = np.array(actions),
                    returns = np.array(returns),
                    init_states = init_states,
                    query_relations = np.repeat(query_relation,len(actions)),
                    rewards = np.array(rewards))
        return self.expand_episode(episode)

    def collect_one_batch(self):
        episodes = []
        len_samples = 0
        while len_samples < self.batch_size:
            episode = self.collect_one_episode()
            episodes.append(episode)
            len_samples += 1 #np.sum(episode["seq_len"]) ## changed so each episode is counted once (rather than each step)!
        # prepare input
        observations = np.concatenate([episode["observations"] for episode in episodes])
        available_actions = np.concatenate(self.expand_available_actions([episode["available_actions"] for episode in episodes], axis=2))
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        returns = np.concatenate([episode["returns"] for episode in episodes])
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)  # moved here following MINERVA
        init_states = tuple(
                       np.concatenate([episode["init_states"][i]
                                       for episode in episodes])
                       for i in range(self.num_layers))
        seq_len = np.concatenate([episode["seq_len"] for episode in episodes])
        query_relations = np.concatenate([episode["query_relations"] for episode in episodes])
        batch = dict(
                    observations = observations,
                    available_actions = available_actions,
                    actions = actions,
                    returns = returns,
                    init_states = init_states,
                    seq_len = seq_len,
                    query_relations = query_relations,
                    rewards = rewards
                    )
        return batch

    def expand_episode(self, episode):
        episode_size = len(episode["returns"])
        if episode_size % self.num_step:
            batch_from_episode = (episode_size // self.num_step + 1)
        else:
            batch_from_episode = (episode_size // self.num_step)

        extra_length = batch_from_episode * self.num_step - episode_size
        last_batch_size = self.num_step - extra_length

        batched_episode = {}
        for key, value in episode.items():
            if key == "init_states":
                truncated_value = tuple(value[i][::self.num_step] for i in
                                        range(self.num_layers))
                batched_episode[key] = truncated_value
            else:
                expanded_value = np.concatenate([value, np.zeros((extra_length,) +
                                                     value.shape[1:])])
                batched_episode[key] = expanded_value.reshape((-1, self.num_step) +
                                                         value.shape[1:])

        seq_len = [self.num_step] * (batch_from_episode - 1) + [last_batch_size]
        batched_episode["seq_len"] = np.array(seq_len)
        return batched_episode

    def expand_available_actions(self, actions, axis=0):
        npad = np.zeros((actions[0].ndim,2), dtype=np.int32)
        n_actions = [np.ma.size(action_list, axis) for action_list in actions]
        max_n_actions = max(n_actions)
        for i in range(len(n_actions)):
            extra_length = max_n_actions - n_actions[i]
            if extra_length > 0:
                npad[axis,1] = extra_length
                actions[i] = np.pad(actions[i], pad_width=npad, mode='constant', constant_values=0)
        return actions

    def samples(self):
        return self.collect_one_batch()
