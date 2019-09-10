import numpy as np
import os
import json
import tensorflow as tf
from rl_reasoner.pg_rnn import PolicyGradientRNN
from rl_reasoner.sampler import Sampler
from rl_reasoner.model import policy_network
from rl_reasoner.environment import Neo4jEnvironment


class Query():

    def __init__(self):  # noqa: E501
        os.chdir("/home/mwawer/src/rl_reasoner/rl_reasoner")
        base_folder = "/home/mwawer/src/rl_reasoner/rl_reasoner/results/" + "2019-07-12_13-26-35.576782"
        config = json.load(open(base_folder + "/configuration.json"))

        # read embeddings
        entity_embeddings = None
        if config['entity_embedding_file'] != '':
                with open(config['entity_embedding_file']) as f:
                        entity_embeddings = np.loadtxt(f, skiprows=1)
                        entities = entity_embeddings[:, 0].astype(int).tolist()
                        entity_embeddings = np.delete(entity_embeddings, [0], axis=1)

        relation_embeddings = None
        if config['relation_embedding_file'] != '':
                with open(config['relation_embedding_file']) as f:
                        relation_embeddings = np.loadtxt(f)

        self.env = Neo4jEnvironment(entities)
        observation_dim = self.env.observation_dim

        embedding_size = config["embedding_size"]
        mlp_hidden_size = config["mlp_hidden_size"]

        # RNN configuration
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_adaptive = config["learning"]["learning_adaptive"]
        if learning_adaptive:
                learning_rate = tf.train.exponential_decay(config["learning"]["learning_rate"],
                                                           global_step,
                                                           config["learning"]["decay_steps"],
                                                           config["learning"]["decay_rate"],
                                                           staircase=True)
        else:
                learning_rate = config["learning"]["learning_rate"]

        # tensorflow
        sess = tf.Session()
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        writer = tf.summary.FileWriter(base_folder + "/query_summary/")
        save_path = base_folder + '/models/'

        pg_rnn = PolicyGradientRNN(sess,
                                   optimizer,
                                   policy_network,
                                   observation_dim,
                                   embedding_size,
                                   (self.env.num_entities, self.env.num_relations),
                                   entity_embeddings,
                                   relation_embeddings,
                                   config["train_entity_embeddings"],
                                   config["train_relation_embeddings"],
                                   mlp_hidden_size,
                                   config["gru_unit_size"],
                                   config["num_step"],
                                   config["num_layers"],
                                   save_path + self.env.spec["id"],
                                   global_step,
                                   config["max_gradient_norm"],
                                   config["entropy_bonus"],
                                   writer,
                                   loss_function=config["loss_function"],
                                   summary_every=10)

        self.sampler = Sampler(pg_rnn,
                               self.env,
                               None,
                               config["gru_unit_size"],
                               config["num_step"],
                               config["num_layers"],
                               config["max_step"],
                               config["batch_size"],
                               config["discount"],
                               writer)

        self.reward = []

    def query(self, query_entity, query_relation):
        episode = self.sampler.collect_one_episode(query_entity=query_entity, query_relation=query_relation)
        n_paths = episode["actions"].shape[0]
        result = []
        for i in range(n_paths):
            path = {"nodes": [], "edges": []}
            n_steps = episode["actions"].shape[1]

            ent_id = self.env.entity_list[int(episode["observations"][i, 0, 1])]
            path["nodes"].append({"id": ent_id,
                                  "name": ent_id,
                                  "type": [record['node']['name'] for record in self.env.graph.get_node_by_id(ent_id)][0]})

            for j in range(n_steps):
                ent_id = self.env.entity_list[int(episode["available_actions"][i, j, int(episode["actions"][i, j])][1])]
                path["nodes"].append({"id": ent_id,
                                      "name": ent_id,
                                      "type": [record['node']['name'] for record in self.env.graph.get_node_by_id(ent_id)][0]})

                path["edges"].append({"id": j,
                                      "name": self.env.relation_list[int(episode["available_actions"][i, j, int(episode["actions"][i, j])][0])],
                                      "source_id": path["nodes"][j]["id"],
                                      "target_id": path["nodes"][j + 1]["id"]})
            result.append(path)
        return(json.dumps(result))
