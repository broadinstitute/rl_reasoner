import datetime
import numpy as np
import json
import os
import tensorflow as tf
from tqdm import tqdm
from pg_rnn import PolicyGradientRNN
from sampler import Sampler
from model import policy_network
from environment import Neo4jEnvironment

from tensorflow.python import debug as tf_debug

config = json.load(open("configuration_kg1.json"))
train = config["train"]

query_file = config["query_file"]


# read embeddings
entity_embeddings = None
if config['entity_embedding_file'] != '':
    with open(config['entity_embedding_file']) as f:
        entity_embeddings = np.loadtxt(f, skiprows=1)
        entities = entity_embeddings[:,0].astype(int).tolist()
        entity_embeddings = np.delete(entity_embeddings, [0], axis=1)

relation_embeddings = None
if config['relation_embedding_file'] != '':
    with open(config['relation_embedding_file']) as f:
        relation_embeddings = np.loadtxt(f)


env = Neo4jEnvironment(query_file, entities)
observation_dim = env.observation_dim

embedding_size = config["embedding_size"]
mlp_hidden_size = config["mlp_hidden_size"]

# RNN configuration
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_adaptive = config["learning"]["learning_adaptive"]
if learning_adaptive:
    learning_rate = tf.train.exponential_decay(
                      config["learning"]["learning_rate"],
                      global_step,
                      config["learning"]["decay_steps"],
                      config["learning"]["decay_rate"],
                      staircase=True)
else:
    learning_rate = config["learning"]["learning_rate"]



# tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

# checkpointing
#base_folder = "_".join([k + "-" + str(v) for k, v in sorted(config.items())
#                                    if k not in ["train", "learning"]])
base_folder = "results/" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
os.makedirs(base_folder, exist_ok=True)
json.dump(config, open(base_folder + "/configuration.json", "w"))
writer = tf.summary.FileWriter(base_folder + "/summary/")
save_path = base_folder + '/models/'
os.makedirs(save_path, exist_ok=True)

pg_rnn = PolicyGradientRNN(sess,
                           optimizer,
                           policy_network,
                           observation_dim,
                           embedding_size,
                           (env.num_entities, env.num_relations),
                           entity_embeddings,
                           relation_embeddings,
                           config["train_entity_embeddings"],
                           config["train_relation_embeddings"],
                           mlp_hidden_size,
                           config["gru_unit_size"],
                           config["num_step"],
                           config["num_layers"],
                           save_path + env.spec["id"],
                           global_step,
                           config["max_gradient_norm"],
                           config["entropy_bonus"],
                           writer,
                           loss_function=config["loss_function"],
                           summary_every=10)

sampler = Sampler(pg_rnn,
                  env,
                  config["gru_unit_size"],
                  config["num_step"],
                  config["num_layers"],
                  config["max_step"],
                  config["batch_size"],
                  config["discount"],
                  writer)

reward = []
for _ in tqdm(range(config["num_itr"])):
    if train:
        batch = sampler.samples()
        #print(batch["query_relations"])
        pg_rnn.update_parameters(batch["observations"],
                                 batch["available_actions"],
                                 batch["actions"],
                                 batch["returns"],
                                 batch["query_relations"],
                                 batch["init_states"],
                                 batch["seq_len"])
    else:
        episode = sampler.collect_one_episode()
        print("reward is {0}".format(np.sum(episode["rewards"])))
