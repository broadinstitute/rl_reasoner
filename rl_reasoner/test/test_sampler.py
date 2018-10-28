import json
import os
import tensorflow as tf
from tqdm import tqdm
from rl_reasoner.pg_rnn import PolicyGradientRNN
from rl_reasoner.sampler import Sampler
from rl_reasoner.model import policy_network
from rl_reasoner.environment import KGEnvironment

from tensorflow.python import debug as tf_debug


graph_file = "../data/simpsons1_graph.tsv"
query_file = "../data/simpsons1_query.tsv"
tmpdir = "../data/tmp"

config = json.load(open("configuration.json"))
train = config["train"]

env = KGEnvironment(query_file, graph_file)
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

#tensorflow
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

# checkpointing
writer = tf.summary.FileWriter(tmpdir + "/summary/")
save_path = tmpdir + '/models/'
os.makedirs(save_path, exist_ok=True)

pg_rnn = PolicyGradientRNN(sess,
                           optimizer,
                           policy_network,
                           observation_dim,
                           embedding_size,
                           (len(env.entities), len(env.relations)),
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
        print(batch["query_relations"])
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
