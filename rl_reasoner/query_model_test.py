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


def read_queries(self, query_file):
    with open(query_file) as f:
        queries = dict()
        csv_file = csv.reader(f, delimiter='\t')
        for line in csv_file:
            query_entity = int(line[0])
            query_relation = line[1]
            query_target = int(line[2])
            if query_entity not in queries:
                queries[query_entity] = dict()
            if query_relation not in queries[query_entity]:
                queries[query_entity][query_relation] = list()
            queries[query_entity][query_relation].append(query_target)
    return queries


def main(query_entity, query_relation):
  query_file = "../data/drug_hasIndication_disease_all.txt"
  base_folder = "results/" + "2018-11-29_13-40-12.074766"
  num_itr = 20

  config = json.load(open(base_folder + "/configuration.json"))
  train = 0


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

  env = Neo4jEnvironment(entities)
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

  writer = tf.summary.FileWriter(base_folder + "/query_summary/")
  save_path = base_folder + '/models/'

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
                    None,
                    config["gru_unit_size"],
                    config["num_step"],
                    config["num_layers"],
                    config["max_step"],
                    config["batch_size"],
                    config["discount"],
                    writer)

  reward = []

  print(os.getwcd())
  print(save_path + env.spec["id"])

  episode = sampler.collect_one_episode(query_entity = query_entity, query_relation = query_relation)
  for i in range(episode["actions"].shape[0]):
    ent_id = env.entity_list[int(episode["observations"][i,0,1])]
    ent = [record['node']['name'] for record in env.graph.get_node_by_id(ent_id)][0]
    print('(' + ent + ', id=' + str(ent_id) + ')', end = '')
    for j in range(episode["actions"].shape[1]):
        rel = env.relation_list[int(episode["available_actions"][i,j,int(episode["actions"][i,j])][0])]
        ent_id = env.entity_list[int(episode["available_actions"][i,j,int(episode["actions"][i,j])][1])]
        ent = [record['node']['name'] for record in env.graph.get_node_by_id(ent_id)][0]
        print('-[' + rel + ']-', end = '')
        print('(' + ent + ', id=' + str(ent_id) + ')', end = '')
        if j == episode["actions"].shape[1]-1:
            print(', ', end = '')
    print("{0}\n".format(np.sum(episode["rewards"][i])), end = '')

  # for i in tqdm(range(num_itr)):
  #     episode = sampler.collect_one_episode(query_entity = query_entity, query_relation = query_relation)
  #     print("reward is {0}".format(np.sum(episode["rewards"])))

  #     with open(base_folder + "/query_results.txt", "a") as f:
  #         f.write("iteration " + str(i) + "\n")
  #         for i in range(episode["actions"].shape[0]):
  #             ent_id = env.entity_list[int(episode["observations"][i,0,1])]
  #             ent = [record['node']['name'] for record in env.graph.get_node_by_id(ent_id)][0]
  #             f.write('(' + ent + ', id=' + str(ent_id) + ')')
  #             for j in range(episode["actions"].shape[1]):
  #                 rel = env.relation_list[int(episode["available_actions"][i,j,int(episode["actions"][i,j])][0])]
  #                 ent_id = env.entity_list[int(episode["available_actions"][i,j,int(episode["actions"][i,j])][1])]
  #                 ent = [record['node']['name'] for record in env.graph.get_node_by_id(ent_id)][0]
  #                 f.write('-[' + rel + ']-')
  #                 f.write('(' + ent + ', id=' + str(ent_id) + ')')
  #                 if j == episode["actions"].shape[1]-1:
  #                     f.write(', ')
  #             f.write("{0}\n".format(np.sum(episode["rewards"][i])))

if __name__ == '__main__':
  main(185855, "HAS_INDICATION")
