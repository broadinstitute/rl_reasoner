import numpy as np
import tensorflow as tf


class PolicyGradientRNN(object):

    def __init__(self, session,
                 optimizer,
                 policy_network,
                 observation_dim,
                 embedding_size,
                 vocab_size,
                 entity_embeddings,
                 relation_embeddings,
                 train_entity_embeddings,
                 train_relation_embeddings,
                 mlp_hidden_size,
                 gru_unit_size,
                 num_step,
                 num_layers,
                 save_path,
                 global_step,
                 max_gradient=5,
                 entropy_bonus=0.001,
                 summary_writer=None,
                 loss_function="l2",
                 summary_every=100):

        # tensorflow machinery
        self.session        = session
        self.optimizer      = optimizer
        self.summary_writer = summary_writer
        self.summary_every  = summary_every
        self.gru_unit_size  = gru_unit_size
        self.num_step       = num_step
        self.num_layers     = num_layers
        self.no_op          = tf.no_op()

        # model components
        self.policy_network  = policy_network
        self.observation_dim = observation_dim
        self.embedding_size  = embedding_size
        self.vocab_size      = vocab_size
        self.mlp_hidden_size = mlp_hidden_size
        self.loss_function   = loss_function

        # training parameters
        self.max_gradient    = max_gradient
        self.entropy_bonus   = entropy_bonus
        self.train_entity_embeddings = train_entity_embeddings
        self.train_relation_embeddings = train_relation_embeddings

        # counter
        self.global_step = global_step

        # create variables
        self.create_variables()

        # intitialize variables
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        # initialize embeddings
        if entity_embeddings is not None:
            entity_embedding_init = self.entity_embedding.assign(self.entity_embedding_placeholder)
            self.session.run([entity_embedding_init], {self.entity_embedding_placeholder: entity_embeddings})

        if relation_embeddings is not None:
            relation_embedding_init = self.relation_embedding.assign(self.relation_embedding_placeholder)
            self.session.run([relation_embedding_init], {self.relation_embedding_placeholder: relation_embeddings})

        # try load saved model
        self.saver = tf.train.Saver(tf.global_variables())
        self.save_path = save_path
        self.load_model()

        if self.summary_writer is not None:
            # graph was not available when journalist was created
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every

    def create_input_placeholders(self):
        with tf.name_scope("inputs"):
            self.observations = tf.placeholder(tf.int32, (None, None) + self.observation_dim, name="observations")
            self.actions = tf.placeholder(tf.int32, (None, None), name="actions")
            self.returns = tf.placeholder(tf.float32, (None, None), name="returns")
            self.init_states = tuple(tf.placeholder(tf.float32, (None, self.gru_unit_size), name="init_states" + str(i))
                                     for i in range(self.num_layers))
            self.seq_len = tf.placeholder(tf.int32, (None,), name="seq_len")
            self.actions_flatten = tf.reshape(self.actions, (-1,))
            self.returns_flatten = tf.reshape(self.returns, (-1,))

            self.available_actions = tf.placeholder(tf.int32, (None, None, None, 2), name="available_actions")
            self.query_relations = tf.placeholder(tf.int32, (None, None), name="query_relations")

            self.entity_embedding_placeholder = tf.placeholder(tf.float32, (self.vocab_size[0], self.embedding_size), name="entity_embedding")
            self.relation_embedding_placeholder = tf.placeholder(tf.float32, (self.vocab_size[1], self.embedding_size), name="relation_embedding")

    def create_variables_for_actions(self):
        with tf.name_scope("generating_actions"):
            with tf.variable_scope("policy_network"):
                self.logit, self.final_state, self.entity_embedding, self.relation_embedding = self.policy_network(self.observations, self.available_actions,
                                                                   self.init_states, self.seq_len,
                                                                   self.gru_unit_size, self.num_layers,
                                                                   self.embedding_size, self.mlp_hidden_size,
                                                                   self.vocab_size, self.query_relations,
                                                                   self.train_entity_embeddings, self.train_relation_embeddings)
            self.probs = tf.nn.softmax(self.logit)
            self.log_probs = tf.nn.log_softmax(self.logit)
            with tf.name_scope("computing_entropy"):
                self.entropy = - tf.reduce_sum(self.probs * self.log_probs, axis=2)

    def create_variables_for_optimization(self):
        with tf.name_scope("optimization"):
            with tf.name_scope("masker"):
                    self.mask = tf.sequence_mask(self.seq_len, self.num_step)
                    self.mask =  tf.cast(self.mask, tf.float32) # tf.reshape(tf.cast(self.mask, tf.float32), (-1,))
            #if self.loss_function == "cross_entropy":
            with tf.name_scope("loss"):
                self.pl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit,
                                                                                  labels=self.actions)
                # elif self.loss_function == "l2":
                #     self.one_hot_actions = tf.one_hot(self.actions_flatten, self.num_actions)
                #     self.pl_loss = tf.reduce_mean((self.probs - self.one_hot_actions) ** 2,
                #                                   axis=1)
                
                #else:
                #        raise ValueError("loss function type is not defined")

                self.pl_loss = tf.multiply(self.pl_loss, self.mask)
                self.pl_loss = tf.reduce_mean(tf.multiply(self.pl_loss, self.returns))

                self.entropy = tf.multiply(self.entropy, self.mask)
                self.entropy = tf.reduce_mean(self.entropy)

                self.loss = self.pl_loss - self.entropy_bonus * self.entropy

                self.actions_out = tf.reduce_mean(self.actions, 0)

            self.trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network")
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.clipped_gradients = [(tf.clip_by_norm(grad, self.max_gradient), var)
                                      for grad, var in self.gradients]
            self.train_op = self.optimizer.apply_gradients(self.clipped_gradients,
                                                           self.global_step)
            self.grad_norm = tf.global_norm([grad for grad, var in self.gradients])
            self.var_norm = tf.global_norm(self.trainable_variables)

    def create_summaries(self):
        self.policy_loss_summary = tf.summary.scalar("loss/policy_loss", self.pl_loss)
        self.entropy_loss_summary = tf.summary.scalar("loss/entropy_loss", self.entropy)
        self.var_norm_summary = tf.summary.scalar("loss/var_norm", self.var_norm)
        self.grad_norm_summary = tf.summary.scalar("loss/grad_norm", self.grad_norm)

    def merge_summaries(self):
        self.summarize = tf.summary.merge([self.policy_loss_summary,
                                           self.entropy_loss_summary,
                                           self.var_norm_summary,
                                           self.grad_norm_summary])

    def load_model(self):
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.session, load_path)
        except:
            print("no saved model to load. starting new session")
        else:
            print("loaded model: {}".format(load_path))
            self.saver = tf.train.Saver(tf.global_variables())

    def create_variables(self):
        self.create_input_placeholders()
        self.create_variables_for_actions()
        self.create_variables_for_optimization()
        self.create_summaries()
        self.merge_summaries()

    def sampleAction(self, observations, available_actions, query_relations, init_states, seq_len=[1]):
        probs, final_state = self.session.run([self.probs, self.final_state],
                                              {self.observations: observations, 
                                               self.available_actions: available_actions,
                                               self.query_relations: query_relations,
                                               self.init_states: init_states,
                                               self.seq_len: seq_len})
        #print(probs[0][0]) ## print probabilities for actions
        return np.random.choice(len(probs[0][0]), p=probs[0][0]), final_state

    def update_parameters(self, observations, available_actions, actions, returns, query_relations, init_states, seq_len):
        write_summary = (self.train_itr + 1) % self.summary_every == 0
        _, summary = self.session.run([self.train_op,
                                       self.summarize if write_summary else self.no_op],
                                      {self.observations: observations,
                                       self.available_actions: available_actions,
                                       self.actions: actions,
                                       self.returns: returns,
                                       self.query_relations: query_relations,
                                       self.init_states: init_states,
                                       self.seq_len: seq_len})
        
        if write_summary:
            self.summary_writer.add_summary(summary, self.train_itr)
            self.saver.save(self.session, self.save_path, global_step=self.global_step)

    @property
    def train_itr(self):
        return self.session.run(self.global_step)
