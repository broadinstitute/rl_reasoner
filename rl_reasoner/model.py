import tensorflow as tf
import numpy as np

def policy_network(observations,
                   available_actions,
                   init_states,
                   seq_len,
                   gru_unit_size,
                   num_layers,
                   embedding_size,
                   mlp_hidden_size,
                   vocab_size):
    """ define policy neural network """
    with tf.variable_scope("embedding"):
        entity_embedding = tf.get_variable("entity_embedding", [vocab_size[0], embedding_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        relation_embedding = tf.get_variable("relation_embedding", [vocab_size[1], embedding_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)


    with tf.variable_scope("rnn"):
        def gru_cell():
            gru = tf.contrib.rnn.GRUCell(gru_unit_size)
            return gru

        # obs_embedding = encoder_network(observations[:,:,0], observations[:,:,1], vocab_size, embedding_size)
        obs_embedding = tf.concat([tf.nn.embedding_lookup(relation_embedding, observations[:,:,0]), tf.nn.embedding_lookup(entity_embedding, observations[:,:,1])], axis=-1)
        gru_cells = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(num_layers)])
        output, final_state = tf.nn.dynamic_rnn(gru_cells, obs_embedding,
                initial_state=init_states, sequence_length=seq_len)

        # output = tf.reshape(output, [-1, gru_unit_size])

    # with tf.variable_scope("softmax"):
    #     w_softmax = tf.get_variable("w_softmax", shape=[gru_unit_size, num_actions],
    #         initializer=tf.contrib.layers.xavier_initializer())
    #     b_softmax = tf.get_variable("b_softmax", shape=[num_actions],
    #         initializer=tf.constant_initializer(0))

    # logit = tf.matmul(output, w_softmax) + b_softmax

    with tf.variable_scope("softmax"):
        # action_embeddings = encoder_network(available_actions[:,:,:,0], available_actions[:,:,:,1], vocab_size, embedding_size)
        action_embeddings = tf.concat([tf.nn.embedding_lookup(relation_embedding, available_actions[:,:,:,0]), tf.nn.embedding_lookup(entity_embedding, available_actions[:,:,:,1])], axis=-1)
        mlp_input = output #tf.concat([output, current_entity, query_embedding], axis=-1)
        hidden = tf.layers.dense(mlp_input, mlp_hidden_size, activation=tf.nn.relu)
        next_action_embedding = tf.layers.dense(hidden, 2 * embedding_size, activation=tf.nn.relu)

        expanded = tf.expand_dims(next_action_embedding, axis=2)
        logit = tf.reduce_sum(tf.multiply(action_embeddings, expanded), axis=3)

        # # Masking PAD actions
        # comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        # mask = tf.equal(next_relations, comparison_tensor)  # The mask
        # dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        # logit = tf.where(mask, dummy_scores, prelim_scores) 
    
    return logit, final_state

# def encoder_network(relation, entity, vocab_size, embedding_size):
#     with tf.variable_scope("embedding"):
#         entity_embedding = tf.get_variable("entity_embedding", [vocab_size[0], embedding_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
#         relation_embedding = tf.get_variable("relation_embedding", [vocab_size[1], embedding_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
#         embedding = tf.concat([tf.nn.embedding_lookup(relation_embedding, relation),
#                            tf.nn.embedding_lookup(entity_embedding, entity)], axis=-1)
#     return embedding
