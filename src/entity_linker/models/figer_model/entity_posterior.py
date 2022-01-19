"""
Modifications copyright (C) 2020 Michael Strobl
"""

import time
import tensorflow as tf
import numpy as np

from entity_linker.models.base import Model

class EntityPosterior(Model):
    """Entity Embeddings and Posterior Calculation"""

    def __init__(self, batch_size, num_knwn_entities, context_encoded_dim,
                 context_encoded, entity_ids, scope_name,
                 device_embeds, device_gpu):

        ''' Defines the entity posterior estimation graph.
          Makes entity embeddings variable, gets encoded context and candidate
          entities and gets entity scores using dot-prod.

          Input :
            context_encoded_dim: D - dims in which context is encoded
            context_encoded: [B, D]
            entity_ids: [B, N]. If supervised, first is the correct entity, and
              rest N-1 the candidates. For unsupervised, all N are candidates.
            num_knwn_entities: Number of entities with supervised data.
          Defines:
            entity_embeddings: [num_knwn_entities, D]
          Output:
            entity_scores: [B,N] matrix of context scores
            '''

        self.batch_size = batch_size
        self.num_knwn_entities = num_knwn_entities

        with tf.compat.v1.variable_scope(scope_name) as s:
            with tf.device(device_embeds) as d:
                self.knwn_entity_embeddings = tf.compat.v1.get_variable(
                  name="known_entity_embeddings",
                  shape=[self.num_knwn_entities, context_encoded_dim],
                  initializer=tf.random_normal_initializer(mean=0.0,
                                                           stddev=1.0/(100.0)))
            with tf.device(device_gpu) as g:
                # [B, N, D]
                self.sampled_entity_embeddings = tf.compat.v1.nn.embedding_lookup(
                    self.knwn_entity_embeddings, entity_ids)

                # # Negative Samples for Description CNN - [B, D]
                trueentity_embeddings = tf.compat.v1.slice(
                  self.sampled_entity_embeddings, [0,0,0],
                  [self.batch_size, 1, context_encoded_dim])
                self.trueentity_embeddings = tf.compat.v1.reshape(
                    trueentity_embeddings, [self.batch_size,
                                            context_encoded_dim])

                # Negative Samples for Description CNN
                negentity_embeddings = tf.compat.v1.slice(
                    self.sampled_entity_embeddings, [0,1,0],
                    [self.batch_size, 1, context_encoded_dim])
                self.negentity_embeddings = tf.compat.v1.reshape(
                    negentity_embeddings, [self.batch_size,
                                           context_encoded_dim])

                # [B, 1, D]
                context_encoded_expanded = tf.compat.v1.expand_dims(
                  input=context_encoded, dim=1)

                # [B, N]
                self.entity_scores = tf.compat.v1.reduce_sum(tf.multiply(
                  self.sampled_entity_embeddings, context_encoded_expanded), 2)

                # SOFTMAX
                # [B, N]
                self.entity_posteriors = tf.compat.v1.nn.softmax(
                    self.entity_scores, name="entity_post_softmax")

    def loss_graph(self, true_entity_ids, scope_name, device_gpu):
        ''' true_entity_ids : [B] is the true ids in the sampled [B,N] matrix
          In entity_ids, [?, 0] is the true entity therefore  this should be
          a vector of zeros
        '''
        with tf.compat.v1.variable_scope(scope_name) as s, tf.device(device_gpu) as d:
            # CROSS ENTROPY LOSS
            self.crossentropy_losses = \
                tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.entity_scores,
                    labels=true_entity_ids,
                    name="entity_posterior_loss")

            self.posterior_loss = tf.compat.v1.reduce_sum(
                self.crossentropy_losses) / tf.compat.v1.to_float(self.batch_size)
