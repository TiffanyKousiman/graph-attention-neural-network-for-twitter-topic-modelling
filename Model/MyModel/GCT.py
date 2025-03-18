import tensorflow as tf
import numpy as np
import time
from layer import attn_head

class GTM():
    def __init__(self, args, data):

        self.parse_args(args, data)
        self.show_config()
        self.generate_placeholders()
        self.generate_variables()

    def parse_args(self, args, data):

        self.data = data
        self.num_doc = self.data.num_doc
        self.tokens = self.data.num_tokens

        self.learning_rate = args.learning_rate
        self.num_epoch = args.num_epoch
        self.minibatch_size = args.minibatch_size
        self.num_topics = args.num_topics
        # self.weight = args.loss_weight
        self.trans_induc = args.trans_induc
        if self.trans_induc == 'transductive':
            self.training_ratio = 1
        else:
            self.training_ratio = args.training_ratio

    def show_config(self):

        print('******************************************************')
        print('#documents:', self.num_doc)
        print('#tokens:', self.data.num_tokens)
        print('learning rate:', self.learning_rate)
        print('training ratio:', self.training_ratio)
        print('minibatch size:', self.minibatch_size)
        print('#epoch:', self.num_epoch)
        print('#topics:', self.num_topics)
        print('transductive or inductive learning:', self.trans_induc)
        print('#hop diffusion:', self.data.num_hop)
        print('degree:', self.data.degree)
        print('doc similarity metric:', self.data.sim_metric)
        print('adjacency matrix cut-off value:', self.data.adj_cut_off)
        print('******************************************************')

    def generate_placeholders(self):

        # self.doc = tf.placeholder('float64', [None, self.data.num_tokens])
        # self.doc_voc = tf.placeholder('float64', [None, self.data.num_tokens])
        # self.pmi = tf.placeholder('float64', [self.data.num_tokens,self.data.num_tokens])
        # self.adj = tf.placeholder('float64', [None,self.data.num_doc])
        # self.sgc_pre_adj = tf.placeholder('float64', [None, self.data.num_tokens])
        self.input_adj = tf.placeholder('float64', [self.data.num_doc, self.data.num_doc])
        self.input_doc = tf.placeholder('float64', [self.data.num_doc, self.data.num_tokens])
        self.input_pmi = tf.placeholder('float64', [self.data.num_tokens,self.data.num_tokens])
        self.sgc_pre_pmi = tf.placeholder('float64', [self.data.num_tokens, self.data.num_tokens])
        self.sgc_pre_doc = tf.placeholder('float64', [self.data.num_doc, self.data.num_tokens])
        self.sgc_pre_adj = tf.placeholder('float64', [self.data.num_doc, self.data.num_doc])
        self.dis = tf.placeholder('float64', [self.data.num_doc, self.data.num_doc])

    def generate_variables(self):

        self.weights = {
            'encoder_d': tf.Variable(tf.random_normal([self.data.num_doc, self.num_topics], dtype='float64'), dtype='float64'),
            'encoder_v': tf.Variable(tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'), dtype='float64'),
            'inter_encoder_d': tf.Variable(tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'), dtype='float64'),
            'inter_encoder_v': tf.Variable(tf.random_normal([self.data.num_doc, self.num_topics], dtype='float64'), dtype='float64'),
            # 'encoder_d': tf.Variable(tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'), dtype='float64'),
            # 'encoder_v': tf.Variable(
            #     tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'),
            #     dtype='float64'),
            # 'decoder_d': tf.Variable(tf.random_normal([self.data.num_doc, self.num_topics], dtype='float64'), dtype='float64'),
            # 'inter_encoder_d': tf.Variable(tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'),
            #                          dtype='float64'),
            # 'inter_encoder_v': tf.Variable(
            #     tf.random_normal([self.data.num_tokens, self.num_topics], dtype='float64'),
            #     dtype='float64'),
        }
        self.biases = {
            'encoder_b_d': tf.Variable(tf.random_normal([self.num_topics], dtype='float64'), dtype='float64'),
            'encoder_b_v': tf.Variable(tf.random_normal([self.num_topics], dtype='float64'), dtype='float64'),
            'inter_encoder_b_d': tf.Variable(tf.random_normal([self.num_topics], dtype='float64'), dtype='float64'),
            'inter_encoder_b_v': tf.Variable(tf.random_normal([self.num_topics], dtype='float64'), dtype='float64'),
            'decoder_b_d_v': tf.Variable(tf.random_normal([self.data.num_tokens], dtype='float64'), dtype='float64'),
            'decoder_b_v_v': tf.Variable(tf.random_normal([self.data.num_tokens], dtype='float64'), dtype='float64'),
            'decoder_b_d_d': tf.Variable(tf.random_normal([self.data.num_doc], dtype='float64'), dtype='float64')

            # 'decoder_b_d_v': tf.Variable(tf.random_normal([len(self.data.input_training[0])], dtype='float64'), dtype='float64'),
            # 'decoder_b_v_v': tf.Variable(tf.random_normal([len(self.data.input_training[0])], dtype='float64'), dtype='float64'),
            # 'decoder_b_d_d': tf.Variable(tf.random_normal([self.data.num_doc], dtype='float64'), dtype='float64')
        }

    def encoder(self):
        # # weight for message passing
        # intra_doc_feature = tf.nn.tanh(tf.add(self.weights['encoder_d'], self.biases['encoder_b_d']))
        # intra_voc_feature = tf.nn.tanh(tf.add(self.weights['encoder_v'], self.biases['encoder_b_v']))
        # inter_doc_feature = tf.nn.tanh(tf.add(tf.matmul(self.doc,self.weights['inter_encoder_d']), self.biases['inter_encoder_b_d']))
        # inter_voc_feature = tf.nn.tanh(tf.add(self.weights['inter_encoder_v'], self.biases['inter_encoder_b_v']))
        
        # # intra-domain message passing
        # intra_doc_embedding = tf.nn.tanh(tf.matmul(self.sgc_pre_adj, intra_doc_feature))
        # intra_voc_embedding = tf.nn.tanh(tf.matmul(self.pmi, intra_voc_feature))
        # # inter-domain message passing
        # inter_doc_embedding = tf.nn.tanh(tf.matmul(self.doc_voc, inter_voc_feature))
        # inter_voc_embedding = tf.nn.tanh(tf.matmul(tf.transpose(self.doc_voc, perm=[1, 0]), inter_doc_feature))
        # self.doc_embed = intra_doc_embedding + inter_doc_embedding
        # self.voc_embed = intra_voc_embedding + inter_voc_embedding

        # intra-domain message passing
        intra_doc_embedding = tf.nn.tanh(tf.add(tf.matmul(self.sgc_pre_adj, self.weights['encoder_d']), self.biases['encoder_b_d']))
        intra_voc_embedding = tf.nn.tanh(tf.add(tf.matmul(self.sgc_pre_pmi, self.weights['encoder_v']), self.biases['encoder_b_v']))

        # inter-domain message passing
        inter_doc_embedding = tf.nn.tanh(tf.add(tf.matmul(self.sgc_pre_doc, self.weights['inter_encoder_d']), self.biases['inter_encoder_b_d']))
        inter_voc_embedding = tf.nn.tanh(tf.add(tf.matmul(tf.transpose(self.sgc_pre_doc, perm=[1, 0]), self.weights['inter_encoder_v']), self.biases['inter_encoder_b_v']))

        # compute final doc and voc embedding 
        doc_embed = intra_doc_embedding + inter_doc_embedding # doc-topic representation θd
        # add attention mechansim to document embedding
        self.doc_embed, _ = attn_head(doc_embed, doc_embed, self.dis, activation=tf.nn.tanh)
        self.voc_embed = intra_voc_embedding + inter_voc_embedding # word-topic representation βv

        return self.doc_embed, self.voc_embed

    def decoder(self):
        r_doc_voc = tf.add(tf.matmul(self.doc_embed, tf.transpose(self.voc_embed)), self.biases['decoder_b_d_v'])
        # r_doc_doc = tf.add(tf.matmul(self.doc_embed, tf.transpose(self.weights['decoder_d'])),self.biases['decoder_b_d_d'])
        r_doc_doc = tf.add(tf.matmul(self.doc_embed, tf.transpose(self.weights['encoder_d'])),self.biases['decoder_b_d_d'])
        r_voc_voc = tf.add(tf.matmul(self.voc_embed, tf.transpose(self.weights['encoder_v'])), self.biases['decoder_b_v_v'])
        dv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_doc_voc, labels=self.input_doc)) #L(x^,X ̃ )
        dd_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_doc_doc, labels=self.input_adj)) #L(A,A ̃ )
        vv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_voc_voc, labels=self.input_pmi)) #L(C,C ̃ )
        loss = dv_loss + dd_loss + vv_loss #self.weight*dv_loss + dd_loss + vv_loss
        return loss, dv_loss, dd_loss, vv_loss

    def construct_model(self):
        self.encoder()
        loss, dv_loss, dd_loss, vv_loss = self.decoder()

        return loss, dv_loss, dd_loss, vv_loss

    def train(self):

        loss, dv_loss, dd_loss, vv_loss= self.construct_model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            t = time.time()
            # best_acc = 0.0
            # best_nmi = 0.0
            for epoch_index in range(1, self.num_epoch + 1):
                # _, one_epoch_loss,o_dv_loss, o_dd_loss, o_vv_loss = sess.run([optimizer, loss, dv_loss, dd_loss, vv_loss], feed_dict={
                #                                                            self.doc: self.data.input_training,
                #                                                            self.doc_voc: self.data.adj_feature_training,
                #                                                            self.sgc_pre_adj:self.data.adj_input_training,
                #                                                            self.pmi:self.data.pmi,
                #                                                            self.adj:self.data.adj_training})

                _, one_epoch_loss,o_dv_loss, o_dd_loss, o_vv_loss = sess.run([optimizer, loss, dv_loss, dd_loss, vv_loss], feed_dict={
                                                                           self.sgc_pre_doc: self.data.sgc_pre_doc, 
                                                                           self.sgc_pre_adj: self.data.sgc_pre_adj,
                                                                           self.sgc_pre_pmi: self.data.sgc_pre_pmi,
                                                                           self.input_doc: self.data.filtered_doc, # reference d_v for reconstruction
                                                                           self.input_pmi: self.data.input_pmi, # reference v_v for reconstruction
                                                                           self.input_adj: self.data.input_adj, # reference d_d for reconstruction
                                                                           self.dis: self.data.dis
                                                                           })
                
                
                if epoch_index % 20 == 0 or epoch_index == 1:
                    print('******************************************************')
                    print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_index, self.num_epoch), '\tLoss: %f' % one_epoch_loss,
                          '\tdv_Loss: %f' % o_dv_loss, '\tdd_Loss: %f' % o_dd_loss, '\tvv_Loss: %f' % o_vv_loss)

                    doc_embed_training = sess.run(self.doc_embed, feed_dict={self.sgc_pre_doc: self.data.sgc_pre_doc, 
                                                                           self.sgc_pre_adj: self.data.sgc_pre_adj,
                                                                           self.sgc_pre_pmi: self.data.sgc_pre_pmi,
                                                                           self.dis: self.data.dis})
                                                #   {self.doc: self.data.input_training, self.doc_voc: self.data.adj_feature_training,
                                                #                         self.pmi:self.data.pmi, self.sgc_pre_adj:self.data.adj_input_training})
                    voc_embed_training = sess.run(self.voc_embed, feed_dict={self.sgc_pre_doc: self.data.sgc_pre_doc, 
                                                                           self.sgc_pre_adj: self.data.sgc_pre_adj,
                                                                           self.sgc_pre_pmi: self.data.sgc_pre_pmi})
                    
                    np.savetxt('./results/2018_2022/' + str(self.num_topics) + '_' + str(self.num_epoch) + '_' + \
                               str(self.trans_induc) +  '_' + str(self.data.degree) +  '_' + str(self.data.num_hop) +  '_' +\
                                  str(self.data.sim_metric) +  '_' + str(self.data.adj_cut_off) + '_train_doc' + '.txt', doc_embed_training, delimiter='\t')
                    np.savetxt('./results/2018_2022/' + str(self.num_topics) + '_' + str(self.num_epoch) + '_' + \
                               str(self.trans_induc) + '_' + str(self.data.degree) +  '_' + str(self.data.num_hop) +  '_' + \
                                str(self.data.sim_metric) +  '_' + str(self.data.adj_cut_off) + '_train_voc' + '.txt', voc_embed_training, delimiter='\t')

                    # if self.trans_induc == 'inductive':
                    #     doc_embed_test = sess.run(self.doc_embed, feed_dict={self.doc: self.data.input_test, self.doc_voc: self.data.adj_feature_test,
                    #                                                     self.pmi:self.data.pmi, self.sgc_pre_adj:self.data.adj_input_test})
                    #     voc_embed_test = sess.run(self.voc_embed,
                    #                             feed_dict={self.doc: self.data.input_test,
                    #                                         self.doc_voc: self.data.adj_feature_test,
                    #                                         self.pmi: self.data.pmi, self.sgc_pre_adj: self.data.adj_input_test})
                    
                    #     np.savetxt('./results/' + str(self.num_topics) + '_' + str(self.num_epoch) + '_' + \
                    #                str(self.trans_induc) +  '_' + str(self.data.degree) +  '_test_doc' + '.txt', doc_embed_test, delimiter='\t')
                    #     np.savetxt('./results/' + str(self.num_topics) + '_' + str(self.num_epoch) + '_' + \
                    #                str(self.trans_induc) +  '_' + str(self.data.degree) + '_test_voc' + '.txt', voc_embed_test, delimiter='\t')

                    # acc = classification_knn('inductive', X_train=doc_embed_training, X_test=doc_embed_test, Y_train=self.data.label_training, Y_test=self.data.label_test)
                    # nmi = clustering_kmeans('inductive', X_train=doc_embed_training, X_test=doc_embed_test, Y_train=self.data.label_training, Y_test=self.data.label_test)
                    # if best_acc < acc and best_nmi < nmi:
                    #     best_acc = acc
                    #     best_nmi = nmi
                    #     print('best_acc_nmi:', best_acc, best_nmi)

            print('Finish training! Training time:', time.time() - t)
            print('Finish saving embeddings!')
