import numpy as np
import pickle as pkl
from scipy import sparse

input_path = '../../../Datasets/model_input_data/'

class Data():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()

    def parse_args(self, args):
        self.degree = args.degree
        self.minibatch_size = args.minibatch_size
        self.trans_induc = args.trans_induc
        self.num_hop = args.num_hop
        self.adj_cut_off = args.adj_cut_off
        self.sim_metric = args.doc_sim_metric
        if self.trans_induc == 'transductive':
            self.training_ratio = 1
        elif self.trans_induc == 'inductive':
            self.training_ratio = args.training_ratio - args.training_ratio * 0.1
            self.validation_ratio = args.training_ratio * 0.1
            self.test_ratio = 1 - args.training_ratio

    def load_data(self):

        # read input matrices
        doc = self.doc_preprocessing(np.loadtxt(input_path + 'content_tfidf.txt')) # document-word tfidf weight matrix X
        # retrieve doc similarity matrix
        if self.sim_metric == 'cosine_tfidf':
            doc_doc_sim = np.loadtxt(input_path + 'doc_doc_sim_tfidf.txt')
        elif self.sim_metric == 'cosine_count':
            doc_doc_sim = np.loadtxt(input_path + 'doc_doc_sim_count.txt')
        else:
            doc_doc_sim = np.loadtxt(input_path + 'doc_doc_sim_jaccard.txt')
        # create adjacency matrix based on cut-off threshold
        input_adj_matrix = np.where(doc_doc_sim > self.adj_cut_off, 1, 0)

        self.input_adj = self.generate_symmetric_adjacency_matrix(input_adj_matrix) # input 0-1 adjacency matrix A
        adj_mat = self.normalize_adj(self.input_adj)  # normalised adjacency matrix A^
        with open(input_path + 'word_word.ppmi', 'rb') as f:
            input_pmi = pkl.load(f)
            self.input_pmi = input_pmi.toarray() # word-word matrix C
        pmi = self.normalize_adj(self.input_pmi) # normalised w-w matrix C^
        voc = np.genfromtxt(input_path + 'vocab.txt', dtype=str)
        self.num_tokens = len(voc)
        self.num_doc = len(doc)

        # compute low-pass filter to get filtered document content X^
        filt = self.sgc_precompute(adj_mat, self.num_hop)   # l = 2
        self.filtered_doc = np.dot(filt, doc)    # X^

        # precompute simple graph convolution with predefined layers of message passing (degree)
        self.sgc_pre_adj = self.sgc_precompute(adj_mat, self.degree) 
        self.sgc_pre_pmi = self.sgc_precompute(pmi, self.degree)
        self.sgc_pre_doc = self.filtered_doc #self.sgc_precompute(self.filtered_doc, self.degree)
        
        # if self.trans_induc == 'transductive':
        #     self.input_training, self.input_test = self.doc, self.doc
        #     self.adj_training, self.adj_test = self.adjacency_matrix, self.adjacency_matrix
        #     self.adj_feature_training, self.adj_feature_test = np.dot(self.filt, self.doc), np.dot(self.filt, self.doc) # filtered_doc
        #     self.adj_input_training, self.adj_input_test = np.dot(self.sgc_pre_adj, self.doc), np.dot(self.sgc_pre_adj, self.doc)
        #     # self.num_train = len(self.input_training)
        #     # self.num_test = len(self.input_test)

        # elif self.trans_induc == 'inductive':
        #     self.input_training, self.input_test = self.doc[:int(self.num_doc * self.training_ratio)], \
        #                                    self.doc[int(self.num_doc * (self.training_ratio + self.validation_ratio)):]
        #     self.num_train = len(self.input_training)
        #     self.num_test = len(self.input_test)
        #     self.adj_training, self.adj_test = self.adjacency_matrix[:int(self.num_doc * self.training_ratio)], self.adjacency_matrix[int(self.num_doc * (self.training_ratio + self.validation_ratio)):]
        #     self.adj_feature_training, self.adj_feature_test = \
        #             np.dot(self.filt[:int(self.num_doc * self.training_ratio),:int(self.num_doc * self.training_ratio)], self.doc[:int(self.num_doc * self.training_ratio)]), np.dot(self.filt[int(self.num_doc * (self.training_ratio + self.validation_ratio)):],self.doc)
        #     self.adj_input_training, self.adj_input_test = \
        #             np.dot(self.sgc_pre_adj[:int(self.num_doc * self.training_ratio),:int(self.num_doc * self.training_ratio)], self.doc[:int(self.num_doc * self.training_ratio)]), np.dot(self.sgc_pre_adj[int(self.num_doc * (self.training_ratio + self.validation_ratio)):],self.doc)

    def doc_preprocessing(self, doc):

        doc_preprocessed = []
        for row in doc:
            max_row = np.log(1 + np.max(row))
            doc_preprocessed.append(np.log(1 + row) / max_row)

        return np.asarray(doc_preprocessed)

    def sgc_precompute(self, adj, degree):
        adj_degree = adj.copy()
        for i in range(degree-1):
             adj_degree = np.dot(adj, adj_degree)
        return adj_degree

    def generate_symmetric_adjacency_matrix(self, adjacency_matrix):

        adjacency_matrix_symm = np.zeros((len(adjacency_matrix), len(adjacency_matrix)))
        for row_idx in range(len(adjacency_matrix)):
            for col_idx in range(len(adjacency_matrix)):
                if adjacency_matrix[row_idx, col_idx] == 1:
                    adjacency_matrix_symm[row_idx, col_idx] = 1
                    adjacency_matrix_symm[col_idx, row_idx] = 1
                if row_idx == col_idx:
                    adjacency_matrix_symm[row_idx, col_idx] = 1

        return adjacency_matrix_symm

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))  # D-degree matrix
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return np.dot(np.dot(adj, d_mat_inv_sqrt).transpose(), d_mat_inv_sqrt)

    def adj_to_bias(adj, sizes, nhood=1):
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])
            for _ in range(nhood):
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        return -1e9 * (1.0 - mt)
