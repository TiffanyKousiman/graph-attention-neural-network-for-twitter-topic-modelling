import argparse
import numpy as np
import tensorflow as tf
import os
from data_preparation import Data
from GCT import GTM
from visualise_results import *
import sys

# Create a custom class that duplicates output to both sys.stdout and the log file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025)
    parser.add_argument('-ne', '--num_epoch', type=int, default=1000) #15000
    parser.add_argument('-ti', '--trans_induc', type=str, default='transductive', help='transductive or inductive, transductive means we input all documents and links for unsupervised training, inductive means we split 80% for training, 20% for test')
    parser.add_argument('-nt', '--num_topics', type=int, default=64)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8, help='This program will automatically split 10% among training set for validation')
    parser.add_argument('-ms', '--minibatch_size', type=int, default=128)
    # parser.add_argument('-weight', '--loss_weight', type=float, default=1.0)
    parser.add_argument('-rs', '--random_seed', type=int, default=950)
    parser.add_argument('-lhop', '--num_hop', type=int, default=2, help='number of hops for multi-hop information diffusion process across doc-doc network')
    parser.add_argument('-degree', '--degree', type=int, default=2)
    parser.add_argument('-co', '--adj_cut_off', type=float, default=0, help='cut off value for document connections')
    parser.add_argument('-sm', '--doc_sim_metric', type=str, default='cosine_tfidf', help='cosine_tfidf, cosine_count or jaccard, document similarity metric for creating document connections')

    return parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    if args.random_seed:
        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)

     # Define the path to the log file and open the log file
    log_file_path = f'./results/2018_2022/{args.num_topics}_{args.num_epoch}_{args.trans_induc}_{args.degree}_{args.num_hop}_{args.doc_sim_metric}_{args.adj_cut_off}_log.txt'
    log_file = open(log_file_path, 'a')

    # Duplicate the output to both sys.stdout and the log file
    sys.stdout = Tee(sys.stdout, log_file)

    #########################################################
    print('Preparing data...')
    data = Data(args)
    print('Initializing model...')
    model = GTM(args, data)
    print('Start training...')
    model.train()
    print('Visualising results...')
    configure_params(args)
    save_keywords(n_top_words=10)
    save_tsne_doc_embed()
    word_topic_heatmap()
    doc_topic_heatmap()
    eval_topic_coherence()
    #########################################################

    # Close the log file
    log_file.close()

    # After capturing the output, sys.stdout will be restored to the original value
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main(parse_arguments())
