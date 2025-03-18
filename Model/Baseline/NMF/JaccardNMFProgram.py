import os
import sys
import time
import argparse
import numpy as np
from pandas.io.parsers import read_csv
# from Measures import Measures
import time
from utils import *
from JacNMFModelNew import *
from visualise_results import *

# Create a custom class that duplicates output to both sys.stdout and the log file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)

######################## model configs and input paths #################
input_path = '../../../Datasets/model_input_data/'
start = time.time()
parser = argparse.ArgumentParser()
# parser.add_argument('--cleaned_file', default=input_path+'cleaned_tweets.csv', help='input text file')
parser.add_argument('--doc_sim_file', default=input_path+'doc_doc_t_jaccard_2018_2022.txt', help='document similarity file')
parser.add_argument('--corpus_file', default=input_path+'content_count_2018_2022.txt', help='term document matrix file')
# parser.add_argument('--vocab_file', default=input_path+'vocab.txt', help='vocab file')
parser.add_argument('--max_iter', type=int, default=500, help='max number of iterations')
parser.add_argument('--n_topics', type=int, default=64, help='number of topics')
parser.add_argument('--alpha', type=float, default=1, help='alpha')
parser.add_argument('--max_err', type=float, default=0.1, help='stop criterion')
parser.add_argument('--fix_seed', type=bool, default=True, help='set random seed 0')
args = parser.parse_args()

# Define the path to the log file and open the log file
log_file_path = f'./results/2018_2022/{args.n_topics}_log.txt'
log_file = open(log_file_path, 'a')

# Duplicate the output to both sys.stdout and the log file
sys.stdout = Tee(sys.stdout, log_file)

# read files
doc_dis = np.loadtxt(args.doc_sim_file)
dt_mat = np.transpose(np.loadtxt(args.corpus_file))
print('Shape of doc-doc similarity matrix: ', doc_dis.shape)
print('Shape of term document matrix: ', dt_mat.shape)

####################### show configs #################################

print('******************************************************')
print('#documents:', doc_dis.shape[0])
print('#tokens:', dt_mat.shape[0])

print('max number of iterations:', args.max_iter)
print('#topics:', args.n_topics)
print('alpha:', args.alpha)
print('stop criterion:', args.max_err)
print('******************************************************')

tmp_folder = 'results/2018_2022/'
if not os.access(tmp_folder, os.F_OK):
    os.mkdir(tmp_folder)

####################### run model and save results #################################

model = JacNMFModel(
    dt_mat, doc_dis,
    alpha=args.alpha,
    n_topic=args.n_topics,
    max_iter=args.max_iter,
    max_err=args.max_err,
    fix_seed=args.fix_seed)

model.save_format(
    H1file=tmp_folder + f'/{args.n_topics}_H1.txt',
    H2file=tmp_folder + f'/{args.n_topics}_H2.txt',
    Wfile=tmp_folder + f'/{args.n_topics}_W.txt')

H1, H2, W = model.get_lowrank_matrix()

####################### visualise results #################################
configure_params(args.n_topics)
save_keywords(n_top_words=10)
save_tsne_doc_embed()
word_topic_heatmap()
doc_topic_heatmap()
eval_topic_coherence()

# Close the log file
log_file.close()

# After capturing the output, sys.stdout will be restored to the original value
sys.stdout = sys.__stdout__