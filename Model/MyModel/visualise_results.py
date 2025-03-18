from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
import pickle as pkl

f_path = 'results/2018_2022/'
vocab_path = '../../Datasets/model_input_data/vocab_2018_2022.txt'
npmi_path = '../../Datasets/model_input_data/word_word_2018_2022.npmi'

def configure_params(args):
    global n_topic, n_epochs, trans_induc, n_degree, file_prefix
    n_topic = args.num_topics
    n_epochs = args.num_epoch
    trans_induc = args.trans_induc
    n_degree = args.degree
    n_hop = args.num_hop
    sim_metric = args.doc_sim_metric
    adj_cut_off = args.adj_cut_off
    file_prefix = f'{n_topic}_{n_epochs}_{trans_induc}_{n_degree}_{n_hop}_{sim_metric}_{adj_cut_off}'

################################ Top Topical Keywords ################################
def print_top_words(beta, feature_names, n_top_words=10):
    with open(f_path + file_prefix + '_keywords.txt', 'w') as file:
        for i in range(len(beta)):
            file.write(" ".join([feature_names[j]
                for j in beta[i].argsort()[:-n_top_words - 1:-1]]) + "\n")
    print("Saved topic keywords!")

def save_keywords(n_top_words):
    with open(f_path + file_prefix + '_train_voc.txt', 'r') as f:
        embedding_lines = f.readlines()

    voc = np.genfromtxt(vocab_path, dtype=str)
    docs = []
    vocab = {}

    for i in range(len(voc)):
        vocab[voc[i]] = i

    for i in range(len(embedding_lines)):
        emb_str = embedding_lines[i].strip().split()
        values = [float(x) for x in emb_str]
        docs.append(values)
    docs = np.array(docs)
    emb = np.array(list(zip(*docs)))

    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0], n_top_words=n_top_words)

################################ TSNE Doc Embedding ################################
def save_tsne_doc_embed():
    if trans_induc == 'transductive':
        run_sets = ['train']
    else:
        run_sets = ['train', 'test']
        
    for set_ in run_sets:
        with open(f_path + file_prefix + f'_{set_}_doc.txt', 'r') as f:
            t_embedding_lines = f.readlines()

        docs = []
        for i in range(len(t_embedding_lines)):
            emb_str = t_embedding_lines[i].strip().split()
            values = [float(x) for x in emb_str]
            docs.append(values)

        fea = TSNE(n_components=2).fit_transform(docs)

        # Create a scatter plot of the t-SNE embedding
        plt.figure(figsize=(7, 8)) 
        plt.scatter(fea[:, 0], fea[:, 1], s=14, alpha=0.3, color='purple')

        # Add labels and a title
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f't-SNE Embedding - {fea.shape[0]} {set_} Samples')
        plt.savefig(f_path + file_prefix + f'_{set_}_TSNE.png')
        # plt.show()

    print("Saved TSNE document embeddings!")

############################## Topic-Word matrix ##################################

def word_topic_heatmap():
    words = []
    with open(f_path + file_prefix + '_train_voc.txt', 'r') as f:
            t_embedding_lines = f.readlines()
            for i in range(len(t_embedding_lines)):
                emb_str = t_embedding_lines[i].strip().split()
                values = [float(x) for x in emb_str]
                words.append(values)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.9)
    heatmap = sns.heatmap(np.transpose(words), vmin=-1, vmax=1, cmap='BrBG')
    heatmap.set_title('Topic-Word matrix', fontsize=15)

    x_labels = np.arange(0, len(words), 1000)  # X-axis labels by increments
    plt.xticks(x_labels, x_labels, rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Vocabulary', fontsize=14, labelpad=10)
    plt.ylabel('Topics', fontsize=14, labelpad=10)

    plt.savefig(f_path + file_prefix + f'_w_t_heatmap.png', dpi=300, bbox_inches='tight')

    print("Saved word-topic matrix heatmap!")


############################## Document-Word matrix ###############################
def doc_topic_heatmap():
    docs = []
    with open(f_path + file_prefix + '_train_doc.txt', 'r') as f:
            t_embedding_lines = f.readlines()
            for i in range(len(t_embedding_lines)):
                emb_str = t_embedding_lines[i].strip().split()
                values = [float(x) for x in emb_str]
                docs.append(values)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.9)
    heatmap = sns.heatmap(np.transpose(docs), vmin=-1, vmax=1, cmap='BrBG')
    heatmap.set_title('Topic-Document matrix', fontsize=15)

    x_labels = np.arange(0, len(docs), 500)  # X-axis labels by increments
    plt.xticks(x_labels, x_labels, rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Documents', fontsize=14, labelpad=10)
    plt.ylabel('Topics', fontsize=14, labelpad=10)

    plt.savefig(f_path + file_prefix + f'_d_t_heatmap.png', dpi=300, bbox_inches='tight')

    print("Saved doc-topic matrix heatmap!")

############################# Topic Coherence Score ################################  
# Function to calculate pairwise topic coherence
def pairwise_coherence(topic, npmi_matrix, vocabulary):
    coherence = 0.0
    word_pairs = list(combinations(topic, 2))  # Generate all word pairs in the topic
    for word_pair in word_pairs:
        w1, w2 = word_pair
        w1_i = vocabulary.index(w1)
        w2_i = vocabulary.index(w2)
        coherence += npmi_matrix[w1_i, w2_i]
    return coherence/len(word_pairs)

def eval_topic_coherence():
    # Load your word-occurrence matrix (NPMI scores)
    with open(npmi_path, 'rb') as f:
        input_pmi = pkl.load(f)
        npmi_matrix = input_pmi.toarray()

    # load keywords
    with open(f_path + file_prefix + '_keywords.txt', 'r') as file:
        topics = [line.strip().split() for line in file.readlines()]

    # Read the vocabulary from the file
    with open(vocab_path, "r") as file:
        vocab = [line.strip() for line in file]

    with open(f_path + file_prefix + '_topic_score.txt', 'w') as file:
        # Function to calculate topic coherence for a given list of topics
        t_c = [] # individual coherence score per topic
        def calculate_topic_coherence(topics, npmi_matrix):
            total_coherence = 0.0
            for i, topic in enumerate(topics):
                coherence = round(pairwise_coherence(topic, npmi_matrix, vocab), 2)
                file.write(f"Topic {i}:" + str(coherence) + "\n")
                t_c.append(coherence)
                total_coherence += coherence
            return total_coherence

        # Calculate topic coherence
        total_coherence = calculate_topic_coherence(topics, npmi_matrix)
        avg_coherence = np.mean(t_c)
        median_coherence = np.median(sorted(t_c))
        file.write('\n'+ '='*50 + "\n")
        file.write("Total Topic Coherence Score: %.2f" % total_coherence + "\n")
        file.write("Average Topic Coherence Score: %.2f" % avg_coherence + "\n")
        file.write("Median Topic Coherence Score: %.2f" % median_coherence+ "\n") 
    
    print("Saved topic coherence score!")