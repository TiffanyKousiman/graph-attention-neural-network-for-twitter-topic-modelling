import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log

for tag in ['2013_2017', '2018_2022']:
    # parameters
    # word_vector_file = 'glove.6B.300d.txt'
    vocab_file = f'vocab_{tag}.txt'
    content_file = f'content_count_{tag}.txt'
    input_path = export_path = '../model_input_data/'
    window_size = 10

    # # load GloVe pretrained word embeddings into memory
    # def loadWord2Vec(filename):
    #     """Read Word Vectors"""
    #     #vocab = []
    #     #embd = []
    #     word_vector_map = {}
    #     file = open(filename, 'r', encoding="utf-8")
    #     for line in file.readlines():
    #         row = line.strip().split(' ')
    #         if (len(row) > 2):
    #             #vocab.append(row[0])
    #             vector = row[1:]
    #             length = len(vector)
    #             for i in range(length):
    #                 vector[i] = float(vector[i])
    #             #embd.append(vector)
    #             word_vector_map[row[0]] = vector
    #     print('Loaded Word Vectors!')
    #     file.close()
    #     return word_vector_map
    # word_vector_map = loadWord2Vec(word_vector_file)

    # # get the word embedding for each token in the vocabulary
    # word_embedding = []
    voc = np.genfromtxt(input_path + vocab_file, dtype=str)
    num_tokens = len(voc)
    # for i in range(num_tokens):
    #     if voc[i] in word_vector_map:
    #         word_embedding.append(word_vector_map[voc[i]])
    #     else:
    #         word_embedding.append(np.zeros(300))

    # get a list of word positions within the window size
    windows = []
    doc = np.loadtxt(input_path + content_file)
    for d in doc:
        word = []
        for i, w in enumerate(d):
            if 0 < w:
                word.append(i)
        length = len(word)
        if length < window_size:
            windows.append(word)
        else:
            for j in range(length - window_size + 1):
                window = word[j: j + window_size]
                windows.append(window)

    # get the word frequency over all word window lists
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # get word co-occurrence count in all word windows
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i_id = window[i]
                word_j_id = window[j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # compute PPMI (Pointwise Mutual Information)
    row = []
    col = []
    weight = []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        # npmi = pmi/-log((1.0 * count / num_window))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)

    # the row, col and weight arrays together specify the non-zero elements and their positions in the matrix
    # use sparse matrix format (CSR) to populate the entire word pairwise matrix including the zero cells
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(num_tokens, num_tokens))

    # write ppmi and word embedding into binary files
    f = open(export_path + f'word_word_{tag}.ppmi', 'wb')
    pkl.dump(adj, f)
    f.close()
    print("Saved word-word ppmi array!")

    # f = open(export_path + 'word_embeddings.word', 'wb')
    # pkl.dump(word_embedding, f)
    # f.close()
    # print("Saved word embedding array!")