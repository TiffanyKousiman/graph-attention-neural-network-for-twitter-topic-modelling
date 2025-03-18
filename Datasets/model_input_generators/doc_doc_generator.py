import pandas as pd
import numpy as np
from tweet_preprocess import *
from sentence_transformers import SentenceTransformer, util

export_path = "../model_input_data/"
tweets_path = "../../Kishen_Tiffany/final_5000_all_features.csv"
embeddings_path = "preprocessed_tweets_embedding.txt" # presaved tweets embeddings from SentenceTransformer('all-mpnet-base-v2')
# doc_vectors_count = np.loadtxt(export_path + "content_count.txt")
# doc_vectors_tfidf = np.loadtxt(export_path + "content_tfidf.txt")
has_embeddings = True # load precomputed embeddings into memory, other compute embeddings in the same script 

# data preparation
tweets = pd.read_csv(tweets_path)
relTweets = tweets[tweets.relevant == 1].reset_index(drop=True)
relTweets_2013_2017 = relTweets[(relTweets.year >= 2013) & (relTweets.year <= 2017)].reset_index(drop=True)
relTweets_2018_2022 = relTweets[(relTweets.year > 2017)].reset_index(drop=True)

for relTweets, tag in zip([relTweets_2013_2017, relTweets_2018_2022], ['2013_2017', '2018_2022']): 
    doc_vectors_count = np.loadtxt(export_path + f"content_count_{tag}.txt")
    # create a subset of features useful for neighbourhood calculation
    sim_features = ['tweet_id', 'tweet_type', 'original_tweet_id', 'tweet_creation', 'hashtags', 'mentions']
    tweets_sim_features_df = relTweets[sim_features]
    # remove the trailing "|"
    tweets_sim_features_df['mentions'] = tweets_sim_features_df['mentions'].apply(lambda x: [] if pd.isna(x) else x[1:].split('|'))
    tweets_sim_features_df['hashtags'] = tweets_sim_features_df['hashtags'].apply(lambda x: [] if pd.isna(x) else x[1:].split('|'))

    # # load or process embeddings
    # if has_embeddings:
    #     # load tweets embeddings into memory
    #     embeddings_array = np.loadtxt(embeddings_path)
    #     print("Loaded tweets embeddings!")
    # else:
    #     # the following code is computationally intensive
    #     mpnet_transformer = SentenceTransformer('all-mpnet-base-v2')
    #     docs = relTweets['tweet_text'].apply(text_preprocessing).to_list()
    #     embeddings_array = [mpnet_transformer.encode(d) for d in docs]
    #     np.savetxt(embeddings_path, fmt='%e')
    #     print("Saved tweets embeddings!")

    # compute doc-doc similarity
    num_tweets = len(tweets_sim_features_df)
    act_matrix = np.zeros((num_tweets, num_tweets))
    po_matrix = np.zeros((num_tweets, num_tweets))
    t_matrix = np.zeros((num_tweets, num_tweets))
    # t_matrix_tfidf = np.zeros((num_tweets, num_tweets))
    t_matrix_count = np.zeros((num_tweets, num_tweets))
    # u_matrix = np.zeros((num_tweets, num_tweets))

    for i, ti in tweets_sim_features_df.iterrows():
        for j, tj in tweets_sim_features_df.iterrows():
            if (i < j): # ignore self comparison
                
                #########################################################################
                # calculate user action score act(RTP_ti ,RTP_tj)
                    # 1 if (RTP_ti = tj ) or (ti = RTP_tj) or (RTP_ti = RTP_tj and not (RTP_ti = RTP_tj = 0))
                    # 0 otherwise
                if (tj.original_tweet_id == ti.original_tweet_id == 0):
                    act_ij = 0
                elif (ti.original_tweet_id == tj.tweet_id) or (tj.original_tweet_id == ti.tweet_id) or (tj.original_tweet_id == ti.original_tweet_id):
                    print(ti.tweet_id, tj.tweet_id)
                    act_ij = 1
                else:
                    act_ij = 0

                act_matrix[i,j] = act_matrix[j,i] = act_ij

                #############################################################################
                # calculate people mention score po(ti, tj)
                    # po(i,j) = |pi ∩ pj|/|pi ∪ pj|

                pi = ti.mentions
                pj = tj.mentions
                common_p = list(set(pi) & set(pj))
                all_p = list(set(pi) | set(pj))

                try:
                    po_ij = len(common_p)/len(all_p)
                except ZeroDivisionError:
                    po_ij = 0
                
                po_matrix[i,j] = po_matrix[j,i] = po_ij
            
                #############################################################################
                # calculate the text semantic and lexical similarity using cosine similarity + pretrained encoder
                # t_matrix[i,j] = t_matrix[j,i] = util.cos_sim(embeddings_array[i], embeddings_array[j]).item()
                # t_matrix_tfidf[i,j] = t_matrix_tfidf[j,i] = util.cos_sim(doc_vectors_tfidf[i], doc_vectors_tfidf[j]).item()
                t_matrix_count[i,j] = t_matrix_count[j,i] = util.cos_sim(doc_vectors_count[i], doc_vectors_count[j]).item()

                ##############################################################################
                # # calculate the user description semantic and lexical similarity using cosine similarity + pretrained encoder

                # if pd.isna(ti.user_description) or pd.isna(tj.user_description):
                #     u_ij = 0
                # else: 
                #     u_ij = util.cos_sim(encoded_user_description_dict[ti['tweet_id']], encoded_user_description_dict[tj['tweet_id']]).item()

                # u_matrix[i,j] = u_matrix[j,i] = u_ij

    # compute final similarity score 
    # s_matrix_mpnet = act_matrix + po_matrix + t_matrix #+ u_matrix
    # s_matrix_tfidf = act_matrix + po_matrix + t_matrix_tfidf
    s_matrix_count = act_matrix + po_matrix + t_matrix_count

    def sigmoid_transform(arr, scale=1.0):
        """
        Apply a sigmoid transformation to a 2D array of values, leaving 0 values unchanged.
        
        Parameters:
        - arr: A 2D NumPy array
        - scale: A scaling factor for the sigmoid function (default is 1.0).
        
        Returns:
        - The transformed 2D array with 0 values unchanged.
        """
        # Create a copy of the input array to store the transformed values
        transformed_arr = arr.copy()
        
        # Apply the sigmoid transformation to non-zero values
        non_zero_indices = arr != 0
        transformed_arr[non_zero_indices] = 1 / (1 + np.exp(-arr[non_zero_indices] * scale))
        
        return transformed_arr

    # sigmoid transformed s_matrix
    # sig_s_matrix_mpnet = sigmoid_transform(s_matrix_mpnet)
    # sig_s_matrix_tfidf = sigmoid_transform(s_matrix_tfidf)
    sig_s_matrix_count = sigmoid_transform(s_matrix_count)

    # create binary adjacency matrix where non-zero values are encoded as 1
    # adjacency_matrix = np.where(sig_s_matrix_tfidf != 0, 1, 0)

    # write all computed matrices
    # save computed arrays
    np.savetxt(export_path + f'doc_doc_act_{tag}.txt', act_matrix, fmt='%0.2f')
    np.savetxt(export_path + f'doc_doc_po_{tag}.txt', po_matrix, fmt='%0.2f')
    # np.savetxt(export_path + f'doc_doc_t_mpnet.txt', t_matrix, fmt='%f')
    # np.savetxt(export_path + f'doc_doc_t_tfidf_{tag}.txt', t_matrix_tfidf, fmt='%f')
    np.savetxt(export_path + f'doc_doc_t_count_{tag}.txt', t_matrix_count, fmt='%f')
    # np.savetxt(export_path + f'doc_doc_sim_mpnet.txt', sig_s_matrix_mpnet, fmt='%f')
    # np.savetxt(export_path + f'doc_doc_sim_tfidf.txt', sig_s_matrix_tfidf, fmt='%f')
    np.savetxt(export_path + f'doc_doc_sim_count_{tag}.txt', sig_s_matrix_count, fmt='%f')
    # np.savetxt(export_path + f'doc_doc_adjacency_matrix.txt', adjacency_matrix, fmt='%d')

    print("Saved doc-doc similarity vectors!")