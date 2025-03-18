import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tweet_preprocess import *

export_path = "../model_input_data/"
tweets_path = "../../Kishen_Tiffany/final_5000_all_features.csv"

# read original tweets data
tweets = pd.read_csv(tweets_path)

# only get the relevant subset and create a list of preprocessed text
relTweets = tweets[tweets.relevant == 1].reset_index()
relTweets['tweet_text_preprocessed'] = relTweets['tweet_text'].apply(text_preprocessing)

relTweets_2013_2017 = relTweets[(relTweets.year >= 2013) & (relTweets.year <= 2017)].reset_index(drop=True)
relTweets_2018_2022 = relTweets[(relTweets.year > 2017)].reset_index(drop=True)

for relTweets, tag in zip([relTweets_2013_2017, relTweets_2018_2022], ['2013_2017', '2018_2022']):
    docs = relTweets['tweet_text_preprocessed'].to_list()

    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english', # do extra stop word removal if there's any
        analyzer='word'
    )
    count_vectorizer  = CountVectorizer(
        stop_words='english', # do extra stop word removal if there's any
        analyzer='word'
    )

    # Fit and transform the documents to create the TF-IDF and word occurence matrix
    tfidf_wm = tfidf_vectorizer.fit_transform(docs)
    count_wm = count_vectorizer.fit_transform(docs)

    # Convert the matrices into a dense NumPy array if needed
    tfidf_wm_array = tfidf_wm.toarray()
    count_wm_array = count_wm.toarray()

    # retrieves the terms found in the corpora (vocabulary)
    vocabs = tfidf_vectorizer.get_feature_names_out() 

    # export vocab
    with open(export_path + f"vocab_{tag}.txt", 'w') as file:
        for voc in vocabs:
            file.write(str(voc) + "\n")
    print(f"Saved vocabs!")

    # export document ids
    with open(export_path + f"docs_{tag}.txt", 'w') as file:
        for id_ in relTweets['tweet_id']:
            file.write(str(id_) + "\n")
    print(f"Saved document ids!")

    # export doc-word vectors
    np.savetxt(export_path + f"content_tfidf_{tag}.txt", tfidf_wm_array, fmt='%f')
    np.savetxt(export_path + f"content_count_{tag}.txt", count_wm_array, fmt='%d') 
    print(f"Saved doc-word vectors!")
