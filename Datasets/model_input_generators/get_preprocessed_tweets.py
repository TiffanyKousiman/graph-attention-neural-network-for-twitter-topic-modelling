from tweet_preprocess import *
import pandas as pd
import numpy as np

export_path = "../model_input_data/"
tweets_path = "../../Kishen_Tiffany/final_5000_all_features.csv"

# read original tweets data
tweets = pd.read_csv(tweets_path)

# only get the relevant subset and create a list of preprocessed text
relTweets = tweets[tweets.relevant == 1].reset_index()
relTweets['tweet_text_preprocessed'] = relTweets['tweet_text'].apply(text_preprocessing)

relTweets_2013_2017 = relTweets[(relTweets.year >= 2013) & (relTweets.year <= 2017)].reset_index(drop=True)
relTweets_2018_2022 = relTweets[(relTweets.year > 2017)].reset_index(drop=True)

# export to txt file
relTweets_2013_2017['tweet_text_preprocessed'].to_csv(export_path + "cleaned_tweets_2013_2017.csv", index=False, header=None) 
relTweets_2018_2022['tweet_text_preprocessed'].to_csv(export_path + "cleaned_tweets_2018_2022.csv", index=False, header=None) 

print("Saved cleaned tweets to csv!")