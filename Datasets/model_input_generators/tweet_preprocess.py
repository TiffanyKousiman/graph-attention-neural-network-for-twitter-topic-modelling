import re
import string
from emot.emo_unicode import EMOTICONS_EMO
from stemming.porter2 import stem 
# nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

# create a new stopwords list from nltk and wordcloud
combined_stopwords = list(set(list(STOPWORDS) + stopwords.words('english')))

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Miscellaneous Symbols
                               u"\U000024C2-\U0001F251"  # Enclosed Characters
                               u"\U0001f926-\U0001f937"  # Additional emojis
                               u"\U00010000-\U0010ffff"  # Supplementary Planes
                               u"\u2640-\u2642"          # Gender Symbols
                               u"\u2600-\u2B55"          # Miscellaneous Symbols
                               u"\u200d"                 # Zero Width Joiner
                               u"\u23cf"                 # Eject Button
                               u"\u23e9"                 # Black Right-Pointing Double Triangle
                               u"\u231a"                 # Watch
                               u"\ufe0f"                 # Variation Selector-16 (used with emojis)
                               u"\u3030"                 # Wavy Dash
                               u"\u23F0"                 # Clock emoji
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emoticons(text):
    """remove detected emoticons from text"""
    emoticon_pattern = re.compile(u'(' + u'|'.join(re.escape(k) for k in EMOTICONS_EMO.keys()) + u')')
    return emoticon_pattern.sub(r'', text)

def remove_spaces(text):
    """removing trailing, leading spaces and extra spaces in text body."""
    text = text.strip()
    text = text.split()
    return ' '.join(text)

# remove punctuations and digits
def remove_punctuations_digits(text):
    """remove digits and punctuations"""
    text = re.sub(r'[‘’“”]','',text)
    text = text.translate(str.maketrans('', '', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # remove digits and punctuation marks
    return text

def remove_stopwords(text, stopwords):
    """remove any stopwords"""
    return ' '.join([word for word in text.split() if word not in stopwords])

def remove_short_words(text):
    """remove words with fewer than 3 characters of length from input text."""
    return ' '.join([word for word in text.split() if (len(word) > 2) | (word == 'ev')])

def remove_non_english_words(text):
    """filter any non-english words in the text"""
    non_eng_pattern = re.compile(r"[^\x00-\x7F]+")
    return non_eng_pattern.sub(r'', text)
    # return ' '.join([word for word in text.split() if detect(word) not in ['zh-cn','zh-tw', 'ar', 'ko', 'ja']]) # 学中文 조선글 سعادة ひらがな"

def stem_words(text):
    return ' '.join([stem(word) for word in text.split()])

def text_preprocessing(text):
    text = remove_emoji(text)
    text = remove_emoticons(text)
    text = remove_non_english_words(text)
    text = str(text).lower() # convert all words to lower case
    text = remove_stopwords(text, combined_stopwords)
    text = remove_punctuations_digits(text)
    text = remove_short_words(text)
    text = remove_spaces(text)
    # text = stem_words(text) # do not perform stemming before topic model
    return text

