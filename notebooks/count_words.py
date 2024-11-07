import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation

def count_most_common(all_tweets):
    # Define the list of punctuation marks to remove
    remove_punctuation = list(punctuation)
    remove_punctuation.extend(["@user", "user", "httpurl", "``", "''", "'s", "n't", "'m", "'re", "'ve", "'ll", "'d"])

    # Convert all tweets to lowercase and remove punctuation
    posts = [tweet.lower() for tweet in all_tweets]

    # Tokenize the tweets into individual words
    tweets_tokenized = [nltk.word_tokenize(post) for post in posts]

    # Remove stop words and punctuation from the list of words
    stop_words = set(stopwords.words('english'))
    tweets_cleaned = [[word for word in tweet if word not in stop_words and word not in remove_punctuation] for tweet in tweets_tokenized]

    word_counts = Counter(word for tweet in tweets_cleaned for word in tweet)
    most_common_ten = word_counts.most_common(10)

    return most_common_ten