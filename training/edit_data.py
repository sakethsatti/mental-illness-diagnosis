import pandas as pd
import os
import re
from concurrent.futures import ProcessPoolExecutor

# Define patterns
USER_PATTERN = re.compile(r'@\w+')
SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ]') # Include spanish characters
REPEAT_CHARS = re.compile(r'(\w)\1{2,}')
REPEAT_NON_WORD = re.compile(r'(\W)\1{2,}')
EXTRA_SPACES = re.compile(r'\s+')

# Define file prefixes in a list
file_prefixes = [
    "../English/Adhd_eng/",
    "../English/Anxiety_eng/",
    "../English/Asd_eng/",
    "../English/Bipolar_eng/",
    "../English/Control_eng/",
    "../English/Depression_eng/",
    "../English/Eating_eng/",
    "../English/Ocd_eng/",
    "../English/Ptsd_eng/",
    "../English/Schizophrenia_eng/",
    "../Spanish/Adhd_esp/",
    "../Spanish/Anxiety_esp/",
    "../Spanish/Asd_esp/",
    "../Spanish/Bipolar_esp/",
    "../Spanish/Control_esp/",
    "../Spanish/Depression_esp/",
    "../Spanish/Eating_esp/",
    "../Spanish/Ocd_esp/",
    "../Spanish/Ptsd_esp/",
    "../Spanish/Schizophrenia_esp/"
]

dataframes = []

# Loop through each prefix and read CSV files
for prefix in file_prefixes:
    # Determine the language based on the prefix
    language = 'English' if 'English' in prefix else 'Spanish'
    
    # Create full paths and read CSV files into DataFrames
    full_paths = [os.path.join(prefix, file) for file in os.listdir(prefix) if file.endswith('.csv')]
    
    for file in full_paths:
        df = pd.read_csv(file)

        # This is for a bug where the classes are named differently in English and Spanish
        if (prefix == "../English/Eating_eng/"):
            df["class"] = "EATING"
        elif (prefix == "../English/Asd_eng/"):
            df["class"] = "ASD"

        df['language'] = language  # Add a new column for the language
        dataframes.append(df)  # Append the DataFrame to the list

# Concatenate all DataFrames into a single DataFrame
all_data = pd.concat(dataframes, ignore_index=True)

def clean_tweet(tweet):
    try:
        # Remove URLs and user mentions
        tweet = USER_PATTERN.sub('', tweet)
        
        # Convert to lowercase
        tweet = tweet.lower()
        
        tweet = re.sub('httpurl', '', tweet)

        # Remove special characters and repeating characters
        tweet = SPECIAL_CHARS.sub('', tweet)
        tweet = REPEAT_CHARS.sub(r'\1', tweet)
        tweet = REPEAT_NON_WORD.sub(r'\1', tweet)
        
        # Split into words
        words = tweet.split()
        
        # Check if tweet is too short
        if len(words) < 3:
            return None
            
        # Join words and clean up spaces
        tweet = EXTRA_SPACES.sub(' ', ' '.join(words)).strip()
        
        return tweet if tweet else None
        
    except Exception:
        return None
    
def process_tweets(tweets, num_workers=None):
    # Use optimal number of workers
    if num_workers is None:
        num_workers = min(32, os.cpu_count() + 4)
    
    # Process in larger chunks for better performance
    chunk_size = max(1000, len(tweets) // (num_workers * 2))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        cleaned_tweets = list(executor.map(clean_tweet, tweets, chunksize=chunk_size))
    
    return cleaned_tweets

cleaned_tweets = process_tweets(all_data['tweet'])

# Create new DataFrame with cleaned tweets and classes
initial_data = pd.DataFrame({
    'tweet': cleaned_tweets,
    'class': all_data['class'],
    'language': all_data['language'],
}).dropna().drop_duplicates()

# Balance the classes - make CONTROL the same size as ADHD
adhd_count = len(initial_data[initial_data['class'] == 'ADHD'])
control_data = initial_data[initial_data['class'] == 'CONTROL'].sample(n=adhd_count, random_state=42)
other_data = initial_data[initial_data['class'] != 'CONTROL']

# Combine the balanced data
final_data = pd.concat([other_data, control_data], ignore_index=True)

# Calculate and print class distribution percentages
class_counts = final_data['class'].value_counts()
total_samples = len(final_data)
print("\nClass distribution after balancing:")
for class_name, count in class_counts.items():
    percentage = (count / total_samples) * 100
    print(f"{class_name}: {count} samples ({percentage:.2f}%)")

print(f"\nTotal samples: {total_samples}")

final_data.to_csv('cleaned_tweets.csv', index=False)
