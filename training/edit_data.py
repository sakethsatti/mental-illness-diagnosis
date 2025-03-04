import pandas as pd
import os
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Process Twitter data for mental illness classification.')
parser.add_argument('--language', type=str, choices=['english', 'spanish', 'both'], 
                    default='both', help='Choose which language data to process (default: both)')
args = parser.parse_args()

# Define patterns
USER_PATTERN = re.compile(r'@\w+')
SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ]') # Include spanish characters
REPEAT_CHARS = re.compile(r'(\w)\1{2,}')
REPEAT_NON_WORD = re.compile(r'(\W)\1{2,}')
EXTRA_SPACES = re.compile(r'\s+')

# Define file prefixes in a list
file_prefixes = []

# Add file prefixes based on selected language
if args.language.lower() in ['english', 'both']:
    file_prefixes.extend([
        "../English/Adhd_eng/",
        "../English/Anxiety_eng/",
        "../English/Asd_eng/",
        "../English/Bipolar_eng/",
        "../English/Control_eng/",
        "../English/Depression_eng/",
        "../English/Eating_eng/",
        "../English/Ocd_eng/",
        "../English/Ptsd_eng/",
        "../English/Schizophrenia_eng/"
    ])

if args.language.lower() in ['spanish', 'both']:
    file_prefixes.extend([
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
    ])

print(f"Processing {args.language} data...")
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

# Remove CONTROL class from the dataset entirely
print(f"\nRemoving CONTROL class from dataset...")
initial_data = initial_data[initial_data['class'] != 'CONTROL']
print(f"Dataset size after removing CONTROL: {len(initial_data)} samples")

# Split the data into train and test sets without any balancing
train_data, test_data = train_test_split(initial_data, test_size=0.2, random_state=42, stratify=initial_data['class'])

# No balancing - use the splits as they are
final_train_data = train_data
final_test_data = test_data

# Calculate and print class distribution percentages for training set
train_class_counts = final_train_data['class'].value_counts()
train_total = len(final_train_data)
print(f"\nTrain set class distribution ({args.language}):")
for class_name, count in train_class_counts.items():
    percentage = (count / train_total) * 100
    print(f"{class_name}: {count} samples ({percentage:.2f}%)")

print(f"\nTotal train samples: {train_total}")

# Calculate and print class distribution percentages for test set
test_class_counts = final_test_data['class'].value_counts()
test_total = len(final_test_data)
print(f"\nTest set class distribution ({args.language}):")
for class_name, count in test_class_counts.items():
    percentage = (count / test_total) * 100
    print(f"{class_name}: {count} samples ({percentage:.2f}%)")

print(f"\nTotal test samples: {test_total}")

# Create language-specific filename suffix
lang_suffix = ""
if args.language.lower() == "english":
    lang_suffix = "_english"
elif args.language.lower() == "spanish":
    lang_suffix = "_spanish"
else:
    lang_suffix = "_both"

# Save both datasets to CSV with language-specific names
final_train_data.to_csv(f'cleaned_tweets_train{lang_suffix}.csv', index=False)
final_test_data.to_csv(f'cleaned_tweets_test{lang_suffix}.csv', index=False)
print(f"\nSaved datasets with {args.language} data.")