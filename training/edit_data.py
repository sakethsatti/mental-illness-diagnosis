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
parser.add_argument('--binary', action='store_true', 
                    help='Create a binary dataset (mental illness vs control) instead of multi-class')
args = parser.parse_args()

# Define patterns
# Keeping @ mentions (hence '@' is preserved in the regex)
SPECIAL_CHARS = re.compile(r'[^@a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ]')
REPEAT_PATTERN = re.compile(r'(.)\1{3,}')
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
    full_paths = [os.path.join(prefix, file) for file in os.listdir(prefix) if file.endswith('.csv')]
    
    for file in full_paths:
        df = pd.read_csv(file)

        # This is for a bug where the classes are named differently in English and Spanish
        if (prefix == "../English/Eating_eng/"):
            df["class"] = "EATING"
        elif (prefix == "../English/Asd_eng/"):
            df["class"] = "ASD"

        df['language'] = language
        dataframes.append(df) 

# Concatenate all DataFrames into a single DataFrame
all_data = pd.concat(dataframes, ignore_index=True)

def clean_tweet(tweet):
    try:
        # Replace 'httpurl' with 'URL'
        tweet = re.sub('httpurl', 'URL', tweet, flags=re.IGNORECASE)

        # Standardize @user mentions
        tweet = re.sub(r'@(?:user|usuario)', '@USER', tweet, flags=re.IGNORECASE)

        # Remove special characters (keeps @ due to modified regex)
        tweet = SPECIAL_CHARS.sub('', tweet)
        
        # Limit repetitive character sequences to a maximum of three occurrences
        tweet = REPEAT_PATTERN.sub(lambda m: m.group(1) * 3, tweet)
        
        # Split into words
        words = tweet.split()
            
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

# If binary classification is requested, convert all non-CONTROL classes to "MENTAL_ILLNESS"
if args.binary:
    print("\nCreating binary classification dataset (MENTAL_ILLNESS vs CONTROL)...")
    initial_data['class'] = initial_data['class'].apply(
        lambda x: 'CONTROL' if x == 'CONTROL' else 'MENTAL_ILLNESS'
    )
    print(f"Classes after binary conversion: {initial_data['class'].unique()}")
    
    # Make sure we have a balanced dataset for binary classification
    control_count = len(initial_data[initial_data['class'] == 'CONTROL'])
    mental_illness_count = len(initial_data[initial_data['class'] == 'MENTAL_ILLNESS'])
    print(f"Initial counts - CONTROL: {control_count}, MENTAL_ILLNESS: {mental_illness_count}")
    
    filename_suffix = "_binary"
else:
    # For multi-class, simply remove CONTROL
    print(f"\nRemoving CONTROL class from dataset...")
    initial_data = initial_data[initial_data['class'] != 'CONTROL']
    print(f"Dataset size after removing CONTROL: {len(initial_data)} samples")
    filename_suffix = ""

# Split the data into train and test sets without any balancing
train_data, test_data = train_test_split(
    initial_data, test_size=0.2, random_state=42, 
    stratify=initial_data['class']
)

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

# Combine suffixes
full_suffix = f"{lang_suffix}{filename_suffix}"

# Save both datasets to CSV with language-specific names
final_train_data.to_csv(f'cleaned_tweets_train{full_suffix}.csv', index=False)
final_test_data.to_csv(f'cleaned_tweets_test{full_suffix}.csv', index=False)
print(f"\nSaved datasets with {args.language} data{' (binary classification)' if args.binary else ''}.")
