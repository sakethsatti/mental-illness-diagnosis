import pandas as pd
import numpy as np
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import csv
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("Reading Dataset...")
# Read the dataset file and take a sample
csv_file_path = '../cleaned_tweets.csv'
df = pd.read_csv(csv_file_path).sample(frac=0.001, random_state=58)
df.columns = ['text', 'label', 'lang']

print("Encoding labels...")
# Create a label mapping
unique_labels = df['label'].unique()
label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Convert labels to integers
df['label'] = df['label'].map(label_to_id)
print("\nLabel mapping:")
for label, idx in label_to_id.items():
    print(f"{label}: {idx}")

# After creating label mapping, get the number of classes
num_classes = len(label_to_id)
print(f"Number of unique classes: {num_classes}")

print("Splitting Train and Test Values...")
# Rename the columns
df.columns = ['text', 'label', 'lang'] 

# Print class distribution by language
print("\nClass distribution by language:")
print(pd.crosstab(df['label'], df['lang']))

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=58, stratify=df['label'])

print("Converting DataFrames to Hugging Face Datasets...")
# Convert the DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Calculate class weights using inverse frequency
print("Calculating class weights...")
label_counts = train_df['label'].value_counts().sort_index()  # Sort by label index
total_samples = len(train_df)
class_weights = {
    label: total_samples / (len(label_counts) * count)
    for label, count in label_counts.items()
}


# Convert weights to tensor, ensuring order matches label IDs
class_weights_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)])
print("\nClass distribution:")
print(label_counts) 
print(f"\nClass weights: {class_weights}")

# Initialize tokenizer and model with correct number of classes
print("Initializing tokenizer and model...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=num_classes  # Use the actual number of classes
)


model = model.to(device)  # Move model to GPU

# Tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )
    # Remove return_tensors="pt" as it causes issues with the dataset mapping
    return tokenized

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Before defining training arguments
os.makedirs("./results", exist_ok=True)

# Modify training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    no_cuda=False,
    fp16=True,
)

# Custom trainer class with weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted',
        zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Also calculate per-class metrics
    per_class_f1 = precision_recall_fscore_support(
        labels, 
        predictions,
        average=None,
        zero_division=0
    )[2]
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    
    # Add per-class F1 scores
    for i, class_f1 in enumerate(per_class_f1):
        metrics[f'f1_class_{i}'] = class_f1
    
    return metrics

# Change the trainer initialization to use WeightedTrainer
trainer = WeightedTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# Save training logs
print("Saving training logs...")
with open(f'weighted_training_logs.json', 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)

# Save the model
print("Saving model...")
model.save_pretrained("./xlm_tweet_classifier")
