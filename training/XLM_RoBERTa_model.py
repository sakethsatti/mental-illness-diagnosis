import pandas as pd
import numpy as np
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import datetime
import os
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

FRAC_DATASET = 0.005
RS = 58 # Random STate
MAX_LENGTH = 128
BATCH_SIZE = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("Reading Dataset...")
# Read the dataset file
csv_file_path = '../cleaned_tweets.csv'
df = pd.read_csv(csv_file_path).sample(frac=FRAC_DATASET, random_state=RS)
df.columns = ['text', 'label', 'lang']

# Convert to binary classification
print("\nConverting to binary classification...")
# Assuming 'mental_illness' is one of your labels
# Map all other labels to 'no_mental_illness'
df['label'] = df['label'].apply(lambda x: 'no_mental_illness' if x.lower() == 'control' else 'mental_illness')

# Add detailed class statistics before balancing
print("\nInitial class distribution:")
class_stats = df['label'].value_counts()
print(class_stats)
print("\nPercentage distribution:")
print(class_stats / len(df) * 100)

print("\nSamples per language in each class:")
lang_class_dist = pd.crosstab(df['label'], df['lang'], margins=True)
print(lang_class_dist)

print("\nBalancing dataset...")
# Get the count of the class with minimum samples
min_class_count = df['label'].value_counts().min()

# Balance the dataset by downsampling each class to the minimum count
balanced_dfs = []
for label in df['label'].unique():
    class_df = df[df['label'] == label].sample(n=min_class_count, random_state=RS)
    balanced_dfs.append(class_df)

# Combine all balanced classes
df = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle the final dataset
df = df.sample(frac=1, random_state=RS).reset_index(drop=True)

# Add more detailed statistics after balancing
print("\nDetailed statistics after balancing:")
print("\nSamples per class:")
balanced_stats = df['label'].value_counts()
print(balanced_stats)

print("\nLanguage distribution per class after balancing:")
balanced_lang_dist = pd.crosstab(df['label'], df['lang'], margins=True)
print(balanced_lang_dist)

print("\nFinal class distribution:")
print(df['label'].value_counts())

print("Encoding labels...")
# Create a label mapping
unique_labels = df['label'].unique()
label_to_id = {
    'no_mental_illness': 0,
    'mental_illness': 1
}
id_to_label = {v: k for k, v in label_to_id.items()}

# Convert labels to integers
df['label'] = df['label'].map(label_to_id)
print("\nLabel mapping:")
for label, idx in label_to_id.items():
    print(f"{label}: {idx}")

# After creating label mapping, get the number of classes
num_classes = len(label_to_id)
print(f"Number of classes: {num_classes}")

print("Splitting Train and Test Values...")
df.columns = ['text', 'label', 'lang'] 

print("\nClass distribution by language:")
print(pd.crosstab(df['label'], df['lang']))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RS, stratify=df['label'])

print("Converting DataFrames to Hugging Face Datasets...")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Initializing tokenizer and model...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=num_classes 
)

model = model.to(device) 

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )
    
print("Tokenizing datasets...")
# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Create directories for detailed logging
os.makedirs('./detailed_logs', exist_ok=True)
class CustomCallback(TrainerCallback):
    def __init__(self):
        self.training_stats = []
        self.eval_stats = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step_stats = {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Training metrics
            if 'loss' in logs:
                step_stats.update({
                    'training_loss': logs.get('loss'),
                    'learning_rate': logs.get('learning_rate'),
                    'train_samples_per_second': logs.get('train_samples_per_second')
                })
                self.training_stats.append(step_stats)
                
                # Save training stats periodically
                with open('./detailed_logs/training_stats.json', 'w') as f:
                    json.dump(self.training_stats, f, indent=2)
            
            # Evaluation metrics
            if 'eval_loss' in logs:
                step_stats.update({
                    'eval_loss': logs.get('eval_loss'),
                    'eval_accuracy': logs.get('eval_accuracy'),
                    'eval_f1': logs.get('eval_f1'),
                    'eval_precision': logs.get('eval_precision'),
                    'eval_recall': logs.get('eval_recall'),
                    'eval_samples_per_second': logs.get('eval_samples_per_second')
                })
                self.eval_stats.append(step_stats)
                
                # Save evaluation stats
                with open('./detailed_logs/eval_stats.json', 'w') as f:
                    json.dump(self.eval_stats, f, indent=2)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",  # Disable wandb logging
)

custom_callback = CustomCallback()

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[custom_callback]  # Add our custom callback
)

print("Starting training...")
trainer.train()

# Save final training summary
final_summary = {
    'total_training_steps': len(custom_callback.training_stats),
    'total_eval_steps': len(custom_callback.eval_stats),
    'final_eval_metrics': trainer.evaluate(),
    'training_duration': str(datetime.datetime.now() - datetime.datetime.strptime(
        custom_callback.training_stats[0]['timestamp'], 
        '%Y-%m-%d %H:%M:%S'
    )),
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
}

with open('./detailed_logs/training_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print("Evaluating model...")
final_results = trainer.evaluate()
print("\nFinal Evaluation Results:")
for metric, value in final_results.items():
    print(f"{metric}: {value:.4f}")

# Save the model and tokenizer
print("Saving model and tokenizer...")
model_save_path = './final_model'
tokenizer_save_path = './final_tokenizer'

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

# Save label mappings
with open(os.path.join(model_save_path, 'label_mappings.json'), 'w') as f:
    json.dump({
        'label_to_id': label_to_id,
        'id_to_label': id_to_label
    }, f)

print("Training completed! Model, tokenizer, and label mappings saved.")

# Optional: Print some example predictions
print("\nGenerating example predictions...")
example_indices = test_df.index[:5]  # Get first 5 indices
example_texts = test_df.loc[example_indices, 'text'].tolist()
example_true_labels = test_df.loc[example_indices, 'label'].tolist()

inputs = tokenizer(example_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(predictions, dim=-1)

print("\nExample Predictions:")
for text, pred_label, true_label in zip(example_texts, predicted_labels, example_true_labels):
    predicted_class = id_to_label[pred_label.item()]
    true_class = id_to_label[true_label]
    confidence = predictions[predicted_labels == pred_label].max().item()
    
    print(f"\nText: {text[:100]}...")
    print(f"Predicted: {'Shows mental illness' if predicted_class == 'mental_illness' else 'No mental illness'} (confidence: {confidence:.2%})")
    print(f"Actual: {'Shows mental illness' if true_class == 'mental_illness' else 'No mental illness'}")
    print(f"Correct: {'✓' if predicted_class == true_class else '✗'}")


print("\nGenerating confusion matrix for test set...")
test_inputs = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
with torch.no_grad():
    test_outputs = model(**test_inputs)
    test_predictions = torch.argmax(test_outputs.logits, dim=-1)

conf_matrix = confusion_matrix(test_df['label'], test_predictions.cpu())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=['No Mental Illness', 'Mental Illness'],
            yticklabels=['No Mental Illness', 'Mental Illness'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('./detailed_logs/confusion_matrix.png')
plt.close()

# Save detailed test set metrics per class
from sklearn.metrics import classification_report
class_report = classification_report(
    test_df['label'],
    test_predictions.cpu(),
    target_names=['No Mental Illness', 'Mental Illness'],
    output_dict=True
)

with open('./detailed_logs/class_report.json', 'w') as f:
    json.dump(class_report, f, indent=2)

print("\nDetailed classification report and confusion matrix have been saved to the detailed_logs directory.")
