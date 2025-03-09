import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add argument parsing for command line control
parser = argparse.ArgumentParser(description='Train mental illness binary classification model')
parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0-1.0)')
parser.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of test data to use (0.0-1.0)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--language', type=str, default='both', choices=['english', 'spanish', 'both'],
                    help='Language of the dataset to use')
args = parser.parse_args()

# Parameters
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"
BATCH_SIZE = args.batch_size
MAX_LENGTH = 64
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
WARMUP_RATIO = 0.1
RS = 42
TRAIN_FRACTION = args.train_fraction
TEST_FRACTION = args.test_fraction
LANGUAGE = args.language

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define language suffix for loading files
lang_suffix = ""
if LANGUAGE == "english":
    lang_suffix = "_english"
elif LANGUAGE == "spanish":
    lang_suffix = "_spanish"
else:
    lang_suffix = "_both"

# Load binary classification datasets
train_df = pd.read_csv(f'./cleaned_tweets_train{lang_suffix}_binary.csv')
test_df = pd.read_csv(f'./cleaned_tweets_test{lang_suffix}_binary.csv')

# Rename columns to match expected format
train_df.columns = ['text', 'label', 'lang']
test_df.columns = ['text', 'label', 'lang']

# Sample fraction of data while maintaining class distribution
if TRAIN_FRACTION < 1.0:
    print(f"Using {TRAIN_FRACTION*100:.1f}% of training data")
    train_df = train_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=TRAIN_FRACTION, random_state=RS)
    ).reset_index(drop=True)

if TEST_FRACTION < 1.0:
    print(f"Using {TEST_FRACTION*100:.1f}% of test data")
    test_df = test_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=TEST_FRACTION, random_state=RS)
    ).reset_index(drop=True)

# Define binary class mapping
class_names = ['control', 'mental_illness']
class_mapping = {
    'control': 0,
    'mental_illness': 1
}

# Map labels to numerical values (ensure all labels are in lowercase)
train_df['label'] = train_df['label'].str.lower().map(class_mapping)
test_df['label'] = test_df['label'].str.lower().map(class_mapping)

# Print the dataset sizes and class distribution after sampling
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Print class distribution
train_class_dist = train_df['label'].value_counts()
print("\nTraining class distribution:")
for class_id, count in train_class_dist.items():
    class_name = [k for k, v in class_mapping.items() if v == class_id][0]
    percent = (count / len(train_df)) * 100
    print(f"  {class_name}: {count} ({percent:.1f}%)")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model with binary classification (2 labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
).to(device)

def tokenize_function(examples):
    return tokenizer(examples['text'],
                     padding='max_length', 
                     truncation=True,
                     max_length=MAX_LENGTH)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True, target_names=class_names)
    
    # Calculate additional binary metrics
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    
    return {
        'accuracy': report['accuracy'],
        'f1': report['weighted avg']['f1-score'],
        'precision': precision,
        'recall': recall,
        'f1_control': report['control']['f1-score'],
        'f1_mental_illness': report['mental_illness']['f1-score'],
    }

steps_per_epoch = len(tokenized_train) // BATCH_SIZE
logging_steps = max(1, steps_per_epoch // 10)

print(f"Steps per epoch: {steps_per_epoch}, logging every {logging_steps} steps")

# Training arguments
training_args = TrainingArguments(
    report_to="none",
    output_dir='./binary_model_results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./binary_logs',
    logging_first_step=True,
    fp16=False,
    bf16=True,
    tf32=True,
    warmup_ratio=WARMUP_RATIO,
    dataloader_pin_memory=True,
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
)

# Standard trainer without class weighting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train model
print("\nStarting training...")
trainer.train()

# Save model and tokenizer
model.save_pretrained('./binary_model_final')
tokenizer.save_pretrained('./binary_tokenizer_final')
print("Training complete! Model and tokenizer saved.")

# Evaluate on test data and get predictions
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Convert numeric labels back to text for better readability
id_to_class = {v: k for k, v in class_mapping.items()}
pred_labels_text = [id_to_class[label] for label in pred_labels]
true_labels_text = [id_to_class[label] for label in true_labels]

# Save predictions
results_df = pd.DataFrame({
    'text': test_df['text'].values,
    'lang': test_df['lang'].values,
    'true_label': true_labels_text,
    'predicted_label': pred_labels_text,
    'true_label_id': true_labels,
    'predicted_label_id': pred_labels
})

results_df.to_csv('binary_test_predictions.csv', index=False)
print("Predictions saved to binary_test_predictions.csv.")

# Create a confusion matrix and save as an image
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Binary Classification Confusion Matrix ({LANGUAGE})")
plt.tight_layout()
plt.savefig("binary_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as binary_confusion_matrix.png")

# Print classification report
print("\nBinary Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))
# Get prediction probabilities for positive class (mental_illness)
pred_probs = predictions.predictions[:, 1]

# Calculate ROC curve points and AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Binary Classification ({LANGUAGE})')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("binary_roc_curve.png")
plt.close()

print(f"\nAUC-ROC Score: {roc_auc:.4f}")
print("ROC curve saved as binary_roc_curve.png")