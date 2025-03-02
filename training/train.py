import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add argument parsing for command line control
parser = argparse.ArgumentParser(description='Train mental illness diagnosis model')
parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0-1.0)')
parser.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of test data to use (0.0-1.0)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
args = parser.parse_args()

# Parameters
MODEL_NAME = "Twitter/twhin-bert-base"
BATCH_SIZE = args.batch_size
MAX_LENGTH = 128
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
WARMUP_RATIO = 0.1
CLASS_WEIGHT_EXPONENT = 1.0
RS = 42
TRAIN_FRACTION = args.train_fraction
TEST_FRACTION = args.test_fraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load separate train and test datasets
train_df = pd.read_csv('./cleaned_tweets_train.csv')
test_df = pd.read_csv('./cleaned_tweets_test.csv')

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

# Define class mapping for multi-class classification
class_names = ['control', 'adhd', 'depression', 'anxiety', 'asd', 
               'bipolar', 'ptsd', 'ocd', 'eating', 'schizophrenia']
class_mapping = {
    'control': 0,
    'adhd': 1,
    'depression': 2,
    'anxiety': 3,
    'asd': 4,
    'bipolar': 5,
    'ptsd': 6,
    'ocd': 7,
    'eating': 8,
    'schizophrenia': 9
}

# Map labels to numerical values (ensure all labels are in lowercase)
train_df['label'] = train_df['label'].str.lower().map(class_mapping)
test_df['label'] = test_df['label'].str.lower().map(class_mapping)

# Print the dataset sizes after sampling
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model with the updated number of labels
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(class_mapping),
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
).to(device)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True, target_names=class_names)
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        **{f'f1_{name}': report[name]['f1-score'] for name in class_names}
    }

# Compute class weights based on training data
label_counts = train_df['label'].value_counts().to_dict()
total_count = len(train_df)
num_classes = len(class_mapping)
class_weights = [(total_count / (num_classes * label_counts[i]))**CLASS_WEIGHT_EXPONENT for i in range(num_classes)]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_tensor = weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


steps_per_epoch = len(tokenized_train) // (BATCH_SIZE)
logging_steps = steps_per_epoch // 20

print(logging_steps)

# Training arguments
training_args = TrainingArguments(
    report_to="none",
    output_dir='./twitter_xlm_results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_first_step=True,
    fp16 = False,
    bf16=True,
    tf32 = True,
    warmup_ratio=WARMUP_RATIO,
    dataloader_pin_memory=True,
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim = "adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

import json

full_training_log = trainer.state.log_history

try:
    with open("full_log_history.json", "w") as f:
        json.dump(full_training_log, f)
except:
    pass

# Save model and tokenizer
model.save_pretrained('./twitter_xlm_final_model')
tokenizer.save_pretrained('./twitter_xlm_final_tokenizer')
print("Training complete! Model and tokenizer saved.")

# Evaluate on test data and get predictions
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

results_df = pd.DataFrame({
    'text': test_df['text'].values,
    'lang': test_df['lang'].values,
    'true_label': true_labels,
    'predicted_label': pred_labels
})

results_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv.")

# Create a confusion matrix and save as an image
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))