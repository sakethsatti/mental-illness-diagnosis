import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Parameters
MODEL_NAME = "MilaNLProc/xlm-emo-t"
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 3
FRAC_DATASET = 0.005
RS = 58 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv('../cleaned_tweets.csv').sample(frac=FRAC_DATASET, random_state=RS)
df.columns = ['text', 'label', 'lang']

df['label'] = df['label'].apply(lambda x: 0 if x.lower() == 'control' else 1)  # Binary classification

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RS, stratify=df['label'])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load BERTweet tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
).to(device)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Training arguments
training_args = TrainingArguments(
    output_dir='./xlm_emo_results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained('./xlm_emo_final_model')
tokenizer.save_pretrained('./xlm_emo_final_tokenizer')

print("Training complete! Model and tokenizer saved.")

# Evaluate on test data and get predictions

predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Save predictions to a DataFrame and CSV (optional)

results_df = pd.DataFrame({
'text': test_df['text'].values,
'true_label': true_labels,
'predicted_label': pred_labels
})
results_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv!")

#Create a confusion matrix and save as an image

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=['Control', 'Other'], yticklabels=['Control', 'Other'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as confusion_matrix.png!")

#Print classification report

print("Classification Report:")
print(classification_report(true_labels, pred_labels))

