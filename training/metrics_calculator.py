import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('weighted_training_logs.json', 'r') as file:
    logs = json.load(file)

# Separate training and evaluation data
training_data = [entry for entry in logs if 'loss' in entry and 'eval_loss' not in entry]
eval_data = [entry for entry in logs if 'eval_loss' in entry]

# Create figure for combined metrics
plt.figure(figsize=(12, 6))

# Plot training loss
epochs = [entry['epoch'] for entry in training_data]
loss = [entry['loss'] for entry in training_data]
plt.plot(epochs, loss, 'b-', label='Training Loss')

# Plot evaluation metrics
eval_epochs = [entry['epoch'] for entry in eval_data]
eval_loss = [entry['eval_loss'] for entry in eval_data]
eval_accuracy = [entry['eval_accuracy'] for entry in eval_data]

plt.plot(eval_epochs, eval_loss, 'r-', label='Evaluation Loss')
plt.plot(eval_epochs, eval_accuracy, 'g-', label='Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Training and Evaluation Metrics over Time')
plt.grid(True)
plt.legend(loc='center right')

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

# Create separate figure for F1 scores
plt.figure(figsize=(10, 6))
class_names = ['ADHD', 'Anxiety', 'ASD', 'Control', 'Depression', 
               'Eating', 'OCD', 'PTSD', 'Schizophrenia']
final_f1_scores = [
    eval_data[-1]['eval_f1_class_0'],
    eval_data[-1]['eval_f1_class_1'],
    eval_data[-1]['eval_f1_class_2'],
    eval_data[-1]['eval_f1_class_4'],
    eval_data[-1]['eval_f1_class_5'],
    eval_data[-1]['eval_f1_class_6'],
    eval_data[-1]['eval_f1_class_7'],
    eval_data[-1]['eval_f1_class_8'],
    eval_data[-1]['eval_f1_class_9']
]

bars = plt.bar(class_names, final_f1_scores)
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.title('Final F1 Scores by Class')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('f1_scores.png')
plt.close()

print("Graphs have been generated and saved as 'training_metrics.png' and 'f1_scores.png'")
