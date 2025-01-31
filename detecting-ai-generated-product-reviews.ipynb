import pandas as pd
import numpy as np
import re
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch


for folder in ["./results", "./logs", "./submission"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)


os.environ["WANDB_DISABLED"] = "true"


def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


train_paths = {
    "tamil": "/kaggle/input/ai-review/tam_training_data_hum_ai.csv",
    "malayalam": "/kaggle/input/ai-review/mal_training_data_hum_ai.csv"
}
test_paths = {
    "tamil": "/kaggle/input/ai-review/tamil-test.xlsx",
    "malayalam": "/kaggle/input/ai-review/mal_test.xlsx"
}


train_data = {lang: pd.read_csv(path) for lang, path in train_paths.items()}
test_data = {lang: pd.read_excel(path) for lang, path in test_paths.items()}


for lang in train_data.keys():
    # Train data
    train_data[lang]['CLEANED_DATA'] = train_data[lang]['DATA'].apply(preprocess_text)
    train_data[lang]['LABEL'] = train_data[lang]['LABEL'].map({"HUMAN": 0, "AI": 1})
    
    # Test data
    test_data[lang]['CLEANED_DATA'] = test_data[lang]['DATA'].apply(preprocess_text)
    test_data[lang]['LABEL'] = test_data[lang]['LABEL'].map({"HUMAN": 0, "AI": 1})


MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    save_strategy="no",
    lr_scheduler_type="linear",
    warmup_steps=300,
    report_to=["none"]
)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, lang):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {lang.capitalize()}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{lang}.png')
    plt.close()

# Training and evaluation loop
for lang in train_data.keys():
    print(f"\n{'='*40}")
    print(f"Training and Evaluating {lang.capitalize()} Model")
    print(f"{'='*40}")
    
    # Tokenize data
    train_encodings = tokenizer(
        list(train_data[lang]['CLEANED_DATA']),
        truncation=True,
        padding=True,
        max_length=256
    )
    test_encodings = tokenizer(
        list(test_data[lang]['CLEANED_DATA']),
        truncation=True,
        padding=True,
        max_length=256
    )


    train_dataset = ClassificationDataset(train_encodings, list(train_data[lang]['LABEL']))
    test_dataset = ClassificationDataset(test_encodings, list(test_data[lang]['LABEL']))


    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {
            "precision": precision_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro"),
            "recall": recall_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro"),
            "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro"),
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
        }
    )


    trainer.train()
    

    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Results for {lang.capitalize()}:")
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1-Score: {test_results['eval_f1']:.4f}")


    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    cm = confusion_matrix(labels, preds)

    plot_confusion_matrix(cm, classes=['HUMAN', 'AI'], lang=lang)
    print(f"Confusion matrix saved as confusion_matrix_{lang}.png")
