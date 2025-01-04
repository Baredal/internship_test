'''
Trains the model, saves it into directory adding evaluation on validation set json and trainer json
'''

import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, get_linear_schedule_with_warmup, AdamW
from datasets import Dataset as HuggingDataset
import numpy as np
from evaluate import load
import ast
import warnings
import json

warnings.filterwarnings('ignore')

# Define constants for label mapping and training parameters
LABEL_LIST = ['O', 'B-MOUNTAIN', 'I-MOUNTAIN']  # List of entity labels
METRIC = load("seqeval")  # Evaluation metric for sequence labeling
TAG2ID = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}  # Map tags to IDs
BATCH_SIZE = 16  # Batch size for training and evaluation
NUM_EPOCHS = 5  # Number of training epochs
WARUMP_STEPS = 88  # Number of warmup steps for learning rate scheduler
TRAINING_STEPS = 880  # Total training steps

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for token classification tasks.

    Args:
        eval_preds: Tuple containing logits and labels.

    Returns:
        A dictionary with precision, recall, F1-score, and accuracy metrics.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)  # Convert logits to predicted labels

    # Filter out special tokens (-100) and map indices to label names
    true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # Compute metrics using the seqeval library
    all_metrics = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def preprocess_data(examples):
    """
    Preprocess input data for token classification.

    Args:
        examples: A dictionary containing "tokens" and "tags" for each sentence.

    Returns:
        Tokenized inputs with aligned labels for token classification.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to word indices
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token padding
            elif word_idx != previous_word_idx:
                previous_word_idx = word_idx
                label_id = -100 if word_idx is None else TAG2ID[label[word_idx]]
                label_ids.append(label_id)
            else:
                # For subword tokens, adjust the label for inside (I-) tokens
                label_id = TAG2ID[label[word_idx]]
                if label_id % 2 == 1:
                    label_id += 1
                label_ids.append(label_id)
            
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels  # Add labels to the tokenized inputs
    return tokenized_inputs

def train_model(model, tokenizer):
    """
    Train a token classification model using Hugging Face Trainer.

    Args:
        model: The pre-trained BERT model for token classification.
        tokenizer: The tokenizer used for data preparation.

    Returns:
        Evaluation results as a dictionary.
    """
    # Define optimizer with separate learning rates for BERT and the classifier
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 0.000011}
    ])

    # Define a learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARUMP_STEPS,
        num_training_steps=TRAINING_STEPS
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results_model",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        weight_decay=0.05
    )

    # Create a Trainer object for training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation dataset
    eval_outputs = trainer.evaluate()

    trainer.state.save_to_json("trainer_state.json")
    
    return eval_outputs

if __name__ == '__main__':
    # Load and preprocess training and validation data
    train_df, val_df = pd.read_csv('./data/train_data_modified.csv'), pd.read_csv('./data/val_data_modified.csv')
    train_df['tokens'] = train_df['tokens'].apply(ast.literal_eval)  # Convert string to list
    train_df['tags'] = train_df['tags'].apply(ast.literal_eval)
    val_df['tokens'] = val_df['tokens'].apply(ast.literal_eval)
    val_df['tags'] = val_df['tags'].apply(ast.literal_eval)

    # Load tokenizer and data collator for token classification
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Convert data to Hugging Face Dataset format
    train_data = HuggingDataset.from_pandas(train_df)
    val_data = HuggingDataset.from_pandas(val_df)

    # Tokenize and preprocess datasets
    train_data = train_data.map(preprocess_data, batched=True, remove_columns=['tokens', 'tags', 'Sentence', 'new_sentence'])
    val_data = val_data.map(preprocess_data, batched=True, remove_columns=['tokens', 'tags', 'Sentence', 'new_sentence'])
    
    # Load pre-trained BERT model for token classification
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    # Train the model
    evaluation = train_model(model, tokenizer)
    
    # Save evaluation on validation dataset
    with open('evaluation_results.json', 'w') as json_file:
        json.dump(evaluation, json_file, indent=1)

    # Save the trained model and tokenizer to a directory
    save_directory = "./model_top"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
