# Named Entity Recognition (NER) for Mountain Names

## Task Overview
In this project, I developed a Named Entity Recognition (NER) model to identify mountain names within text. The task involved dataset creation, model selection, training/fine-tuning, and preparing demo code to showcase the model's inference capabilities. The solution is implemented in Python 3.10, with all relevant code and files provided in this repository. Code make possible to save model locally but best trained model is already loaded on the Hugging Face, and used in inferences from Hugging Face so there is no need to train again and save model locally into folders. [Link to model weights](https://huggingface.co/Darebal/mountain-names-ner/tree/main)

## Setup Instructions

### Prerequisites

This project requires Python 3.10+.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Baredal/internship_test.git
   cd task_1
   
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt

### Running the Code
**Training Script:**
To train or fine-tune and save the NER model, run the following script (no need to train, model already saved on hugging face and all inferences uses model from [trained model](https://huggingface.co/Darebal/mountain-names-ner):
```bash
python training.py
```

**Inference Script:**
To run the inference on a given sentence, use:
```bash
python inference.py
```

**Demo Jupyter Notebook:**
Jupyter notebook is already run with provided results for inference but if you want run:
```bash
jupyter notebook inference_demo.ipynb
```
## Files Overview
### Dataset
The dataset used in this project contains labeled text data with mountain names marked for NER. It was manually created from scraping 3 vocabulary websites. You can find the dataset in the `data/` folder.

- `all.csv`: Contains all original sentences.
- `train_data.csv` and `val_data.csv`: Contain data for training and validating, split in an 80-20 way.
- `train_data_modified.csv` and `val_data_modified.csv`: Contain original sentences, new sentences (mountain and mountains words are replaced with mountain names), BIO tags, and tags for model training (0, 1, and 2) corresponding.

#### Labels:
- **O**: Non-mountain words.
- **B-MOUNTAIN**: Beginning of a mountain name.
- **I-MOUNTAIN**: Inside a mountain name.
  
### Data Creation Notebook (`dataset_creation.ipynb`)
This script contains dataset creation from scraping websites, balanced mountain names replacing, tokenizing and bio taging.

### Model Training Script (`training.py`)
This script fine-tunes a pre-trained BERT-based model for the NER task. The script loads the dataset, tokenizes the text, trains the model with appropriate configurations, saving each model on each epoch in `results_model/` and then saves the best model in `model_top/` folder based on epochs, saves trainer hisory (state) and evaluation results.

### Model Inference Script (`inference.py`)
This script takes a sentence and returns the entity labels (e.g., mountain names) using the trained model.

### Demo Notebook (`inference_demo.ipynb`)
A Jupyter notebook that demonstrates trained model prediction on set of sentences, classifying each token to predicted class

### Evaluation results (`evaluation_results.json`)
Saved results from evaluation on validation set.

### Trainer history (`trainer_state.json`)
Evaluation of model on each epoch on validation set.

## Results and Evaluation
The evaluation of the NER best model from training process on validation set:

| Metric                     | Value                     |
|----------------------------|---------------------------|
| eval_loss                  | 0.009353181347250938      |
| eval_precision             | 0.9497716894977168        |
| eval_recall                | 0.9674418604651163        |
| eval_f1                    | 0.9585253456221198        |
| eval_accuracy              | 0.9968582275166906        |
| eval_runtime               | 1.954                     |
| eval_samples_per_second    | 113.101                   |
| eval_steps_per_second      | 7.165                     |
| epoch                      | 5.0                       |


## Report
A report containing potential improvements and further optimizations for this task is provided in a PDF file in the root directory.


