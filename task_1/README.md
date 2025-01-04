# Named Entity Recognition (NER) for Mountain Names

## Task Overview
In this project, we developed a Named Entity Recognition (NER) model to identify mountain names within text. The task involved dataset creation, model selection, training/fine-tuning, and preparing demo code to showcase the model's inference capabilities. The solution is implemented in Python 3.10, with all relevant code and files provided in this repository. [Link to model weights](https://huggingface.co/Darebal/mountain-names-ner/tree/main)

### Contents:
1. [Dataset Creation](#dataset-creation)
2. [Model Selection and Training](#model-selection-and-training)
3. [Model Inference and Demo](#model-inference-and-demo)
4. [Evaluation and Results](#evaluation-and-results)
5. [Potential Improvements (Report)](#potential-improvements-report)

---

## Setup Instructions

### Prerequisites

This project requires Python 3.10+.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/mountain-names-ner.git
   cd mountain-names-ner
   
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
Jupyter notebook is already run but you can run the Jupyter notebook to see the inference:
```bash
jupyter notebook inference_demo.ipynb
```
## Files Overview
### Dataset
The dataset used in this project contains labeled text data with mountain names marked for NER. It was manually created from scraping 3 vocabulary websites. You can find the dataset in the `data/` folder.
all.csv - contains all original sentences <br>
train_data.csv and val_data.csv contain data for training and validating, splited in 80-20 way <br>
train_data_modified.csv and val_data_modified.csv contsins original sentences, new sentences (mountain and mountains words are replaced with mountain names), bio-tags ('O', 'B-MOUNTAIN', 'I-MOUNTAIN') and tags for model training (0, 1 and 2)

### Model Training Script (`training.py`)
This script fine-tunes a pre-trained BERT-based model (or another suitable architecture) for the NER task. The script loads the dataset, tokenizes the text, and trains the model with appropriate configurations.

### Model Inference Script (`inference.py`)
This script takes a set of sentences and predicts the entity labels (e.g., mountain names) for each token using the trained model.

### Demo Notebook (`inference_demo.ipynb`)
A Jupyter notebook that demonstrates the complete process from dataset preparation to model training and inference. It also showcases the evaluation metrics and results.


