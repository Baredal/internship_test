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
To train or fine-tune the NER model, run the following script (no need to train, model already saved on hugging face and all inferences uses model from [trained model](https://huggingface.co/Darebal/mountain-names-ner):
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

