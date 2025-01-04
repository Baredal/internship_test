'''
Makes inference of model on sentence to retrieve only mountain names from sentence with their positions indexes
'''
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline


def make_inference(model, tokenizer, sentence):
    """
    Make token classification predictions using a pre-trained BERT model for token classification.
    
    Args:
        model: The pre-trained BERT model for token classification.
        tokenizer: The tokenizer corresponding to the model.
        sentences: A list of sentences to perform inference on.
    
    Returns:
        List. All mountains tokens
    """
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer)
    prediction = nlp(sentence)
    all_mountains = []
    for token in prediction:
        if token['entity'] != 'LABEL_0':
            all_mountains.append(token)

    return all_mountains


if __name__ == '__main__':
    # Load the pre-trained model and tokenizer from Hugging Face
    model = BertForTokenClassification.from_pretrained("Darebal/mountain-names-ner")
    tokenizer = BertTokenizerFast.from_pretrained("Darebal/mountain-names-ner")

    # Define the custom label2tag mapping to map label indices to human-readable tags
    label2tag = {'LABEL_1': 'B-MOUNTAIN', 'LABEL_2': 'I-MOUNTAIN'}
    text = "The valleys between the tilted Beartooth Mountains blocks are smooth and often trough-like, and are often the sites of shallow salt lakes or playas."

    mountains = make_inference(model, tokenizer, text)
    for mountain in mountains:
        label = label2tag[mountain['entity']]
        print(f"Word - {mountain['word']}, Label - {label}, Start index: {mountain['start']}, End index - {mountain['end']}")

