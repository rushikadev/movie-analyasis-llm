# Fine tuned model to classify movie reviews 

https://huggingface.co/rdev610/fine-tuned-distilbert-movie-sentiment

This model classifies movie reviews as either positive or negative. 

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model classifies movie reviews as either positive or negative. It was trained on the cornell-movie-reviews dataset and was based upon the dilbert-base uncased model. 

- **Developed by:** Rushika Devineni
- **Model type:** distilbert
- **Language(s) (NLP):** English
- **License:** MIT License (Inherited from the base model: distilbert-base-uncased)
- **Finetuned from model [optional]:** distilbert-base-uncased model


## Uses
The main use is analyze movie reviews and classify them as negative or positive.
Users can input raw text (a movie review or similar piece of text) and the model will output a predicted sentiment label (either 'positive' or 'negative') along with associated confidence probabilities for each class.
Foreseeable direct users include researchers, developers, or hobbyists who need a pre-trained and fine-tuned tool for sentiment analysis on textual data, particularly within or related to the film domain.

## How to Get Started with the Model

Use the code below to get started with the model form Hugging Face Hub.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the model ID on the Hugging Face Hub
# REPLACE [Your HF Username] with your actual username if it's different from rdev610
model_id = "rdev610/fine-tuned-distilbert-movie-sentiment" # Your model's ID on the Hub

# Load the model and tokenizer from the Hub
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    print(f"Model and tokenizer loaded successfully from {model_id}")

    # Example of how to use the model for inference:
    text = "This movie was fantastic and I loved every minute of it!" # Replace with your desired input text

    # Prepare the input for the model
    inputs = tokenizer(text, return_tensors="pt")

    # If using GPU, move model and inputs to GPU
    if torch.cuda.is_available():
        model.to('cuda')
        inputs = {key: val.to('cuda') for key, val in inputs.items()}

    # Perform inference
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs = model(**inputs)

    # Interpret the output
    # Assuming a binary classification (0 for negative, 1 for positive)
    predictions = torch.argmax(outputs.logits, dim=1)
    label_map = {0: 'negative', 1: 'positive'} # Confirm this mapping is correct
    sentiment = label_map[predictions.item()]

    print(f"\nInput Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")

except Exception as e:
    print(f"Error loading or using the model: {e}")

## Training Details

### Training Data

The model was fine-tuned on the Rotten Tomatoes (rtr) dataset, which is available on the Hugging Face datasets library (https://huggingface.co/datasets/rotten_tomatoes). This dataset consists of movie review snippets labeled with binary sentiment (positive or negative).
