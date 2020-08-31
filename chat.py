from utils import (
    loadTrainedTransformerModel,
    loadFitTokenizer,
    predict
)

model = loadTrainedTransformerModel()
tokenizer = loadFitTokenizer()

while True:
    userInput = input('Enter your query [q to quit]: ')

    if userInput == 'q':
        break

    predict(userInput, model, tokenizer)
