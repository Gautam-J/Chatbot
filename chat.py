from utils import (
    loadTrainedTransformerModel,
    loadFitTokenizer,
    predict
)

tokenizer = loadFitTokenizer('models/myTokenizer')
model = loadTrainedTransformerModel('models/final_model_weight.hdf5', tokenizer)

while True:
    userInput = input('Enter your query [q to quit]: ')

    if userInput == 'q':
        break

    predict(userInput, model, tokenizer)
