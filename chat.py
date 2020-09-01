from data_configs import loadTokenizer
from utils import loadTrainedTransformerModel, predict

tokenizer = loadTokenizer('models/myTokenizer.json')
model = loadTrainedTransformerModel('models/final_model_weight')

while True:
    userInput = input('You: ')

    if userInput == 'q':
        break

    predictedResponse = predict(userInput, model, tokenizer)
    print('AI:', predictedResponse)
    print()
