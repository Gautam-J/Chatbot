import os

from data_configs import loadTokenizer
from utils import loadTrainedTransformerModel, predict

tokenizer = loadTokenizer('models/myTokenizer.json')
model = loadTrainedTransformerModel('models/final_model_weight')

predict('Hello!', model, tokenizer)
os.system('clear')

print('Type your query and press ENTER, q to quit')
while True:
    userInput = input('\nYou: ')

    if userInput == 'q':
        break

    predictedResponse = predict(userInput, model, tokenizer)
    print('AI:', predictedResponse)
