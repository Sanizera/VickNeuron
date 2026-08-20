import math
import random
#Função de Ativação (deixa o resultado sempre entre 0 e 1)
def sigmoid(x):
    return 1/(1 + math.exp(-x))

class Neuron:

    def __init__(self, numOfEntries):
     
        ## weights
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(numOfEntries)]

        ## bias
        self.bias = random.uniform(-1, 1)

    def processEntries(self, entries):
        # Soma ponderada
        total = self.bias

        for x, w in zip(entries, self.weights):
            total += x*w
        
        # Activation
        output = sigmoid(total)

        # Last Results
        self.last_entries = entries
        self.last_output = output

        return output

    def backward(self, delta, learnRate):
        for i in range(len(self.weights)):
            gradient = delta * self.last_entries[i]
            self.weights[i] += learnRate * gradient

        self.bias += learnRate * delta
