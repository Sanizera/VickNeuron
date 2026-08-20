from neuron import Neuron

def sigmoid_derivative(sigmoidOutput):
    return sigmoidOutput * (1 - sigmoidOutput)

class Layer:
    def __init__(self, n_entries, n_neurons):
        self.n_entries = n_entries
        self.n_neurons = n_neurons

        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron(n_entries))
            

    def process(self, entries):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.processEntries(entries))

        return outputs
    
    def backward_output(self, expected, learnRate):

        deltas = []
        for neuron, expect in zip(self.neurons, expected):
            foreseen = neuron.last_output
            error = expect - foreseen
            delta = error *sigmoid_derivative(foreseen)
            neuron.backward(delta, learnRate)
            deltas.append(delta)
        return deltas

