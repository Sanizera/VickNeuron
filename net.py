from neuron import Neuron, processEntries

class Net:
    def __init__(self, n_entries, n_neurons):
        self.n_entries = n_entries
        self.n_neurons = n_neurons

        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron(n_entries))
            

    def processar(self, entries):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.processEntries(entries))

        return outputs
