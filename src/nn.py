import random
from activations import activation_types

class Neuron(object):
    
    def __init__(self, ac):
        self.activation = ac
        self.bias = 0
        self.upstream_neurons = []
        self.upstream_weights = []
        self.downstream_neurons = []
        self.downstream_weights = []
        self.current_value = None
        self.current_z = None

    def _link_input(self, input_neuron, weight):
        self.upstream_neurons.append(input_neuron)
        self.upstream_weights.append(weight)
        input_neuron.downstream_neurons.append(self)
        input_neuron.downstream_weights.append(weight)

    def _link_output(self, output_neuron, weight):
        self.downstream_neurons.append(output_neuron)
        self.downstream_weights.append(weight)
        output_neuron.upstream_neurons.append(self)
        output_neuron.upstream_weights.append(weight)

    def _set_bias(self, bias):
        self.bias = bias

    def _set_activation(self, ac):
        self.activation = ac

    def _set_value(self, v):
        self.current_value = v


    def activate(self):
        total = 0
        for i in range(len(self.upstream_neurons)):
            neuron = self.upstream_neurons[i]
            weight = self.upstream_weights[i]
            total += neuron.current_value * weight
        total += self.bias
        self.current_z = total
        total = self.activation.activate(total)
        self.current_value = total


class NeuralNet(object):
    
    def __init__(self):
        self.layers = []
        self.input_size = 0
        self.output_size = 0
        self.n_layers = 0

    def _add_layer(self, n, f):
        self.layers.append([Neuron(f) for _ in range(n)])

    def _fully_connect_layers(self, l1, l2, weights):
        for i in range(len(l1)):
            for j in range(len(l2)):
                input_neuron = l1[i]
                output_neuron = l2[j]
                weight = weights[j][i]
                input_neuron._link_output(output_neuron, weight)

    # Spec of the format [("identity|sigmoid|tanh|relu", n)]
    # Initialize a neural net with all weights 1 and all biases 0 
    def build_from_spec(self, spec):
        for (activation, n) in spec:
            if activation not in activation_types:
                print(f"Unrecognized activation function: {activation}")
                return
            ac = activation_types[activation]
            self._add_layer(n, ac)
        for i in range(len(self.layers) - 1):
            l1 = self.layers[i]
            l2 = self.layers[i + 1]
            n = len(l1)
            m = len(l2)
            weights = [[1 for _ in range(n)] for _ in  range(m)]
            self._fully_connect_layers(l1, l2, weights)
        self.input_size = len(self.layers[0])
        self.output_size = len(self.layers[-1])
        self.n_layers =  len(self.layers)


    # input_vector is just a list-like object that can be indexed
    def passthrough(self, input_vector):
        if len(input_vector) != self.input_size:
            print(f"Mismatched input: expected size {self.input_size} but got size {len(input_vector)}")
        for i in range(self.input_size):
            self.layers[0][i]._set_value(input_vector[i])
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.activate()
        output_vector = [neuron.current_value for neuron in self.layers[-1]]
        return output_vector
        

            
                
        
        
    
