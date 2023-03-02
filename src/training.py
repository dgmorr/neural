import random

# t = training set of the form [(x, y)]
def mse(t):
    total_error = 0
    for (x, y) in t:
        total_error += squared_error(x, y, net)
    return total_error / len(t)

def squared_error(x, y, net):
    a = net.passthrough(x)
    return 0.5 * sum([(a[i] - y[i])**2 for i in range(len(a))])

def squared_error_prime(aL, y):
    return [aL[i] - y[i] for i in range(len(aL))]

# x and y are a training example
# cost_x_prime is an arbitary single-example cost function derivative 
# that takes an output vector a^L and an expected vector y
def final_layer_error(net, x, y, cost_x_prime):
    aL = net.passthrough(x)
    dC_da = cost_x_prime(aL, y)
    sigma_prime_zL = [neuron.activation.deriv(neuron.pre_activation_value) for neuron in net.layers[-1]]
    deltaL = [dC_da[i] * sigma_prime_zL[i] for i in range(len(dC_da))]
    return deltaL

def layer_error(net, x, y, cost_x_prime):
    deltaL = final_layer_error(net, x, y, cost_x_prime)
    error_vectors = [deltaL]
    for l in range(2, net.n_layers):
        layer = net.layers[-l]  # start with layers[-2], work back to layers[-n_layers + 1]
        # First layer does not have an error associated with it because it doesn't have input weights
        weights_transpose = [neuron.downstream_weights for neuron in layer]
        next_delta = error_vectors[0]
        weight_delta_product = []
        for weights in weights_transpose:
            weight_delta_product.append(sum([weights[i] * next_delta[i] for i in range(len(next_delta))]))
        sigma_prime_zl = [neuron.activation.deriv(neuron.pre_activation_value) for neuron in layer]
        deltal = [weight_delta_product[i] * sigma_prime_zl[i] for i in range(len(sigma_prime_zl))]
        error_vectors = [deltal] + error_vectors
    return error_vectors

# No need for (3) as bias gradient for each layer is just the error vector for that layer
def weight_gradient(net, error_vectors):
    weight_derivs = []
    for l in range(1, net.n_layers):
        deltal = error_vectors[l - 1]
        layer = net.layers[l]  # No need to go backwards this time
        prev_layer = net.layers[l - 1]
        layer_derivs = []
        for j in range(len(layer)):
            layer_derivs_j = []
            for k in range(len(prev_layer)):
                prev_a_k = prev_layer[k].activation_value
                deltal_j = deltal[j]
                dC_dwjk = prev_a_k * deltal_j
                layer_derivs_j.append(dC_dwjk)
            layer_derivs.append(layer_derivs_j)
        weight_derivs.append(layer_derivs)
    return weight_derivs

def test_train_split(dataset, p):
    n = len(dataset)
    take = n * p
    copy = dataset.copy()
    random.shuffle(copy)
    return (copy[:take], copy[take:])

def batch_data(training_data, n_batches=None, batch_size=0)
    batches = []
    n = len(training_data)
    copy = training_data.copy()
    random.shuffle(copy)
    if n_batches:
        batch_size = n / n_batches
    if not batch_size:
        return [copy]
    while copy:
        batches.append(copy[:batch_size])
        copy = copy[batch_size:]
    return batches




    
