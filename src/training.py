# Cost function for an individual training example
def squared_error_f(ex, ac):
    assert len(ex) == len(ac)
    diff_sq = [(ex_i - ac_i) ** 2 for (ex_i, ac_i) in zip(ex, ac)]
    return 0.5 * sum(diff_sq)
# Derivative WRT particular activation ac_i
def squared_error_fprime(ex_i, ac_i):
    return ac_i - ex_i

# 1. Error for each output neuron
def output_error(net, ex)
    delta_L = []
    for j in range(len(net.layers[-1])):
        neuron = net.layers[-1][j]
        a_j = neuron.current_value
        z_j = neuron.current_z
        y_j = ex[j]
        delta_j = (a_j - y_j) * neuron.activation.deriv(z_j)
        delta_L.append(delta_j)
    return delta_L
        


    
