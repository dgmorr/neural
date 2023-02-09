import math

class Activation(object):
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def activate(self, x):
        return self.f(x)

    def deriv(self, x):
        return self.f_prime(x)

def identity_f(x):
    return x
def identity_fprime(x):
    return 1
identity = Activation(identity_f, identity_fprime)

def sigmoid_f(x):
    return 1 / (1 + math.e ** (-x))
def sigmoid_fprime(x):
    return sigmoid(x) * (1 - sigmoid(x))
sigmoid = Activation(sigmoid_f, sigmoid_fprime)

def tanh_f(x):
    z = math.e ** 2*x
    return (z - 1) / (z + 1)
def tanh_fprime(x):
    return 1 - tanh(x) ** 2
tanh = Activation(tanh_f, tanh_fprime)

def relu_f(x):
    return max(0, x)
def relu_fprime(x):
    if x < 0:
        return 0
    return 1
relu = Activation(relu_f, relu_fprime)

def leaky_relu_f(x):
    if x >= 0:
        return x
    return x * 0.01
def leaky_relu_fprime(x):
    if x < 0:
        return 0.01
    return 1
leaky_relu = Activation(leaky_relu_f, leaky_relu_fprime)

activation_types = {
    "identity": identity,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}


