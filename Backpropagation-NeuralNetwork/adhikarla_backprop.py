#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are
DATA_PATH = "C:\\Users\\shrik\\Downloads\\UofR\\Coursework\\CSC 446 - Machine Learning\\Assignments\\Assignment-2\\inputs" 

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.


    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    model = (w1, w2)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model

def sigmoid_activation(x):
    # Required Argument(s) : 
    # x : integer or real value or vector or even matrix
    
    # This function returns the sigmoid activation value
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_d(sig):
    return sig * (1.0 - sig)

def forward_prop(model, X):
    w1, w2 = extract_weights(model)

    # Layer-1 Computation
    a1 = np.dot(w1, X)
    try:
        z1 = sigmoid_activation(a1).reshape(a1.shape[0], 1)
    except:
        z1 = sigmoid_activation(a1)
    # Layer-2 Computation
    bias_row = np.ones((1, z1.shape[1]))
    #print("z1 ", z1.shape, " bias ", bias_row.shape)
    extended_z1 = np.concatenate((z1, bias_row), axis = 0)
    a2 = np.dot(w2, extended_z1)
    z2 = sigmoid_activation(a2)
    # Cache to store the values (to be used in Backward propagation)
    cache = {"a1" : a1, "z1" : z1, "extended_z1" : extended_z1, "a2" : a2, "z2" : z2}

    return z2, cache

def calculate_error(y_hat, y):
    error = 0.5 * np.sum(np.square(y - y_hat))
    return error

def predict(model, X):
    y_hat, c = forward_prop(model, X)
    print(y_hat)
    y_hat = (np.squeeze(y_hat) > 0.50).astype(int)
    print(y_hat)

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    #TODO: Implement training for the given model, respecting args
    #raise NotImplementedError #TODO: delete this once you implement this function
    model = init_model(args)
    w1, w2 = model
    N = train_ys.shape[0]
    hidden_dim = w2.shape[1] - 1

    train_acc_history = list()
    dev_acc_history = list()

    for i in range(args.iterations):
        for n in range(N):
            x_vector = train_xs[n].reshape(train_xs[n].shape[0], 1)

            # Forward Propagation step
            y_hat, cache = forward_prop(model, x_vector)
            error = calculate_error(y_hat, train_ys[n])
            #print("Itr-", i, " n-", n, " y_hat-",y_hat, " y-",train_ys[n], "err--", error)

            # Backward Propagation step (Calculating gradients)
            del2 = (y_hat - train_ys[n]) * sigmoid_d(cache["z2"])
            #print("del2---------------------", del2)
            dw2 = np.dot(del2, cache["extended_z1"].T)
            reduced_w2 = w2[:, 0:hidden_dim]
            del1 = del2 * (reduced_w2.T * sigmoid_d(cache["z1"]))
            #print("del1---------------------", del1)
            dw1 = np.dot(del1, x_vector.T)
            
            # Weight matrices update
            w2 = w2 - args.lr * dw2
            w1 = w1 - args.lr * dw1
            model = (w1, w2)

        train_acc = test_accuracy(model, train_ys, train_xs)
        #print(i, "-iteration Train Accuracy:", train_acc)
        train_acc_history.append(train_acc)

        if not args.nodev:
            dev_acc = test_accuracy(model, dev_ys, dev_xs)
            dev_acc_history.append(dev_acc)

        #print(i, "-iteration Dev Accuracy:", dev_acc)
        # z2 = forward_prop(model, train_xs.T)[0]
        # print("Error/Cost after {0} iteration : {1}".format(i, calculate_error(z2, train_ys.T)/N))
        #print("Train accu", test_accuracy(model, train_ys, train_xs), " Dev", test_accuracy(model, dev_ys, dev_xs))
    
    import matplotlib.pyplot as plt
    # Plotting Accuracy vs Iterations plot for training and development set
    plt.plot(list(range(1, args.iterations + 1)), train_acc_history, label = "Training")
    if not args.nodev:
        plt.plot(list(range(1, args.iterations + 1)), dev_acc_history, label = "Dev")
    plt.xlabel("# iterations")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs iterations for {1} hidden nodes and Learning rate:{0}".format(args.lr, hidden_dim))
    plt.legend()
    plt.savefig("adhikarla_backprop-n_h-{1}-lr-{0}.png".format(args.lr, hidden_dim))
    plt.show()
    # CAN BE COMMENTED SAFELY (ALGORITHM WILL STILL WORK)

    #print(model)
    return model

def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    N = test_xs.shape[0]
    y_hat = forward_prop(model, test_xs.T)[0]
    
    y = test_ys.T == 1
    y_hat = y_hat >= 0.5
    accuracy = np.sum(y == y_hat)
    accuracy = accuracy/N
    return accuracy

def extract_weights(model):
    w1 = model[0]
    w2 = model[1]
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    
    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
