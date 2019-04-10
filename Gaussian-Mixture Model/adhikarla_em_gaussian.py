#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

DATA_PATH = "C:/Users/shrik/Downloads/UofR/Coursework/CSC 446 - Machine Learning/Assignments/Assignment-7/em/"

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args, train_xs, dev_xs):
    clusters = []
    if args.cluster_num: # If the number of clusters is provided as an argument
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        dimensions = train_xs.shape[1]
        mu_dimensions = np.mean(train_xs, axis = 0)
        sd_dimensions = np.std(train_xs, axis = 0)

        lambdas = np.repeat(1/args.cluster_num, args.cluster_num)

        mus = np.array(mu_dimensions + np.random.rand(args.cluster_num, 2) * sd_dimensions)
        if not args.tied:
            sigmas = np.array([np.eye(dimensions)] * args.cluster_num)
        else:
            sigmas = np.eye(dimensions)
        
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2]) # Mean Vector for each cluster will consist of d-dimensions (mean values in d-dimensions i.e. 2 in this case)
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]]) # Covariance matrix of each cluster will consist of d x d matrix of variances and covariances
        lambdas = np.asarray(lambdas) # Lambda for each cluster will be just a single value i.e. weight associated with each posterior probability calculated P(N_1|Xi)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = {'lambdas' : lambdas, 'mus' : mus, 'sigmas' : sigmas}
    #print(model)
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    lambdas, mus, sigmas = extract_parameters(model)

    N, dimensions = train_xs.shape
    cluster_num = len(lambdas)

    # ------------------- PLOTTING ----------------------------- #
    try:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.scatter(train_xs[:, 0], train_xs[:, 1], marker = ',', c = 'green', alpha = 0.2)
        plt.title("Cluster-Centers representation for {0} clusters and {1} Iterations-{2} in X-Space".format(cluster_num, args.iterations, '[Tied]' if args.tied else '[Un-Tied]'))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.scatter(mus[:, 0], mus[:, 1], marker = 'v', c = 'black')

    except:
        pass
    # --------- PLOTTING TO BE CONTINUED IN CHUNKS -------------- #


    train_ll_history = []
    dev_ll_history = []

    # ------ Loop through iterations (TRAINING BEGINS) --------- #
    for i in range(0, args.iterations):

        #-------------------- E-Step Begins--------------------- #
        eta_matrix = np.zeros((N, cluster_num))
        for k in range(0, cluster_num):
            if args.tied:
                p_z_n = multivariate_normal(mean = mus[k], cov = sigmas).pdf(train_xs)
            else:
                p_z_n = multivariate_normal(mean = mus[k], cov = sigmas[k]).pdf(train_xs)
            eta_matrix[:, k] = lambdas[k] * p_z_n

        # Normalizing the eta_matrix, so that the columns add up to 1
        eta_matrix = eta_matrix/np.sum(eta_matrix, axis = 1).reshape((N, 1))
        # ------------------- E-Step ends ----------------------- #
        
        # ------------------- M-Step Begins---------------------- #
        # Lambda-updates
        lambdas = np.sum(eta_matrix, axis = 0)/N
        
        for k in range(0, cluster_num):
            # Mu-updates
            mus[k] = np.dot(eta_matrix[:, k].T, train_xs)/np.sum(eta_matrix[:, k])

            # Plotting (This try-except code chunk can be ignored)
            try:
                plt.scatter(mus[k][0], mus[k][1], marker = 'o', c = k)
            except:
                pass
            # Until here

            # Sigma-updates [TIED]
            if args.tied:
                diff = train_xs - mus[k]
                sigmas += np.dot(diff.T, eta_matrix[:, k].reshape((N, 1)) * diff)                
            # Sigma-updates [UN-TIED]
            else:
                diff = train_xs - mus[k]
                sigmas[k] = np.dot(diff.T, eta_matrix[:, k].reshape((N, 1)) * diff)
                sigmas[k] = sigmas[k]/np.sum(eta_matrix[:, k])

        if args.tied:
            sigmas = sigmas/N
        # ------------------- M-Step Ends ----------------------- #

        model = {'lambdas' : lambdas, 'mus' : mus, 'sigmas' : sigmas}
        train_ll_history.append(average_log_likelihood(model, args, train_xs))
        if not args.nodev:
            dev_ll_history.append(average_log_likelihood(model, args, dev_xs))
    # ------------------------ TRAINING COMPLETED --------------- #


    # --------------------- PLOTTING ---------------------------- #
    # 1. Plotting Gaussian Contours for each cluster
    try:
        import matplotlib
        import matplotlib.cm as cm
        import matplotlib.mlab as mlab
        import matplotlib.pyplot as plt

        delta = 0.025
        x = np.arange(-11.0, 8.0, delta)
        y = np.arange(-8.0, 6.0, delta)
        X, Y = np.meshgrid(x, y)
        for k in range(0, args.cluster_num):
            if args.tied:
                z_k = mlab.bivariate_normal(X, Y, sigmax = sigmas[0, 0], sigmay = sigmas[1, 1], mux = mus[k, 0], muy = mus[k, 1], sigmaxy = sigmas[0, 1])
            else:
                z_k = mlab.bivariate_normal(X, Y, sigmax = sigmas[k][0, 0], sigmay = sigmas[k][1, 1], mux = mus[k, 0], muy = mus[k, 1], sigmaxy = sigmas[k][0, 1])
            CS = plt.contour(X, Y, z_k, 6, colors='k', alpha = 0.5)
            plt.clabel(CS, fontsize=9, inline=1)
        # Add custom legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], color='black', lw=1, alpha = 0.5, label='Contours'),
        Line2D([0], [0], marker='s', color='w', label='Data Point', markerfacecolor='green', markersize=10, alpha = 0.5),
        Line2D([0], [0], marker='v', color='w',  label='Initial Cluster-Centers', markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w',  label='Cluster-centers trace', markerfacecolor='black', markersize=10)]

        plt.legend(handles=legend_elements, loc='lower right')

        # 2. Plotting Accuracy vs log-likelihood plot for training and development set
        plt.subplot(1, 2, 2)
        plt.plot(list(range(1, args.iterations + 1)), train_ll_history, label = "Train")
        if not args.nodev:
            plt.plot(list(range(1, args.iterations + 1)), dev_ll_history, label = "Dev")
        plt.xlabel("# iterations \n {0}".format("Max-DevLL ({0}, {1})".format(max(dev_ll_history), np.argmax(dev_ll_history)) if not args.nodev else ""))
        plt.ylabel("Log-Likelihood for data")
        plt.title("Accuracy vs Log-Likelihood for {0} clusters and {1} Iterations-{2}".format(cluster_num, args.iterations, '[Tied]' if args.tied else '[Un-Tied]'))
        plt.legend(loc = 'lower right')
        plt.savefig("Sub2adhikarla_em_gaussian-k-{0}-itr-{1}-{2}.png".format(cluster_num, args.iterations, 'tied' if args.tied else 'untied'))
        plt.show()
    except:
        pass
    # -------------------- PLOTTING COMPLETED --------------------- #
    return model

def average_log_likelihood(model, args, data):
    from math import log
    from scipy.stats import multivariate_normal

    N = data.shape[0]
    p_z, mus, sigmas = extract_parameters(model)
    cluster_num = len(p_z)
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    p_each_data_point = np.zeros(N)
    
    for k in range(0, cluster_num):
        if args.tied:
            p_x_given_zk = multivariate_normal(mean = mus[k], cov = sigmas).pdf(data)
        else:
            p_x_given_zk = multivariate_normal(mean = mus[k], cov = sigmas[k]).pdf(data)
        p_x_given_zk = p_z[k] * p_x_given_zk
        p_each_data_point += p_x_given_zk
    
    ll = np.sum(np.log(p_each_data_point))/N
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model['lambdas']
    mus = model['mus']
    sigmas = model['sigmas']
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args, train_xs, dev_xs)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, args, train_xs)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, args, dev_xs)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
