#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
	print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
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

# Model initializer function : Initializes 
# 1. Initial distribution of Latent states (PI) of size 1 x clusters
# 2. Transition probability matrix (A) of size clusters x clusters
# 3. Emission distribution parameters 
# 	3.a. mus : mean of the gaussian densities for each cluster. This will be a matrix of size clusters x dimensionality
#	3.b sigmas : sigmas of the gaussian densities for each cluster. This will be a tensor of size clusters x dimensionilty x dimensionality when args.tied is not provided otherwise it will be matrix of size dimensionality x dimensionality
def init_model(args, train_xs, dev_xs):
	if args.cluster_num:
		dimensions = train_xs.shape[1]
		mu_dimensions = np.mean(train_xs, axis = 0)
		sd_noise = np.std(train_xs, axis = 0)

		#TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
		mus = np.array(mu_dimensions + np.random.rand(args.cluster_num, dimensions) * sd_noise)
		initials = np.repeat(1/args.cluster_num, args.cluster_num) #probability for starting in each state
		transitions = np.random.rand(args.cluster_num, args.cluster_num) #transitions[i][j] = probability of moving from cluster i to cluster j
		# Normalize Transitions to add to 1 row-wise
		transitions = transitions/np.sum(transitions, axis = 1, keepdims = True) # IMPORTANT Note: Keepdims = True makes sure that we get a column vector after summing each row; if keepdims = False => then np.sum will result in row-vector

		if not args.tied:
			sigmas = np.array([np.eye(dimensions)] * args.cluster_num)
		else:
			sigmas = np.eye(dimensions)
	else:
		mus = []
		sigmas = []
		transitions = []
		initials = []
		with open(args.clusters_file,'r') as f:
			for line in f:
				#each line is a cluster, and looks like this:
				#initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
				vals = list(map(float,line.split()))
				initials.append(vals[0])
				mus.append(vals[1:3])
				sigmas.append([vals[3:5],vals[5:7]])
				transitions.append(vals[7:])
		initials = np.asarray(initials)
		transitions = np.asarray(transitions)
		mus = np.asarray(mus)
		sigmas = np.asarray(sigmas)
		args.cluster_num = len(initials)

	#TODO: Pack mus, sigmas, initals, and transitions into the model variable (Dictionary of these variables)
	model = {'initials' : initials, 'transitions' : transitions, 'mus' : mus, 'sigmas' : sigmas}
	#print('π - \n{0},\n A - \n{1},\n mus - \n{2},\n sigmas - \n{3}'.format(initials, transitions, mus, sigmas))
	return model

# Forward function to calculate the alphas(α)
def forward(model, data, args):
	from scipy.stats import multivariate_normal
	from math import log

	# Extract Model
	initials, transitions, mus, sigmas = extract_parameters(model)

	N, dimensions = data.shape
	emission_matrix = np.zeros((N, args.cluster_num))
	alphas = np.zeros((N, args.cluster_num))

	#TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
	#NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.

	# 1. Non-Vectorized Implementation
	log_likelihood = 0.0
	c = np.zeros(N)
	for n in range(0, N):
		for k in range(0, args.cluster_num):
			if not args.tied:
				emission = multivariate_normal(mean = mus[k], cov = sigmas[k]).pdf(data[n, :])
			else:
				emission = multivariate_normal(mean = mus[k], cov = sigmas).pdf(data[n, :])
			
			if n == 0:
				alphas[n, k] = initials[k] * emission
			else:
				alphas[n, k] = emission * np.sum(alphas[n - 1, :] * transitions[:, k])

			# Storing emissions. These will be used in backward step
			emission_matrix[n, k] = emission
		# Re-Normalize alphas
		row_norm = np.sum(alphas[n, :])
		alphas[n, :] = alphas[n, :]/row_norm
		log_likelihood += np.log(row_norm)
		c[n] = row_norm

	# 2. Vectorized Implementation
	log_likelihood = 0.0
	c = np.zeros(N)
	# Calculate emission-matrix (Could be vectorized by scipy.stats.multivariate_normal function allows vector means)
	for k in range(0, args.cluster_num):
		if not args.tied:
			emission_matrix[:, k] = multivariate_normal(mean = mus[k], cov = sigmas[k]).pdf(data)
		else:
			emission_matrix[:, k] = multivariate_normal(mean = mus[k], cov = sigmas).pdf(data)

	# Forward algorithm to calculate alphas
	for n in range(0, N):
		if n == 0:
			alphas[n, :] = initials * emission_matrix[n, :]
		else:
			alphas[n, :] = emission_matrix[n, :] * np.dot(alphas[n - 1, :], transitions)
		# Re-Normalize alphas
		row_norm = np.sum(alphas[n, :])
		alphas[n, :] = alphas[n, :]/row_norm
		c[n] = row_norm
	
	log_likelihood = np.sum(np.log(c))

	# print(alphas, '\nll-', log_likelihood, '\nem-', emission_matrix)
	return alphas, log_likelihood, emission_matrix, c

# Backward function to calculate the betas(β)
def backward(model, data, args, data_emission_matrix):
	from scipy.stats import multivariate_normal

	# Extract Model
	initials, transitions, mus, sigmas = extract_parameters(model)

	N, dimensions = data.shape
	emission_matrix = data_emission_matrix

	betas = np.zeros((N, args.cluster_num))
	#TODO: Calculate and return backward probabilities (normalized like in forward before)
	# 1. Non-vectorized implementation
	for n in range(N - 1, -1, -1):
		for k in range(0, args.cluster_num):
			if n == N - 1:
				betas[n, k] = 1
			else:
				betas[n, k] = np.sum(emission_matrix[n + 1, :] * betas[n + 1, :] * transitions[k, :])
		# Re-Normalize betas
		row_norm = np.sum(betas[n, :])
		betas[n, :] = betas[n, :]/row_norm

	# 2. Vectorized implementation
	for n in range(N - 1, -1, -1):
		if n == N - 1:
			betas[n, :] = 1
		else:
			betas[n, :] = np.dot(transitions, emission_matrix[n + 1, :] * betas[n + 1, :])
		# Re-Normalize betas
		row_norm = np.sum(betas[n, :])
		betas[n, :] = betas[n, :]/row_norm

	return betas

def train_model(model, train_xs, dev_xs, args):
	from scipy.stats import multivariate_normal

	N, dimensions = train_xs.shape
	initials, transitions, mus, sigmas = extract_parameters(model)

	#TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
	# ------------------------------------------ Iterations Loop Begins ----------------------------------- #
	for itr in range(0, args.iterations):

		# ------------------- E-Step Begins (Objective: to calculate gamma-γ and psi-ξ) ------------------- #
		alphas, ll, em, c = forward(model, train_xs, args)
		betas = backward(model, train_xs, args, em)
		
		# gamma-γ Calculatons
		gammas = alphas * betas
		# Re-Normalize betasi
		row_norm = np.sum(gammas, axis = 1, keepdims = True)
		gammas = gammas/row_norm


		# psi-ξ calculations - Non-Vectorized
		psi_tensor = np.zeros((args.cluster_num, args.cluster_num, N))
		for t in range(1, N):
			for j in range(0, args.cluster_num):
				for k in range(0, args.cluster_num):
					psi_tensor[j, k, t] = alphas[t - 1, j] * transitions[j, k] * betas[t, k] * em[t, k]
					# psi_tensor[i, j, t] = alphas[t - 1, j] * transitions[j, i] * betas[t, i] * em[t, i]
			psi_tensor[:, :, t] /= np.sum(psi_tensor[:, :, t])
		
		
		# psi-ξ calculations - Vectorized
		# for t in range(1, N):
		# 	psi_tensor[:, :, t] = 1/c[t] *alphas[t-1, :].dot(betas[t, :]) * transitions

		# 	# Multiply emissions
		# 	for k in range(args.cluster_num):
		# 		psi_tensor[:, k, t] *= em[t, k]
		
		# print(psi_tensor)
		# ----------------------------------------- E-Step ends ------------------------------------------- #


		# -------- M-Step Begins (Objective: Update PI-π, A and emission params - mus, sigmas)------------- #
		# π-updates (Initial distributions)
		initials = gammas[0, :]/np.sum(gammas[0, :], keepdims = True)
		
		# A-updates (Transition probability matrix)
		psiSum = np.sum(psi_tensor[:, :, 1:], axis=2)
		transitions = psiSum / np.sum(gammas, axis = 0, keepdims = True).T

		# B-emission distribution parameters updates i.e. means and sigmas
		# μ-updates
		gammaSum = np.sum(gammas, axis = 0, keepdims = True)
		mus = np.dot(gammas.T, train_xs)/gammaSum.T
		
		# Σ-updates
		for k in range(0, args.cluster_num):
			# Sigma-updates [TIED]
			if args.tied:
				diff = train_xs - mus[k]
				sigmas += np.dot(diff.T, gammas[:, k].reshape((N, 1)) * diff)
			# Sigma-updates [UN-TIED]
			else:
				diff = train_xs - mus[k]
				sigmas[k] = np.dot(diff.T, gammas[:, k].reshape((N, 1)) * diff)
				sigmas[k] = sigmas[k]/np.sum(gammas[:, k])
		
		if args.tied:
			sigmas = sigmas/N
		# ----------------------------------------- M-Step Ends ------------------------------------------- #
		model = {'initials' : initials, 'transitions' : transitions, 'mus' : mus, 'sigmas' : sigmas}

	# ---------------------------------------------Iterations end ------------------------------------------#
	return model

def average_log_likelihood(model, data, args):
	#TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
	#NOTE: yes, this is very simple, because you did most of the work in the forward function above
	
	N, dimensions = data.shape
	initials, transitions, mus, sigmas = extract_parameters(model)

	alphas, log_likelihood, emission_matrix, c = forward(model, data, args)
	average_ll = log_likelihood/N

	return average_ll

def extract_parameters(model):
	#TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
	initials = model['initials']
	transitions = model['transitions']
	mus = model['mus']
	sigmas = model['sigmas']
	return initials, transitions, mus, sigmas

def main():
	import argparse
	import os
	print('Gaussian') #Do not change, and do not print anything before this.
	parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
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
	nll_train = average_log_likelihood(model, train_xs, args)
	print('Train LL: {}'.format(nll_train))
	if not args.nodev:
		nll_dev = average_log_likelihood(model, dev_xs, args)
		print('Dev LL: {}'.format(nll_dev))
	initials, transitions, mus, sigmas = extract_parameters(model)
	if args.print_params:
		def intersperse(s):
			return lambda a: s.join(map(str,a))
		print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
		print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
		print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
		if args.tied:
			print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
		else:
			print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
	main()
