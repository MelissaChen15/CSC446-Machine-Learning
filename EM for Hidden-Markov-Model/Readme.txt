Name - Shrikant Adhikarla
email - sadhika4@ur.rochester.edu
URID - 31407229
Coursework - CSC446-Machine Learning
Assignment - 8
Topic - Implement Hidden Markov model with gaussian emissions using Expectation-Maximization(EM implementation from the scratch)


--------------- Objective ------------------------------------
		1. Implement EM to train an HMM for the points.dat(a 2D dataset) dataset using Python. 
		2. Experiment: see how Log-Likelihood of data changes when you vary the hyperparameters, including number of iterations, tied/untied covariance matrices and number of hidden states in HMM. Use development data to choose the best set of hyperparameters.
		3. Answer the question : Does the HMM model the data better than the original non-sequence model? What is the best number of states?
		4. Discuss the interpretation of the results.
		5. Create a README file for the same.
---------------------------------------------------------------


--------------- Files (included in submission) ---------------
1. adhikarla_hmm_gaussian.py : This is the Expectation-Maximization implementation python3 file to train a Hidden Markov Model which has a gaussain parameterized emissions. 
2. README : Currently, open file.
3.1. Sub2adhikarla_hmm_gaussian-k-2-itr-250-tied.png : 2-plots attached side-by side. 
		FIRST plot shows the spread of the datapoints in X-Space and also, provides a sense of fitted cluster centers movement as the iterations increase and their respective final contour plots of the last fitted gassian distribution for each cluster. 
		SECOND plot shows the number of iterations vs. Log-Likelihood. This provides an idea to cut-off the number of iterations where the dev set likelihood reaches a maximum.   
3.2. Sub2adhikarla_hmm_gaussian-k-2-itr-250-untied.png
3.3. Sub2adhikarla_hmm_gaussian-k-3-itr-250-tied.png
3.4. Sub2adhikarla_hmm_gaussian-k-3-itr-250-untied.png
3.5. Sub2adhikarla_hmm_gaussian-k-4-itr-250-tied.png
3.6. Sub2adhikarla_hmm_gaussian-k-4-itr-250-untied.png
3.7. Sub2adhikarla_hmm_gaussian-k-5-itr-250-tied.png
3.8. Sub2adhikarla_hmm_gaussian-k-5-itr-250-untied.png
--------------------------------------------------------------


--------- Discuss the interpretation of results ---------------
Way of coding and a few extras:
I have coded the entire EM algorithm to train hidden markov model in a very concise and vectorized way which doesn't require many loops through each data point in the data. There are a few loops which cannot be avoided as the calculations require us to address time period which are dissimilar and vectorized dot products cannot directly compute them unless the matrices themselves are modified and shifted by 1 row/columns. The codes also contain some matplotlib plotting related code chunks which are encapsulated inside try and except statement. If the optional libraries are not available/installed, then the code will automatically ignore the plotting related code chunks and jump to the necessary components.

Usage of Development Dataset: 
Looking at the results of the number of iterations vs. Log-Likelihood plot with both training and development data. We can see clearly that the results vary according to the hyperparamters (mainly number of states in the state space of the hidden variables which is very similar to the gaussian mixture model as long as the number of iterations are large enough), there are some random effects related to the intialization of cluster centroids/means. The trajectory of cluster centres and also their shapes(Covariance matrices) depend a lot on the initialization. If the number of states are small enough, then we can see that the gaussians of the states(or clusters) always converge in a small number of iterations. When we increase the number of states, then the algorithm converges but we also need to run more number of iterations to achieve the same converging means and sigmas. Also, tied covariances tend to perform somewhat poorly when compared to their untied counter-parts. This is comprehensible as the shape of all the k-gaussians need to be the same if tied covariance were chosen which will restrict the learning of differences in the shapes of the underlying gaussians. Plots are attached as PNG images along with this readme files.

Interpretation of experiments:
1. 2-state; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each state
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 200 iterations{This varies from 150-200 on multiple runs of algorithm} Please look at the text below the x-axis label to get the maximum-likelihood of dev dataset and the iteration at which it was achieved.
2. 2-state; 250-iterations; tied-covariances; Random-initialization of centres; Initial-Identity Covariance matrix for all state
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 70 iterations{This varies from 10-100 on multiple runs of algorithm}. I have also noticed a few cases when the dev-likelihood in tied case is higher than untied case.
3. 3-state; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each state
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 36 iterations{This varies from 20-60 on multiple runs of algorithm}. The contours also look right on target for the most runs and seem to have the best likelihood without increasing the complexity of state either (more state = more complex)
4. 3-state; 250-iterations; tied-covariances; Random-initialization of centres; Initial-Identity Covariance matrix for all state
	The maximum log-likelihood of dev-dataset in this case is lower than (3) where we enforced untied covariances which is expected as the enforcement of tied covariances restricts the shape of the clusters.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 15 iterations{This varies from 10-25 on multiple runs of algorithm}
5. 4-state; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each state
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 73 iterations{This varies from 50-100 on multiple runs of algorithm}. The contours also look right on target for the most runs and seem to have the best likelihood without increasing the complexity of state either (more state = more complex). I believe this might be the best possible approximation of the underlying distribution and this set of hyperparameters gives the highest log-likelihood among all the other choice of hyperparameters.
6. 4-state; 250-iterations; tied-covariances; Random-initialization of centres; Initial-Identity Covariance matrix for all state
	The maximum log-likelihood of dev-dataset in this case is lower than (5) where we enforced untied covariances as expected in almost all the cases.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 132 iterations{This varies from 120-150 on multiple runs of algorithm}
7. 5-state; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each state
	In my experimentation, this results achieved the very high log-likelihood estimates for the dev-dataset really close to the experiment(5). The Contour plots also look right on target and data seems to have these kind of clustered points in X-space. However, looking at the haphazard nature of (iterations vs LL) plots in multiple iterations, I am inclined to believe that the number of state may be too many and might be causing some random effects and complicated gaussian model. 
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 200 iterations{Actually, it never stops increasing, but the increments become negligibly small} 
8. 5-state; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each state
	The algorithm generates 5 gaussian contours in the X-space. However, the left most gassian contour doesn't seem to have enough points in the centre. This might point us to wonder that there may not be a 4th gaussian in the underlying distribution of data.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 64 iterations{This varies from 20-70 on multiple runs of algorithm}

Best Results on points.dat dataset : 
#Iterations : 50-100 (in our case the early stopping at ~73 iterations gives the best results)
#States in state-space of hidden variables: 4
Covariances of emission distribution : Untied

--- Answer the question : Does the HMM model the data better than the original non-sequence model? What is the best number of states?
Yes, the HMM using EM seems to provide higher log-likelihood for the same hyperparameters when compared with the Gaussian mixture model. Also, it achieved a very high log-likelihood of -3.70998 at 4 clusters and untied covariance matrices. The gaussian mixture model never achieved anything higher than -3.9 in any cases that we have tested. 
This can be expected as HMM is a more sophisticated model and the hidden variables have more adjustable hyperparameters which makes it more robust and gets closer to the real distribution when tuned with rightly chosen hyperparameters.
---------------------------------------------------------------


--------------------- Instructions ----------------------------
adhikarla_hmm_gaussian.py is the main executable script which is a python3 file and requires a few arguments for running:

1. --nodev : If provided, no dev data will be used. If not provided, dev data is used by default.
2. --iterations : An positive integer argument is required along with the keyword, which limits the maximum number of iterations the algorithm goes through before stopping. [Default = 1]
3. --data_file : Requires a string argument which will contain the location of the training data file. By default, the codes will pick up the file from "/u/cs246/data/em/" from CSUG server.
4. --cluster_num : A Positive integer argument is required along with this keyword i.e. User-defined number of state.
5. --cluster_file : Requires a string argument which will contain the location of the information pertaining to the initialization of states. Please note that only one of the arguments out of (4) or (5) needs to be provided for execution and NOT both.
7. --print_params : Doesn't require any extra argument. enabling this prints the final lambdas, mus and sigmas for each cluster on the console screen. 
8. --tied : If provided, all the states are assumed to have same covariance matrices and consequently same shape. 

Example commands : 
1. python3 adhikarla_hmm_gaussian.py --cluster_num 2 

2. python3 adhikarla_hmm_gaussian.py --clusters_file '/home/hoover/u8/sadhika4/ml_hw7/em_smoke_test_files/gaussian_smoketest_clusters.txt'

3. python adhikarla_hmm_gaussian.py --cluster_num 5 --iterations 250 --print_params --tied

4. python adhikarla_hmm_gaussian.py --cluster_num 3 --iterations 250 --print_params
----------------------------------------------------------------
