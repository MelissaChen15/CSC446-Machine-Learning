Name - Shrikant Adhikarla
email - sadhika4@ur.rochester.edu
URID - 31407229
Coursework - CSC446-Machine Learning
Assignment - 7 
Topic - Implement Gaussian Mixture model using Expectation-Maximization


--------------- Objective ------------------------------------
		1. Implement Gaussian Mixture model using Expectation-Maximization for the points.dat dataset using Python. 
		2. Experiment: see how Log-Likelihood of data changes when you vary the hyperparameters, including number of iterations, tied/untied covariance matrices and number of clusters. Use development data to pick the best set of hyperparameters.
		3. Discuss the interpretation of the results.
		4. Create a README file for the same.
---------------------------------------------------------------


--------------- Files (included in submission) ---------------
1. adhikarla_em_gaussian.py : This is the Expectation-Maximization implementation python3 file for Gaussian mixture model.
2. README : Currently, open file.
3.1. Sub2adhikarla_em_gaussian-k-2-itr-250-tied.png : 2-plots attached side-by side. 
		FIRST plot shows the spread of the datapoints in X-Space and also, provides a sense of fitted cluster centers movement as the iterations increase and their respective final contour plots of the last fitted gassian distribution for each cluster. 
		SECOND plot shows the number of iterations vs. Log-Likelihood. This provides an idea to cut-off the number of iterations where the dev set likelihood reaches a maximum.   
3.2. Sub2adhikarla_em_gaussian-k-2-itr-250-untied.png
3.3. Sub2adhikarla_em_gaussian-k-3-itr-250-tied.png
3.4. Sub2adhikarla_em_gaussian-k-3-itr-250-untied.png
3.5. Sub2adhikarla_em_gaussian-k-5-itr-250-tied.png
3.6. Sub2adhikarla_em_gaussian-k-5-itr-250-untied.png
--------------------------------------------------------------


--------- Discuss the interpretation of results ---------------
Way of coding and a few extras:
I have coded the entire EM algorithm for gaussian mixture models in a very concise and vectorized way which doesn't require the loops through each data point in the data. The only loop which was required for 1 iteration of EM has to go through a user-defined hyperparameter which is the number of cluster. The codes also contain some matplotlib plotting related code chunks which are encapsulated inside try and except statement. If the optional libraries are not available/installed, then the code will automatically ignore the plotting related code chunks and jump to the necessary components.

Usage of Development Dataset: 
Looking at the results of the number of iterations vs. Log-Likelihood plot with both training and development data. We can see clearly that the results vary according to the hyperparamters (mainly number of cluster as long as the number of iterations are large enough), there are some random effects related to the intialization of cluster centroids/means. The trajectory of cluster centres and also their shapes(Covariance matrices) depend a lot on the initialization. Also, tied covariances tend to perform somewhat poorly when compared to their untied counter-parts. This is comprehensible as the shape of all the k-gaussians need to be the same if tied covariance were chosen which will restrict the learning of differences in the shapes of the underlying gaussians. Plots are attached as PNG images along with this readme files.

Interpretation of experiments:
1. 2-clusters; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each clusters
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 5 iterations{This varies from 5-20 on multiple runs of algorithm} Please look at the text below the x-axis label to get the maximum-likelihood of dev dataset and the iteration at which it was achieved.
2. 2-clusters; 250-iterations; tied-covariances; Random-initialization of centres; Initial-Identity Covariance matrix for all clusters
	Interestingly, the maximum log-likelihood of dev-dataset in this case is consistently higher than the first case where we enforced untied covariances.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 5 iterations{This varies from 5-10 on multiple runs of algorithm}
3. 3-clusters; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each clusters
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 20 iterations{This varies from 20-250 on multiple runs of algorithm, actually in never stops increasing. But the increments become extremely small to quantify}. The contours also look right on target for the most runs and seem to have the best likelihood without increasing the complexity of clusters either (more clusters = more complex)
4. 3-clusters; 250-iterations; tied-covariances; Random-initialization of centres; Initial-Identity Covariance matrix for all clusters
	The maximum log-likelihood of dev-dataset in this case is lower than (3) where we enforced untied covariances.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 15 iterations{This varies from 15-25 on multiple runs of algorithm}
5. 5-clusters; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each clusters
	In my experimentation, this results achieved the highest log-likelihood estimates for the dev-dataset. The Contour plots also look right on target and data seems to have these kind of clustered points in X-space. However, looking at the haphazard nature of (iterations vs LL) plots, I am inclined to believe that the number of clusters may be too many and might be causing some random effects and complicated gaussian model. 
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 62 iterations{This varies from 60-70 on multiple runs of algorithm} 
6. 5-clusters; 250-iterations; Untied-covariances; Random-initialization of centres; Initial-Identity Covariance matrices for each clusters
	The algorithm fails to generate 5 gaussian contours in the X-space. We can only see 4 gaussian contours. Also the right most gassian contour doesn't seem to have enough points in the centre. This might point us to wonder that there may not be a 4th gaussian in the underlying distribution of data.
	The learning reaches a maximum likelihood value for dev-dataset at around ~ 23 iterations{This varies from 20-30 on multiple runs of algorithm}

Best Results on points.dat : 
Iterations : 15-20
clusters : 3
Covariances : Untied

More Dev-logLikelihood  :
Iterations : 50-70
Clusters : 5
Covariances : Untied
(These results can be noisy. But have a chance of achieving the maximum log-likelihood for the dev dataset)
---------------------------------------------------------------


--------------------- Instructions ----------------------------
adhikarla_em_gaussian.py is the main executable script which is a python3 file and requires a few arguments for running:

1. --nodev : If provided, no dev data will be used. If not provided, dev data is used by default.
2. --iterations : An positive integer argument is required along with the keyword, which limits the maximum number of iterations the algorithm goes through before stopping. [Default = 1]
3. --data_file : Requires a string argument which will contain the location of the training data file. By default, the codes will pick up the file from "/u/cs246/data/em/" from CSUG server.
4. --cluster_num : A Positive integer argument is required along with this keyword i.e. User-defined number of clusters.
5. --clusters_file : Requires a string argument which will contain the location of the information pertaining to the initialization of clusters. Please note that only one of the arguments out of (4) or (5) needs to be provided for execution and NOT both.
7. --print_params : Doesn't require any extra argument. enabling this prints the final lambdas, mus and sigmas for each cluster on the console screen. 
8. --tied : If provided, all the clusters are assumed to have same covariance matrices and consequently same shape. 

Example commands : 
1. python3 adhikarla_em_gaussian.py --cluster_num 2 

2. python3 adhikarla_em_gaussian.py --clusters_file '/home/hoover/u8/sadhika4/ml_hw7/em_smoke_test_files/gaussian_smoketest_clusters.txt'

3. python adhikarla_em_gaussian.py --cluster_num 5 --iterations 250 --print_params --tied

4. python adhikarla_em_gaussian.py --cluster_num 3 --iterations 250 --print_params
----------------------------------------------------------------
