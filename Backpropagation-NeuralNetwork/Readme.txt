This directory contains python3 implementation of a 2-layer neural network using Backpropagation algorithm 
(Stochastic gradient descent - reading random tuple by tuple and making the updates)

Name - Shrikant Adhikarla
URID - 31407229
Coursework - CSC446-Machine Learning
Assignment - 3

--------------- Objective ------------------------------------
		1. Implement Backpropagation algorithm with 1 hidden layer for the adult income dataset using Python. 
		2. Experiment: see how accuracy changes when you vary the hyperparameters, including number of iterations, learning rate, and hidden layer size. Use developmen
			data to pick the best set of hyperparameters.
		3. Discuss the interpretation of the results.
		4. Create a README file for the same.
---------------------------------------------------------------


--------------- Files (included in submission) ---------------
1. adhikarla_backprop.py : This is the Backpropagation implementation python3 file.
2. README : Currently, open file.
3. README.pdf : Same file as this in PDF format.
4.1. adhikarla_backprop-n_h-10-lr-0.01.png : Shows the plot of Accuracy vs. #Iterations. Implementation is commented out in the python script for respective learning rate.
4.2. adhikarla_backprop-n_h-10-lr-0.1.png
4.3. adhikarla_backprop-n_h-10-lr-1.0.png
4.4. adhikarla_backprop-n_h-5-lr-0.01.png
4.5. adhikarla_backprop-n_h-5-lr-0.1.png
4.6. adhikarla_backprop-n_h-5-lr-1.0.png
--------------------------------------------------------------


--------------- Algorithm implemented ------------------------
Backpropagation algorithm implemented on a generic basis and adult income dataset is used as a case to predict 
whether a person's income exceeds $50K/yr based on census data.
repeat
	for n = 1 to N
		weight_matrix_for_layer1, weight_vector_for_layer2 = initialize_weights(no_of_hidden_neurons)
		output = Forward_Propagation(weight_matrices, input_x)
		error = calculate_error(output, true_y)
		
		delta2 = error_derivative * output_derivative
		gradient_weights_for_layer2 = delta2 * input_for_layer2

		delta1 = delta2 * weight_matrix_for_layer1 * output_of_layer1_derivative
		gradient_weights_for_layer1 = delta1 * input_x

		weight_vector_for_layer2 = weight_vector_for_layer2 + learning_rate * gradient_weights_for_layer2
		weight_matrix_for_layer1 = weight_matrix_for_layer1 + learning_rate * gradient_weights_for_layer1

	end for
until for all n t_n = y_n or maxiters
---------------------------------------------------------------


--------- Discuss the interpretation of results ---------------
Usage of Development Dataset: 
Looking at the results of the accuracy vs #Iterations plot with both training and development data. We can see clearly that the learning rate of 0.1 gives the best results without any haphazard behaviour. Rest of the chosen learning rates are either too high (causes up and down in the accuracy curve) or too low (required larger number of iterations to converge fully to the global minimum of the cost function). We can see that around 20-25 iterations, the network is capable enough to reach the optimum dev accuracy of 85.4% and test accuracy of ~85%, which seems to be the highest it can achieve given this dataset. In all the experiments, the weights were taken as randomly assigned weight values based on the number of hidden nodes specified by the user. If we increase the number of neurons to 10, the convergence to the global minimum happens very quickly in around ~10-15 iterations itself (at lr ~ 0.1). I have also tried larger values of learning rates like 0.5 and 1.0. All of these values of learning rates cause the accuracy to move haphazardly as they are overshooting the minimum and minimum value of the cost function and changing directions again and again. Lower learning rates like 0.01 were taking too long to converge. Plots are attached as PNG images along with this readme files.

Best Results on adult dataset : 
Iterations : 15-20
learning rate : 0.1
hidden_dim : 5

Even faster convergence :
Iterations : 10-15
learning rate : 0.2
hidden_dim : 15
---------------------------------------------------------------


--------------------- Instructions ----------------------------
adhikarla_backprop.py is the main executable script which is a standalone file and requires a few arguments for running:

1. --nodev : If provided, no dev data will be used. If not provided, dev data is used by default.
2. --iterations : An positive integer argument is required along with the keyword, which limits the maximum number of iterations the algorithm goes through before stopping. [Default = 5]
3. --lr : Learning rate or eta. Requires a positive real number as argument with the given keyword. Defines the size of the step taken in the direction of gradient. [Default = 0.1]
4. --train_file : Requires a string argument which will contain the location of the training data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG server.
5. --dev_file : Requires a string argument which will contain the location of the dev data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG server.
6. --test_file : Requires a string argument which will contain the location of the test data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG server.
7. --print_weights : Doesn't require any extra argument. enabling this prints the weights in the console screen. 
8. --weights_files : Requires 2 string arguments to pass txt files which contain list of weight matrix and weight vector corresponding to the hidden layer and output layer. The contents of these files will be chosen as the initialization weights of the respective neurons. This argument cannot be used with --hidden_dim argument. 
9. --hidden_dim : Requires an integer. This argument creates a respective number of hidden neurons and creates weight matrix and weight vector by random initialization. This argument cannot be used with --weights_files

Example commands : 
1. python3 adhikarla_backprop.py --nodev --iterations 1 --weights_files w1.txt w2.txt --lr 0.01 --train_file /u/cs246/data/adult/a7a.train --test_file /u/cs246/data/adult/a7a.test --print_weights 

2. python3 adhikarla_backprop.py --iterations 10 --lr 0.01 --train_file /u/cs246/data/adult/a7a.train --dev_file /u/cs246/data/adult/a7a.dev --test_file /u/cs246/data/adult/a7a.test --print_weights --hidden_dim 5

3. python3 adhikarla_backprop.py --hidden_dim 5
This last command will take all the default arguments.
----------------------------------------------------------------
