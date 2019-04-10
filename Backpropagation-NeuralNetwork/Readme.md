This directory contains python3 implementation of a 2-layer neural network using Backpropagation algorithm 
<br>(Stochastic gradient descent - reading random tuple by tuple and making the updates)

<br>Name - Shrikant Adhikarla
<br>URID - 31407229
<br>Coursework - CSC446-Machine Learning
<br>Assignment - 3
<br>
<br>--------------- Objective ------------------------------------
<br>		1. Implement Backpropagation algorithm with 1 hidden layer for the adult income dataset using Python. 
<br>		2. Experiment: see how accuracy changes when you vary the hyperparameters, including number of iterations, learning rate, and hidden layer size. Use developmen
<br>			data to pick the best set of hyperparameters.
<br>		3. Discuss the interpretation of the results.
<br>		4. Create a README file for the same.
<br>---------------------------------------------------------------
<br>
<br>
<br>--------------- Files (included in submission) ---------------
<br>1. adhikarla_backprop.py : This is the Backpropagation implementation python3 file.
<br>2. README : Currently, open file.
<br>3. README.pdf : Same file as this in PDF format.
<br>4.1. adhikarla_backprop-n_h-10-lr-0.01.png : Shows the plot of Accuracy vs. #Iterations. Implementation is commented out in the python script for respective learning rate.
<br>4.2. adhikarla_backprop-n_h-10-lr-0.1.png
<br>4.3. adhikarla_backprop-n_h-10-lr-1.0.png
<br>4.4. adhikarla_backprop-n_h-5-lr-0.01.png
<br>4.5. adhikarla_backprop-n_h-5-lr-0.1.png
<br>4.6. adhikarla_backprop-n_h-5-lr-1.0.png
<br>--------------------------------------------------------------
<br>
<br>
<br>--------------- Algorithm implemented ------------------------
<br>Backpropagation algorithm implemented on a generic basis and adult income dataset is used as a case to predict 
<br>whether a person's income exceeds $50K/yr based on census data.
<br>repeat
<br>	for n = 1 to N
<br>		weight_matrix_for_layer1, weight_vector_for_layer2 = initialize_weights(no_of_hidden_neurons)
<br>		output = Forward_Propagation(weight_matrices, input_x)
<br>		error = calculate_error(output, true_y)
<br>		
<br>		delta2 = error_derivative * output_derivative
<br>		gradient_weights_for_layer2 = delta2 * input_for_layer2
<br>
<br>		delta1 = delta2 * weight_matrix_for_layer1 * output_of_layer1_derivative
<br>		gradient_weights_for_layer1 = delta1 * input_x
<br>
<br>		weight_vector_for_layer2 = weight_vector_for_layer2 + learning_rate * gradient_weights_for_layer2
<br>		weight_matrix_for_layer1 = weight_matrix_for_layer1 + learning_rate * gradient_weights_for_layer1
<br>
<br>	end for
<br>until for all n t_n = y_n or maxiters
<br>---------------------------------------------------------------
<br>
<br>
<br>--------- Discuss the interpretation of results ---------------
<br>Usage of Development Dataset: 
<br>Looking at the results of the accuracy vs #Iterations plot with both training and development data. We can see clearly that the learning rate of 0.1 gives the best results without any <br>haphazard behaviour. Rest of the chosen learning rates are either too high (causes up and down in the accuracy curve) or too low (required larger number of iterations to converge fully<br> to <br>the global minimum of the cost function). We can see that around 20-25 iterations, the network is capable enough to reach the optimum dev accuracy of 85.4% and test accuracy of ~85%, <br>which seems to be the highest it can achieve given this dataset. In all the experiments, the weights were taken as randomly assigned weight values based on the number of hidden nodes <br>specified by the user. If we increase the number of neurons to 10, the convergence to the global minimum happens very quickly in around ~10-15 iterations itself (at lr ~ 0.1). I have <br>also <br>tried larger values of learning rates like 0.5 and 1.0. All of these values of learning rates cause the accuracy to move haphazardly as they are overshooting the minimum and minimum v<br>alue <br>of the cost function and changing directions again and again. Lower learning rates like 0.01 were taking too long to converge. Plots are attached as PNG images along with this readme <br>files.<br>
<br>
<br>Best Results on adult dataset : 
<br>Iterations : 15-20
<br>learning rate : 0.1
<br>hidden_dim : 5
<br>
<br>Even faster convergence :
<br>Iterations : 10-15
<br>learning rate : 0.2
<br>hidden_dim : 15
<br>---------------------------------------------------------------
<br>
<br>
<br>--------------------- Instructions ----------------------------
<br>adhikarla_backprop.py is the main executable script which is a standalone file and requires a few arguments for running:
<br>
<br>1. --nodev : If provided, no dev data will be used. If not provided, dev data is used by default.
<br>2. --iterations : An positive integer argument is required along with the keyword, which limits the maximum number of iterations the algorithm goes through before stopping. [Default = 5]
<br>3. --lr : Learning rate or eta. Requires a positive real number as argument with the given keyword. Defines the size of the step taken in the direction of gradient. [Default = 0.1]
<br>4. --train_file : Requires a string argument which will contain the location of the training data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG <br>server.<br>
<br>5. --de<br>v_file : Requires a string argument which will contain the location of the dev data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG server.<br>
<br>6. --te<br>st_file : Requires a string argument which will contain the location of the test data file. By default, the codes will pick up the file from /u/cs246/data/adult/ from CSUG serve<br>r.
<br>7. --pr<br>int_weights : Doesn't require any extra argument. enabling this prints the weights in the console screen. <br>
<br>8. --we<br>ights_files : Requires 2 string arguments to pass txt files which contain list of weight matrix and weight<br> vector corresponding to the hidden layer and output layer. The content<br>s <br>of thes<br>e files will be chosen as the initialization weights of the respective neurons. This argument cannot be us<br>ed with --hidden_dim argument. <br>
<br>9. --hi<br>dden_dim : Requires an integer. This argument creates a respective number of hidden neurons and creates we<br>ight matrix and weight vector b<br>y random initialization. This argument <br>cannot <br>be used with --weights_files<br>
<br>
<br>Example commands : 
<br>1. python3 adhikarla_backprop.py --nodev --iterations 1 --weights_files w1.txt w2.txt --lr 0.01 --train_file /u/cs246/data/adult/a7a.train --test_file /u/cs246/data/adult/a7a.test <br>--print_weights <br>
<br>
<br>2. python3 adhikarla_backprop.py --iterations 10 --lr 0.01 --train_file /u/cs246/data/adult/a7a.train --dev_file /u/cs246/data/adult/a7a.dev --test_file /u/cs246/data/adult/a7a.test <br>--print_weights --hidden_dim 5<br>
<br>
<br>3. python3 adhikarla_backprop.py --hidden_dim 5
<br>This last command will take all the default arguments.
<br>----------------------------------------------------------------<br>
