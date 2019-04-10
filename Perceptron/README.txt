Name - Shrikant Adhikarla
email - sadhika4@ur.rochester.edu
URID - 31407229
Coursework - CSC446-Machine Learning
Assignment - 2

--------------- Objective ------------------------------------
		1. Implement perceptron for the adult income dataset using Python. 
		2. Experiment with performance as a function of number of iterations.
		3. Discuss the interpretation of the results.
		4. Create a README file for the same.
---------------------------------------------------------------


--------------- Files (included in submission) ---------------
1. adhikarla_perceptron.py : This is the perceptron implementation python3 file.
2. README.txt : Currently, open file.
3. adhikarla_perceptron.png : Shows the plot of Accuracy vs. #Iterations. Implementation is commented out in the python script.
--------------------------------------------------------------


--------------- Algorithm implemented ------------------------
Perceptron algorithm implemented on a generic basis and adult income dataset is used as a case to predict 
whether a person's income exceeds $50K/yr based on census data.
repeat
	for n = 1 to N
		if t_n * y_n <= 0 then
			w = w + learning_rate * y_n * x_n
		end if
	end for
until for all n t_n = y_n or maxiters
---------------------------------------------------------------


--------- Discuss the interpretation of results ---------------
Usage of Development Dataset: 
Looking at the results of the accuracy vs #Iterations plot with both training and development data. 
Accuracy doesn't seem to improve with the number of Iterations at learning rate = 1. 
Results remain almost similar at lower values of learning rates as well. This most likely mean that the data points are non-linearly separable. Hence, there is swiggly behavior in the plot provided as the perceptron will never converge in this case and keep updating the weight vector indefinitely.
However, we're still getting around ~80% accuracy(+- 4% variability) in the development sets, which is a good classifier accuracy for starters.
---------------------------------------------------------------


--------------------- Instructions ----------------------------
adhikarla_perceptron.py is the main executable script which is a standalone file and requires a few arguments for running:

1. --nodev : If provided, no dev data will be used. If not provided, dev data is used by default.
2. --iterations : An positive integer argument is required along with the keyword, which limits the maximum number of iterations the 							algorithm goes through before stopping. [Default = 50]
3. --lr : Learning rate or eta. Requires a positive real number as argument with the given keyword. Defines the size of the step taken in 						the direction of gradient. [Default = 1.0]
4. --train_file : Requires a string argument which will contain the location of the training data file. By default, the codes will pick up 						the file from /u/cs246/data/adult/ from CSUG server.
5. --dev_file : Requires a string argument which will contain the location of the dev data file. By default, the codes will pick up 						the file from /u/cs246/data/adult/ from CSUG server.
6. --test_file : Requires a string argument which will contain the location of the test data file. By default, the codes will pick up 						the file from /u/cs246/data/adult/ from CSUG server.

Example commands : 
1. python3 adhikarla_perceptron.py --nodev --iterations 100 --lr 1.0 --train_file /u/cs246/data/adult/a7a.train --test_file /u/cs246/data/adult/a7a.test

2. python3 adhikarla_perceptron.py --iterations 10 --lr 0.5 --train_file /u/cs246/data/adult/a7a.train --dev_file /u/cs246/data/adult/a7a.dev --test_file /u/cs246/data/adult/a7a.test

3. python3 adhikarla_perceptron.py
This last command will take all the default arguments.
----------------------------------------------------------------
