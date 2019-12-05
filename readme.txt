Neural Network Playground

Description:
An interactive game-like introduction to neural networks. In the playground, the user first gives a function they want to try to 
approximate, and then gives different hyperparameters and watches the network learn and can modify it as it does. 
When the user is satisfied, they can see how well they performed on the testing data.

How to run the playground:
Run playground.py

How to add your own data:
To add inputs, place 2D array .npy files in the same directory as playground.py
To add outputs, place 1D array .npy files in the same directory as playground.py 
(.npy files can be generated using np.save, sample files "x.npy" and "y.npy" are included)

Required Libraries: numpy (https://pypi.org/project/numpy/)
