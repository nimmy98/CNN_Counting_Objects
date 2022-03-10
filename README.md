Steps to Run CNN:

1. Install Python
2. Install Tensorflow-Datasets (Ensure it matches your version of python) using pip.
3. Install Tensorflow using pip.
4. Download files packaged with this project.
5. Zip the dataset2 folder in our project and place it in the Tensorflow-datasets manual installation folder. Should be in C:\Users\ [USER]\tensorflow_datasets\downloads\manual
	The manual folder may not be there, if so, create a folder of that name.
6. Use the CMD for your operating system inside of the folder containing the project files and run "tfds build". The command should automatically locate the generation script.
7. If you add a dataset, change in the code the references to our dataset (all will be in quotation marks) and replace with your own.
8. If you want to change the training and testing data numbers, go into the cnn code "example2_test.py" and change the numbers
in the for loops using ".batch(NUMBER)" on line 15-22.
9. Run the "example2_test.py" code using whatever python IDE or running method you prefer.


To use the data generation code, also install scikit-image, numpy, and matplotlib with pip.
![image](https://user-images.githubusercontent.com/25381992/157581651-e32c97b2-415e-4f66-b40e-0fa002d2d076.png)
