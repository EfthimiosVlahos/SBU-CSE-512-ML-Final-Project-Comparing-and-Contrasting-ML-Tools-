# ML-CSE-512-Final-Project-Comparing-and-Contrasting-Machine-Learning-Tools
Aimed to find out which tools out of SVM (Support Vector Machines), Random Forests, and ANN (Artificial Neural Networks) work best for classification on different data modalities with respect to gender classification.
# Abstract
Does machine learning have a one size fits all technique? Does data coming in all different modalities make any difference? In this repository, several machine learning tools, more specifically classification tools, are investigated. Two gender classification datasets with different data types, i.e., image and numerical, are used to compare the performance of those tools. The efficiency of the features and the misclassification cases are analyzed.


# DATASET, CLASSIFICATION TOOLS, AND PERFORMANCE METRICS
**Dataset:** The gender classification dataset from Kaggle.com will be used in this experimental study. This dataset has two independent sub-datasets, one is an image dataset, another is a numerical dataset.

**ImageDataset:** The image dataset is of cropped images of about 28,500 male and 28,500 female faces.

**Classification Tools:** SVMs, ANNs, and Random Forests, all classic classification tools, are used in this experimental study.

**Performance Metrics:** In this study, three performance metrics Accuracy, F1-score, and AUC-ROC curve are used to gauge the comparison of the above classification tools.

# EXPERIMENTS AND RESULTS

**Image Data:** The images were preprocessed into a dataframe so they can be inputted into the machine learning algorithms. Each image was opened and converted to greyscale. We then resized the images so that each image was the same size 50×50 pixels. We normalized the pixel data by dividing each pixel value by 255. We then reshaped the data of each image to a single array consisting of 2,500 factors containing the normalized information for each pixel in the image. 

**Numerical Data:** Among the seven descriptive features in the Numerical dataset, no preprocessing was applied to the five binary features. For each of the two real number features, i.e., foreheadwidthcm and foreheadheightcm, the value of each feature V was normalized by using the maximum value Vmax and the minimum value Vmin of that feature as (V −Vmin)/(Vmax −Vmin) such that all the normalized values are in the range of [0.0, 1.0].

**Experiment Processes:**  In this study, we have used grid search to find the hyperparameters for each of those classification tools.
For SVMs, we have tested a few combinations of C and gamma on the polynomial, RBF, and sigmoid kernels.
For ANNs, we have tested a few combinations of number of hidden layers, number of neurons in each layer, and activation functions.
For Random Forests, we have tested a few combinations of n estimators and max depth.
  Once the hyperparameter set has been found for a classification tool, that hyperparameter set is used to train the classification tool and the trained classification tool is used to test the unseen data. The rest of the results and graphs can be found in the CSE-512-Project-Report-CCSV (1).pdf file uploaded.
  
# EXPERIMENTS AND RESULTS

From this experimental study, it has been found that the tools performs much better on the numerical dataset than on the image dataset, in terms of the performance metrics and also elapsed time. This is because of human interaction with the machine, where the human tells the machine which features may be important and should be looked at. Therefore, when we combine human intelligence and artificial intelligence we get the best results. This could become a future research such that more human intelligence is involved in machine learning in an organic way.
Moreover, there is still room for the numerical features to improve. Although there are four significant numerical features, other numerical features are still in the play. Some other relevant features can be added in the future, though they might be less significant.
