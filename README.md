# ML-CSE-512-Final-Project-Comparing-and-Contrasting-Machine-Learning-Tools
# Table of contents

- [00. Abstract](#abstract-main)
- [01. Introduction](#intro-overview)
- [02. Dataset, Classification Tools, and Performance Metrics](#metrics-overview)
    - [02.1 Dataset](#dataset-overview)
    - [02.2 Image Dataset](#image-overview)
    - [02.3 Numerical Dataset](#numerical-overview)
    - [02.4 Classification Tools](#classification-overview)
    - [02.5 Performance Metrics](#performance-overview)
- [03. Experiments And Results](#experiments-overview)
    - [03.1 Image Data](#images_-overview)
    - [03.2 Numerical Data](#num-overview)
    - [03.3 Experiment Processes](#experiment-overview)
    - [03.4 Experiment Results](#results-overview)
    - [03.5 Feature Efficiency and Misclassification Analysis of Numerical Dataset](#efficiency-overview)
- [04. Conclusions and Future Work](#conclusion-summary)
- [05. Contributions](#contribute-summary)
- [06. Referneces](#references-summary)

# Abstract <a name="abstract-main"></a>
Is there a universally applicable technique in machine learning? Does the diversity of data modalities have an impact? This empirical study aims to explore multiple machine learning methodologies, specifically those related to classification, within the context of gender classification. Two distinct datasets, comprising both image and numerical data, are employed to evaluate and compare the performance of these methodologies. The study focuses on analyzing the effectiveness of features and examining cases of misclassification.


# Introduction <a name="intro-overview"></a>
Machine learning involves training a machine learning tool with specific data to enable it to perform desired tasks when presented with new data. It encompasses various tasks such as classification, regression, detection, and ranking. Numerous machine learning tools have emerged over the years, many of which demonstrated impressive performance prior to the advent of deep learning.

However, can a single method be relied upon to address all machine learning problems, or does the choice of tool depend on the specific task and data modalities involved? Several comprehensive survey papers have shed light on this topic. Reference [1] provides a comprehensive overview of machine learning algorithms that can enhance the intelligence and capabilities of applications. It elucidates the principles behind different machine learning techniques and their applicability in various real-world domains. The paper also highlights the challenges and potential research directions in the field. Reference [2] discusses several machine learning tools and their application to different tasks. It provides a succinct summary of these tools, highlighting their key features and their applicability to real-world problem-solving. Additionally, it explores various parameters and notable features, as well as specific frameworks compatible with processing platforms.

One seminal study, published sixteen years ago, [3] undertook a large-scale empirical comparison of supervised learning methods. The authors employed a variety of performance criteria to evaluate these methods. The findings revealed several important insights: 1) learning methods may perform well on one metric but poorly on another, 2) calibration significantly impacts tool performance, and 3) no universally superior learning algorithm exists. Even the best models may exhibit poor performance on certain problems, while models with low average performance may excel on specific problems or metrics.

Motivated by the aforementioned papers, our study aims to determine the optimal tools—specifically Support Vector Machines (SVM), Random Forests, and Artificial Neural Networks (ANN)—for classification tasks involving different data modalities. Specifically, we seek to: 1) identify the top-performing classification tool for each dataset, 2) compare the performance of each machine learning tool across both similar and distinct modalities, 3) analyze the efficiency of the employed features, and 4) investigate instances of misclassification.

The remaining sections of this report are organized as follows: Section 2 provides a brief introduction to the datasets, machine learning tools, and performance evaluation metrics. Section 3 presents the experimental setup, results, and analysis. The conclusion is drawn in Section 4, while Section 5 outlines the individual contributions of each team member.

# Dataset, Classification Tools, and Performance Metrics <a name="metrics-overview"></a>

## Dataset <a name="dataset-overview"></a>
The gender classification dataset from Kaggle.com will be used in this experimental study. This dataset has two
independent sub-datasets, one is an image dataset [4], another is a numerical dataset [5].

## Image dataset <a name="image-overview"></a>
The image dataset is of cropped images of about 28,500 male and 28,500 female faces. Some sample images are shown below:
<img width="706" alt="Screenshot 2023-07-07 at 9 05 11 AM" src="https://github.com/EfthimiosVlahos/SBU-CSE-512-ML-Final-Project-Comparing-and-Contrasting-ML-Tools-/assets/56899588/f09e94d0-57d3-4f93-b6ba-f7dba0808459">

## Numerical dataset <a name="numerical-overview"></a>
The numerical dataset contains a label and seven descriptive features for each of the 5,001 subjects. The label is the
gender of a subject, which is either ”Male” or ”Female”. The features are described as follows:

longhair - If the subject has ”long hair”, this feature is 1. Otherwise, this feature is 0.

foreheadwidthcm - This is the width of the subject’s forehead in centimeters. foreheadheightcm -
This is the height of the subject’s forehead in centimeters.

nosewide - If the subject has a ”wide nose”, this feature is 1. Otherwise, this feature is 0. noselong - If
the subject has a ”long nose”, this feature is 1. Otherwise, this feature is 0. 
lipsthin - If the subject has ”thin lips”, this feature is 1. Otherwise, this feature is 0. 

distancenosetoliplong - 1 if the subject has ”long distance between nose and lips” and 0 otherwise.

## Classification Tools <a name="classification-overview"></a>
SVMs, ANNs, and Random Forests, all classic classification tools [6], are used in this experimental study.

## Performance Metrics <a name="performance-overview"></a>
In this study, three performance metrics Accuracy, F1-score [7], and AUC-ROC curve [8] are used to gauge the
comparison of the above classification tools.

# Experiments And Results <a name="experiments-overview"></a>

## Image Data <a name="images_-overview"></a>

The images underwent preprocessing to facilitate their utilization in machine learning algorithms. Firstly, each image was opened and converted to greyscale. Subsequently, the images were resized uniformly to dimensions of 50×50 pixels. To normalize the pixel data, we divided each pixel value by 255, ensuring consistency across the dataset. Following this normalization step, the data for each image was reshaped into a single array, comprising 2,500 elements representing the normalized information for each pixel in the image.


## Numerical Data <a name="num-overview"></a>

Among the seven descriptive features in the Numerical dataset, no preprocessing was applied to the five binary features.
For each of the two real number features, i.e., foreheadwidthcm and foreheadheightcm, the value of each feature V was
normalized by using the maximum value Vmax and the minimum value Vmin of that feature as (V −Vmin)/(Vmax −Vmin) such
that all the normalized values are in the range of [0.0, 1.0].




## Experiment Processes <a name="experiment-overview"></a>


In this study, we conducted a comprehensive exploration of hyperparameters for each of the classification tools employed. To achieve this, we employed grid search, a widely-used technique for hyperparameter optimization.

For Support Vector Machines (SVMs), we systematically tested various combinations of hyperparameters, namely C and gamma, across different kernel functions such as polynomial, radial basis function (RBF), and sigmoid.

In the case of Artificial Neural Networks (ANNs), we experimented with different hyperparameter configurations, including the number of hidden layers, the number of neurons within each layer, and the selection of activation functions.

Similarly, for Random Forests, we explored diverse combinations of hyperparameters, specifically n estimators and max depth.

Once the optimal hyperparameter set was determined for each classification tool, we employed this set to train the respective tool. Subsequently, the trained classification tool was used to evaluate the performance on unseen data, serving as a robust assessment of its predictive capabilities.



## Experiment Results <a name="results-overview"></a>

The Accuracy, F1-score, and AUC/ROC of each of the classification tools (including different kernels) on the test dataset
are given in Table 1. The elapsed time of each case is also given in Table 1.

<img width="950" alt="Screenshot 2023-07-07 at 9 26 09 AM" src="https://github.com/EfthimiosVlahos/SBU-CSE-512-ML-Final-Project-Comparing-and-Contrasting-ML-Tools-/assets/56899588/e4642d61-9faf-43c9-ba65-00704aa9f11b">

To compare the performances of different classification tools on different modalities, the ROC curves of each of the
tools are grouped into two plots, one on the numerical dataset and the other on the image dataset, and shown in Figure
2. To compare the performances of a classification tool on different modalities, the ROC curve on numerical dataset and
that on image dataset of each of the tools are grouped together. Figure 3 shows five plots, each of which is for a specific
tool (or kernel).


<img width="956" alt="Screenshot 2023-07-07 at 9 29 45 AM" src="https://github.com/EfthimiosVlahos/SBU-CSE-512-ML-Final-Project-Comparing-and-Contrasting-ML-Tools-/assets/56899588/c042f84f-5363-4344-a26a-aeec1a8968a3">


For numerical dataset, the performances of all the tools are very comparable. For image dataset, ANN has better
performance than all other classification tools. Apparently, ANN takes much longer time than all the other tools (please
notice, on image dataset, ANN uses GPU while others just use regular CPU). It is also clear that time elapsed on image
dataset is way longer than that on numerical dataset.


## Feature Efficiency and Misclassification Analysis of Numerical Dataset <a name="efficiency-overview"></a>

<img width="920" alt="Screenshot 2023-07-07 at 9 36 19 AM" src="https://github.com/EfthimiosVlahos/SBU-CSE-512-ML-Final-Project-Comparing-and-Contrasting-ML-Tools-/assets/56899588/34cc1501-9e13-401c-a3d8-3bb1a5fef958">

<img width="920" alt="Screenshot 2023-07-07 at 9 37 17 AM" src="https://github.com/EfthimiosVlahos/SBU-CSE-512-ML-Final-Project-Comparing-and-Contrasting-ML-Tools-/assets/56899588/fe813863-0d0f-464e-b16a-dece2fb4cf60">


It has been found that long hair is very insignificant to determine if a subject is female or not and the four most
significant features are wide nose, long nose, thin lips, and long distance nose to lip. When a female is at least three out
of those four significant features, the female is misclassified as a male. If a male either has at most one of those four
significant features, or at most two of those features plus a forehead width between 12.8cm and 12.9cm, the male is
misclassified as a female.


# Conclusion and Future Work <a name="conclusion-summary"></a>

In recent years, machine learning has experienced remarkable advancements, largely driven by the significant progress in computational power, including the emergence of powerful tools such as Graphics Processing Units (GPUs). Artificial Neural Networks (ANNs) have emerged as the cornerstone of contemporary machine learning methodologies. In our experimental study, we observed that ANNs surpassed other tools in terms of classification performance, albeit with a limited dataset. However, the computational time required by ANNs remains a notable concern.

Based on the findings of this study, it was evident that the machine learning tools exhibited superior performance when applied to numerical datasets compared to image datasets, as indicated by both performance metrics and computational time. This disparity can be attributed to the interplay between human expertise and artificial intelligence, whereby human guidance enables the identification of crucial features deserving of attention. Consequently, the combination of human intelligence and artificial intelligence yields optimal results. This observation suggests a promising avenue for future research, promoting the integration of human intelligence into machine learning methodologies in a seamless and organic manner.

Furthermore, there is still potential for enhancement in numerical feature selection. Although four significant numerical features were identified, there are still other relevant numerical features that warrant consideration. In future studies, the inclusion of such features, even if their significance may be relatively less pronounced, could further contribute to the overall improvement of classification performance. One illustrative example of a potentially valuable feature is the measurement of eyebrow characteristics.

These observations emphasize the ongoing pursuit of refining machine learning methodologies by leveraging both human expertise and technological advancements, with the aim of achieving even higher levels of performance and accuracy in diverse applications.

# Contributions <a name="contribute-summary"></a>

**Edward**

Performance metric investigation, image preprocessing and postprocessing, numerical SVM grid
search and model fitting, numerical data misclassification analysis, image data misclassification analysis, and assisted
in the composition of the final report.

**Michael**

Reviewed literature, composed part of poster slides, investigated and provided codes for
AUC/ROC.

**Jainam**

Set-up pre-trained deep learning models (AlexNet and ResNet50) for image dataset,
hyperparameter search to find learning rate and number of epochs, trained the ANNs and got predictions on test data.
Prototyped code for the models (SVMs, ANNs, Random Forests) used for numerical data.

**Efthimios**

Coordinated the collaboration, worked with the team to pick up the topic, to review the
literature, to compose and submit abstract, proposal, report, and poster, preprocessed the numerical dataset, worked
on the Random Forest coding, training, and test, participated coding, training, and test on other classification tools and
tally such as result collection, feature efficiency analysis, and misclassification analysis

# References <a name="references-summary"></a>

1. I. H. Sarker, Machine Learning: Algorithms, Real-World Applications and Research Directions, SN Computer
Science 2.3 (2021): 1-21.
2. S. Sarumathi, M. Vaishnavi, S. Geetha, P. Ranjetha, Comparative Analysis of Machine Learning Tools: A Review,
International Journal of Computer and Information Engineering 15.6 (2021): 354-363.
3. R. Caruana, A. Niculescu-Mizil, An Empirical Comparison of Supervised Learning Algorithms, Proceedings of the
23rd International Conference on Machine Learning, Pittsburgh, PA, 2006.
4. https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset.
5. https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset.
6. https://www.wikipedia.org.
7. J Korstanje, The F1 score. https://towardsdatascience.com/the-f1-score-bec2bbc38aa6.
8. R Draelos, Measuring Performance: AUC (AUROC).
https://glassboxmedicine.com/2019/02/23/measuringperformance-auc-auroc/.
