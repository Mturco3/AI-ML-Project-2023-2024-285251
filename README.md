# AI-ML-Project-2023-2024

# Project 1: TRAINS

### Group members:
- Simone Filosofi (284531)
- Simone Angelo Meli (289631)
- Michele Turco ()

## INTRODUCTION

As part of your duties as senior data scientist for the famous ThomasTrain company, we were assigned to understand the satisfaction of the customers even without a direct evaluation. To accomplish this task,we were provided  with the “trains_dataset.csv”.

First of all we used the Explanatory Data Analysis (EDA) to study carefully the the csv file and each categorical and numerical variable we had. We preprocessed the data handling missing values (substituting these with the mode), encoding the categories and removing the outilers. 

The final goal was to choose and improve a machine learning model able to classify the customers as "satisfied" or "unsatisfied" with high accuracy, taking into consideration the understanding behind the satisfaction of the clients.

## METHODS

Which methods have we used to realise our project?
In this section we are going to explain how we approached each part of our work.

- Reading the input file: we used the Pandas library to open the given csv file;
### 1) Understanding the dataset
- 1.1)General overview of the dataset: with *'df_trains.head()'* we extract the first rows of the dtataset just to visualize it;
- 1.2) Showing the dimension of the dataset: We saw that the dataset has 129880 rows and 25 columns;
- 1.3)Gathering informations from the data: *'df_trains.info()'* allowed us to output the columns' names and data types, and *'df_trains.nunique()'* outputs the number of unique values in each column;
- 1.4)Handling missing values: we foud out that 'Arrival Delay in minutes' column has 393 missing values and we substitute these with the mode (even if another option would have been to simply delete the 393 missing values);
- 1.5) Data Reduction: assuming they don’t have any predictive power to predict the dependent variable, we removed 'Ticket ID' and 'Date and Time' features;
- 1.6) Outliers detection: outliers are relevant to build our model since they can negatively affect the performance, so as a first step, we  plotted boxplots in order to have a general idea on what is going on.
### 2) EDA for feature understanding
- 2.1) Descriptive statistics:
- 2.2) Univariate Analysis:
- 2.3) Correlation Analysis:
### 3) Feature selection:
### 4) Praparing data for modeling
- 4.1) Encoding Categorical Variables:
- 4.2) Removing Outliers
- 4.3) Data Splitting: we generated a training set containing 75% of the observations and a test set containing the remaining 25%.
- 4.4) Distribution of the target variable in the different sets: The ratio between the two classes is the same in both the training and the test set. This is a good thing, since it means that the model will be trained on a balanced dataset and will be able to generalize well. In addittion, we do not have to deal with stratification (splitting the dataset mantaining a balanced ratio between the two classes);
- 4.5) Feature Scaling (Fit and Transform)  we apply fit_transform on the training set and transform on the test set in order to standardize the data.
- 4.6) Creating a Validation Set
### 5) Model Building
- Model selection: for our classification task we chose Logistic Regression, Decision Tree, and Random Forest;
- 5.1) Testing Different Models: to have an overview before the tuning, we compute the training and the validation accuracy for all the three models;
- 5.2) Hyperparameter Tuning Using Cross-Validation: as a first step, we searched the best hyperparameters for the model using the RandomizedSearchCV function (this allowed us to explicitly control the number of parameter combinations that are attempted);
### 6) Plotting Learning Curves
### 7) Models Evaluation
- 7.1) Classification metrics
- 7.2) Confusion Matrices
- 7.3) ROC Curves
- 7.4) Models Comparison




### Model Training and Hyperparameter Tuning:

- Model selection: for our classification task we chose Logistic Regression, Decision Tree, and Random Forest;
- Testing Different Models: to have an overview before the tuning, we compute the training and the validation accuracy for all the three models;
- Hyperparameter Tuning: as a first step, we searched the best hyperparameters for the model using the RandomizedSearchCV function (this allowed us to explicitly control the number of parameter combinations that are attempted);
- Learning curves:
- Models Evaluation:
- Confusion matrices:
- ROC curves:
- Models comparison:




## EXPERIMENTAL DESIGN

#### Introduction to the Experimental Approach: 

Our experimental methodology centered on accurately anticipating customer contentment while pinpointing the most influential attributes. This strategy involved in-depth scrutiny, model curation, and assessment;

#### Choice of Models and Baseline Development: 

- Logistic Regression: picked due to its computational efficiency and explanatory power, acting as a fundamental reference point.
- Decision Trees:sSelected for their capacity to depict non-linear associations and interpretability, negating the necessity for feature scaling.
- Random Forest: an amalgamation of Decision Trees intended to boost efficiency and steadiness, diminishing the likelihood of overfitting while adeptly handling diverse attributes and interactions.

#### Evaluation Metrics:

- Accuracy: Evaluated the models' comprehensive performance.
- Precision: Crucial in reducing false positives in predicting customer contentment.
- Recall: Vital in accurately identifying all dissatisfied instances.
- F1-Score: Offered a balanced measure encompassing precision and recall, particularly critical in an uneven dataset.
- ROC-AUC Score: Assessed the models' capacity to differentiate contented and discontented customers.

All these metrics contributed significantly to a comprehensive model assessment.

## RESULTS


## CONCLUSIONS

