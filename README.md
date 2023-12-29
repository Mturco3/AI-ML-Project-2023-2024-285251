# AI-ML-Project-2023-2024

# Project 1: TRAINS

### Group members:
- Michele Turco ()
- Simone Filosofi (284531)
- Simone Angelo Meli (289631)

## INTRODUCTION

As part of your duties as senior data scientist for the famous ThomasTrain company, we were assigned to understand the satisfaction of the customers even without a direct evaluation. To accomplish this task,we were provided  with the “trains_dataset.csv”.

First of all we used the Explanatory Data Analysis (EDA) to study carefully the the csv file and each categorical and numerical variable we had. We preprocessed the data handling missing values (substituting these with the mode), encoding the categories and removing the outilers. 

The final goal was to choose and improve a machine learning model able to classify the customers as "satisfied" or "unsatisfied" with high accuracy, taking into consideration the understanding behind the satisfaction of the clients.

## METHODS

Which methods have we used to realise our project?
In this section we are going to explain how we approached each part of our work.

### Explanatory Data Analysis (EDA):

- Reading the input file: we used the Pandas library to open the given csv file;
- 1.1)General overview of the dataset: with *'df_trains.head()'* we extract the first rows of the dtataset just to visualize it;
- 1.2) Showing the dimension of the dataset: We saw that the dataset has 129880 rows and 25 columns;
- 1.3)Gathering informations from the data: *'df_trains.info()'* allowed us to output the columns' names and data types, and *'df_trains.nunique()'* outputs the number of unique values in each column;
- 1.4)Handling missing values: we foud out that 'Arrival Delay in minutes' column has 393 missing values and we substitute these with the mode (even if another option would have been to simply delete the 393 missing values);
- 1.5)Relevant features of numeric variables: we used the *'.describe()'* function that offers a comprehensive overview of key statistics for each numerical variable;
- 1.6)Insights on categorical variables: we went through every categorical variable and found how many times each unique class appears, also plotting some pie charts and histograms to show;
- 1.7)Insights on numerical variables: focusing on numerical variables, it was possible to compute correlation using the *'.corr()'* function (we also show the distibution of each variable through subgraphs);
- 1.8) Data Reduction: assuming they don’t have any predictive power to predict the dependent variable, we removed 'Ticket ID' and 'Date and Time' features;
- 1.9) Inspecting for Outliers: outliers are relevant to build our model since they can negatively affect the performance, so as a first step, we  plotted boxplots in order to have a general idea on what is going on.

### Preprocessing:

- 2.1) Handling missing values: we quoted this point but since we have already covered point 1 in the previous scetion (EDA), we didn't do anything more about it;
- 2.2) Inspecting for redundancy: since there are many features that are not really correlated with the target variable, we drop them, since they are not relevant to train the model. In particular, we set the threshold at 0.15;
<img width="829" alt="m1" src="![Alt text](image.png)">
- 2.3) Column cleanup: for simpicity we converted all columns' names into lowercase (yes, we could have done it even earlier 😅);
- 2.4) StandardScaler and OneHotEncoder:  are preprocessing techniques used in machine learning to prepare data for modeling. They serve different purposes: StandardScaler is used for standardizing numerical features by scaling them to have mean 0 and variance 1, OneHotEncoder is used for converting categorical variables into a numerical format. Then, we look for categorical variables with a low correlation with the target variable "Satisfied", using the same threshold as before (0.15), and we drop them;
- 2.5) Removing outliers: we used two-steps approach, defining a lower_bound (determined by subtracting 1.5 times the interquartile range from the first quartile) and an upper_bound (adding 1.5 times the IQR to the third quartile), because this range helps identify values that fall significantly below or above the bulk of the data, indicating potential outliers.

### Generating training set and test set

- 3.1) Split the dataset: we generated a training set containing 75% of the observations and a test set containing the remaining 25%.
- 3.2) Distribution of the target variable in the different sets: The ratio between the two classes is the same in both the training and the test set. This is a good thing, since it means that the model will be trained on a balanced dataset and will be able to generalize well. In addittion, we do not have to deal with stratification (splitting the dataset mantaining a balanced ratio between the two classes);
- 3.3) Fit and Transform: we apply fit_transform on the training set and transform on the test set in order to standardize the data. 

### Model Training and Hyperparameter Tuning:

- Model selection: for our classification task we chose Logistic Regression, Decision Tree, and Random Forest;
- Testing Different Models: to have an overview before the tuning, we compute the training and the validation accuracy for all the three models;
- 







## EXPERIMENTAL DESIGN


## RESULTS


## CONCLUSIONS

