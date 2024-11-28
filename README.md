# Breast Cancer Detection

This project aims to detect breast cancer using machine learning techniques. The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set.

## Dataset

The dataset contains 569 samples of malignant and benign tumor cells. Each sample has 30 features, which are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe the characteristics of the cell nuclei present in the image.

## Features

The dataset includes the following features:

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

## Data Preprocessing

1. **Loading the Data**: The dataset is loaded using pandas.
2. **Dropping Irrelevant Columns**: The 'ID' column is dropped as it is not relevant for the analysis.
3. **Renaming Columns**: Columns with spaces in their names are renamed to use underscores instead.
4. **Checking for Missing Values**: The dataset is checked for any missing values.
5. **Visualization**: Various visualizations such as histograms, scatter plots, and correlation heatmaps are created to understand the distribution and relationships between features.

## Feature Selection

Several methods are used to determine the most significant features:

1. **Ordinary Least Squares (OLS)**: Used to find the p-values of features.
2. **Decision Tree Classifier**: Used to find feature importance.
3. **Backward Elimination**: Used for feature selection based on p-values.
4. **Lasso Regression**: Used for feature selection with L1 regularization.

## Model Training

The following models are trained and evaluated:

1. **Logistic Regression with K-Fold Cross-Validation**: The dataset is split into 5 folds, and logistic regression is trained and evaluated on each fold.
2. **Logistic Regression with Random Splits**: The dataset is split into 80/20 train-test splits, and logistic regression is trained and evaluated on each split.
3. **AdaBoost with Logistic Regression**: An AdaBoost model with logistic regression as the base estimator is trained and evaluated.

## Results

The performance of the models is evaluated using accuracy, precision, recall, F1 score, and ROC-AUC score. The results are printed and visualized using learning curves.

## Conclusion

This project demonstrates the use of various machine learning techniques for breast cancer detection. The feature selection methods help in identifying the most significant features, and the models provide good accuracy in detecting breast cancer.

## Requirements

- Python 3.12.3
- pandas
- numpy
- seaborn
- matplotlib
- statsmodels
- scikit-learn
- mglearn

## Usage

To run the project, execute the Jupyter notebook `breast_cancer_detection.ipynb`.
