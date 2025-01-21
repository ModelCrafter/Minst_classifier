
# MNIST Classifier Implementation

This project demonstrates the implementation of various machine learning classifiers to work with the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits. The code covers binary and multi-class classification tasks and includes error analysis and hyperparameter tuning for improved performance.

## Table of Contents

1. **Libraries Used**
2. **Dataset Loading and Visualization**
3. **Binary Classification**
4. **Multi-Class Classification**
5. **Error Analysis**
6. **Hyperparameter Tuning**

---

### 1. Libraries Used

The following libraries are used in the project:

- `numpy` for numerical computations.
- `matplotlib.pyplot` for visualization.
- `sklearn` for machine learning models, evaluation metrics, and preprocessing.

### 2. Dataset Loading and Visualization

- The MNIST dataset is loaded using `fetch_openml`.
- Functions are defined to visualize individual digits (`plot_digit`) and multiple digits (`plot_digits`).
- The dataset is split into training and testing sets:
  - `x_train` and `y_train` for training.
  - `x_test` and `y_test` for testing.

### 3. Binary Classification

- The project begins with a binary classifier to identify the digit `5`.
- The `SGDClassifier` is used for the task:
  - `cross_val_score` and `cross_val_predict` are utilized to evaluate the classifier.
  - Metrics like precision, recall, and F1-score are calculated.
  - Precision-Recall curves are plotted to analyze the classifier's performance at various thresholds.
- A function is provided to adjust the threshold for better precision or recall as required.

### 4. Multi-Class Classification

- Various multi-class classification approaches are demonstrated:
  - `SVC` (Support Vector Classifier) for direct multi-class classification.
  - `OneVsOneClassifier` and `OneVsRestClassifier` for alternative approaches.
- Predictions and confidence scores are computed for the classifiers.

### 5. Error Analysis

- The `SGDClassifier` is scaled using `StandardScaler` to improve performance.
- Error analysis is performed:
  - Predictions are compared to ground truth.
  - Confusion matrices are visualized using `ConfusionMatrixDisplay`.

### 6. Hyperparameter Tuning

- The `KNeighborsClassifier` is optimized using `GridSearchCV`.
- A grid search is conducted to find the best parameters (`n_neighbors` and `weights`) for achieving higher accuracy.
- The results include the best score and parameters.

---

### How to Use

1. Install the necessary libraries.
2. Load the MNIST dataset and visualize it.
3. Run individual sections of the code to train and evaluate classifiers.
4. Use the hyperparameter tuning section to optimize the classifier for higher accuracy.

### Expected Results

- Binary classification accuracy using `SGDClassifier` with a precision-recall tradeoff.
- Multi-class classification results using `SVC` and alternative approaches.
- Error analysis via confusion matrices.
- Accuracy improvement for `KNeighborsClassifier` using grid search.

## License
This project is licensed under the Apache License 2.0.

---
### Developer

***Youssef Khaled***



This project provides a comprehensive overview of building, analyzing, and optimizing classifiers for the MNIST dataset.


