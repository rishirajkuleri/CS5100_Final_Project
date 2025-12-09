# student_project/student_project.py
"""
Student implementation for CS5100 Phase 1.
Complete implementation of data loading, preprocessing, Gradient Boosting pipeline,
and from-scratch Random Forest.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import Counter

# Section A: Data Loading

def load_data(path=None):
    """
    Load the student dataset.

    Behavior expected by autograder / tests:
    - If path is None:
        prefer "student-mat-mini.csv" in repo root (fast), else
        prefer "datasets/student-mat-mini.csv", else
        fall back to "datasets/student-mat.csv" or "student-mat.csv".
    - Return: pandas.DataFrame
    """    
    if path is None:
        paths_to_try = [
            "student-mat-mini.csv",
            "datasets/student-mat-mini.csv",
            "datasets/student-mat.csv",
            "student-mat.csv"
        ]        
        for try_path in paths_to_try:
            if os.path.exists(try_path):
                path = try_path
                break
        
        if path is None:
            raise FileNotFoundError("Could not find student dataset in any expected location")

    if 'mini' in path or path.endswith('.csv'):
        try:
            df = pd.read_csv(path, sep=',')
            if 'G3' in df.columns:
                return df
        except:
            pass
    try:
        df = pd.read_csv(path, sep=';')
        if 'G3' in df.columns:
            return df
    except:
        pass
    return pd.read_csv(path)

# Section B: EDA

def summary_stats():
    """
    Return a dictionary of summary statistics, e.g.:
        {"mean_G3": ..., "median_absences": ...}
    """
    df = load_data()
    return {
        "mean_G3": df["G3"].mean(),
        "median_absences": df["absences"].median()
    }


def compute_correlations():
    """
    Compute and return a pandas DataFrame of correlations (df.corr()) for numeric columns.
    """
    df = load_data()
    return df.corr(numeric_only=True)


def preprocess_data(df):
    """
    Preprocess the provided DataFrame and return a processed DataFrame ready for modeling.

    Expected contract:
    - Create target column 'at_risk' as: (df['G3'] < 10).astype(int)
    - Drop grade columns (G1, G2, G3) from the feature matrix to avoid leakage
    - Encode categorical variables (one-hot or similar) so NO object dtypes remain
    - Impute missing values
    - Scale numeric columns to [0,1] range
    - Return a pandas DataFrame that includes 'at_risk' and only numeric columns otherwise
    """
    df = df.copy()

    df['at_risk'] = (df['G3'] < 10).astype(int)

    df = df.drop(columns=['G1', 'G2', 'G3'])
    
    # Separate target from features
    target = df['at_risk']
    features = df.drop(columns=['at_risk'])
    
    # Handle missing values
    for col in features.columns:
        if features[col].isnull().any():
            if features[col].dtype == 'object':
                features[col].fillna(features[col].mode()[0], inplace=True)
            else:
                features[col].fillna(features[col].median(), inplace=True)
    
    # Identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    # OneHot encode categorical variables
    if categorical_cols:
        features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    
    # Scale numerical columns to [0, 1]
    for col in numerical_cols:
        min_val = features[col].min()
        max_val = features[col].max()
        if max_val - min_val > 0:
            features[col] = (features[col] - min_val) / (max_val - min_val)
        else:
            features[col] = 0.0
    
    # Combine features and target
    result = features.copy()
    result['at_risk'] = target
    
    return result

# Section B: Gradient Boosting Pipeline

def train_gb_pipeline(X_train=None, y_train=None):
    """
    Build and fit a sklearn Pipeline that includes:
      ("preprocessor", ColumnTransformer(...)) and ("classifier", GradientBoostingClassifier)

    - Must return a fitted sklearn-like pipeline with .predict() and preferably .predict_proba()
    - Tests expect a named step "preprocessor" to exist (if you return a Pipeline)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier

    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = list(range(X_train.shape[1]))
    preprocessor = ColumnTransformer(
        transformers=[
            ('passthrough', 'passthrough', feature_names)
        ],
        remainder='passthrough'
    )
    
    # Create pipeline with preprocessor and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    
    return pipeline

# Section C: Random Forest (From Scratch)

class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Simple DecisionTree implementation using recursive splitting.
        """
        self.max_depth = max_depth
        self.tree = None

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _split_data(self, X, y, feature_idx, threshold):
        """Split data based on a feature and threshold"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            # Get unique values for this feature
            unique_values = np.unique(X[:, feature_idx])

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                (X_left, y_left), (X_right, y_right) = self._split_data(X, y, feature_idx, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n_left, n_right = len(y_left), len(y_right)
                gini = (n_left / n_samples) * self._gini_impurity(y_left) + \
                       (n_right / n_samples) * self._gini_impurity(y_right)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if depth == self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}

        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}

        (X_left, y_left), (X_right, y_right) = self._split_data(X, y, best_feature, best_threshold)

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Build the decision tree from training data"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict a single sample by traversing the tree"""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict labels for samples in X"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, sample_size=None, random_state=42):
        """
        RandomForest implementation using bagging (bootstrap aggregating).
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - sample_size: Size of bootstrap samples (if None, use full dataset size)
        - random_state: Random seed for reproducibility
        """
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """
        Train the random forest using bootstrapped samples.
        Each tree is trained on a random sample (with replacement) of the data.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        n_samples = X.shape[0]

        if self.sample_size is None:
            sample_size = n_samples
        else:
            sample_size = min(self.sample_size, n_samples)

        np.random.seed(self.random_state)

        self.trees = []
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        """
        Predict labels by majority voting across all trees.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        predictions = []
        for i in range(X.shape[0]):
            sample_predictions = tree_predictions[:, i]
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
