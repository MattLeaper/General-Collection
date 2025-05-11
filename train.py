import joblib
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


MODEL_FILE = 'model.joblib'

if os.path.exists(MODEL_FILE):
    # Load the saved model
    print(f"model already exists: {MODEL_FILE}")
    best_model = joblib.load(MODEL_FILE)
    print("Loaded  model successfully.")
else:

    # Load training data
    data_train = joblib.load('train.joblib')
    data_train['images'] = data_train['data'].reshape((2200, 2, 62, 47))

    # Load evaluation data
    data_eval = joblib.load('eval1.joblib')
    data_eval['images'] = data_eval['data'].reshape((1000, 2, 62, 47))

    # Flatten images
    X_train = data_train['images'].reshape(data_train['images'].shape[0], -1)
    y_train = data_train['target']

    X_eval = data_eval['images'].reshape(data_eval['images'].shape[0], -1)
    y_eval = data_eval['target']


    # Pipeline with PCA and KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalize data
        ('pca', PCA(n_components=50)),  # Dimensionality reduction
        ('knn', KNeighborsClassifier())  # KNN classifier
    ])

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'cosine', 'minkowski'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Perform grid search
    print("Performing grid search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    # Extract the best model
    best_model = grid_search.best_estimator_
    print("Grid search complete. Best model identified.")

    # Save the best model
    joblib.dump(best_model, MODEL_FILE)
    print(f"Best model saved to '{MODEL_FILE}'")
