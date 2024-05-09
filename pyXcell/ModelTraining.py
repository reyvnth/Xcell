import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import xgboost
from harmony import harmonize
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_model(
    X,
    meta,
    covariates,
    y_variable,
    method="rf",
    harmonize=True,
    batch_keys=["sample"],
    outpath="/.results/",
):
    """Function to train the classification model

    Parameters:
        X (dataframe): the latent dimensions
        y_variable (str): the name of the column to be used as the target
        meta (dataframe): matrix of covariates + target to be used in the model
        covariates (list): list of column names for covariates to include in the model
        method (str): 'rf' for random forest, or 'xgb' for xgboost
        harmonize (bool): perform harmony?
        batch_keys (list): list of column names to correct on if harmonize=True

    Returns:
        classification model object: classification model that will be used in SHAP analysis
    """

    X = pd.DataFrame(X)
    X.columns = [f"dim{col+1}" for col in X.columns]
    # correct for intersample differences if specified
    if harmonize:
        X = harmony_correction(X, meta, batch_keys)

    # encode the categorical features as numerics
    categoricalColumnNames = meta.select_dtypes(
        include=["object"]
    ).columns.values.tolist()
    # print("the categorical columns", categoricalColumnNames)
    for column_name in categoricalColumnNames:
        label_encoder = LabelEncoder()
        encoded_column = label_encoder.fit_transform(meta[column_name])
        meta[column_name] = encoded_column

    # get the target column
    y = meta[y_variable]
    # print("shape of y: ", y.shape)
    # print("shape of X (before adding covariates): ", X.shape)
    # print("\tnumber of columns in X: ", len(X.columns))
    # print("\tnumber of columns in meta: ", len(meta.columns))
    # join the rest of the meta data table with the latent dimensions
    # X = pd.concat([pd.DataFrame(X), meta.drop(y_variable, axis=1)])
    # X = meta[['cell', 'sample', 'nUMI']].join(X)
    # X = X.drop('cell', axis=1)
    meta.set_index(X.index, inplace=True)
    for covariate in covariates:
        X[covariate] = meta[covariate]
    X = X.rename(str, axis="columns")
    # X = pd.concat([X, meta.drop(y_variable, axis=1)])
    # print("shape of X (after adding covariates): ", X.shape)
    # print("\t", X.columns)
    # print("length of y: ", len(y))

    # training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44
    )
    if method == "rf":
        model = rf(X_train, X_test, y_train, y_test, outpath)
    elif method == "xbg":
        model = xgb(X, y, X_train, X_test, y_train, y_test, outpath)
    else:
        print(
            f"Invalid method `{method}` provided. Please pass a valid method: 'rf' \
              for Random Forest classifier; 'xgb' for XGBoost."
        )

    return (model, X)


# TODO: hyperparameter tuning and train/test split cross validation (stratified/balanced?)
def rf(Xtrain, Xtest, ytrain, ytest, outpath):
    """Train the Random Forest classifier.

    Parameters:
        Xtrain (dataframe): the training data
        Xtest (series): the target labels for the training data
        ytrain (dataframe): the testing data
        ytest (series): the target labels for the testing data

    Returns:
        random forest model object: the random forest model
    """

    # train RF model
    print("Training RF model")
    rf = RandomForestClassifier(
        max_depth=100, max_leaf_nodes=500, random_state=44
    )  # max_depth=50, max_leaf_nodes200
    rf.fit(Xtrain, ytrain)
    y_pred = rf.predict(Xtest)

    # performance metrics
    accuracy = accuracy_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    print("Performance: ")
    print(
        f"\tAccuracy: {accuracy:.4f}\n\tPrecision: {precision:.4f}\n\tF1 Score: {f1:.4f}\n\tRecall: {recall:.4f}"
    )
    y_prob = rf.predict_proba(Xtest)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ytest, y_prob)

    # Calculate AUC score
    auc_score = auc(fpr, tpr)

    # Calculate accuracy
    y_pred = rf.predict(Xtest)
    accuracy = accuracy_score(ytest, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="blue",
        lw=2,
        label=f"NMF ranks (AUC = {auc_score:.2f}, Accuracy = {accuracy:.2f})",
    )
    # plt.plot(fpr_cov, tpr_cov, color='red', lw=2, label='NMF ranks + sex + age (AUC = {:.2f}, Accuracy = {:.2f})'.format(auc_score_cov, accuracy_cov))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{outpath}modelROC.png")
    return rf


# TODO: hyperparameter tuning and train/test split cross validation (stratified/balanced?)
# TODO: evaluation metrics
def xgb(X, y, Xtrain, Xtest, ytrain, ytest, outpath):
    """Train the XGBoost classifier

    Parameters:
        X (dataframe): the latent dimensions
        y (series): target variables
        Xtrain (dataframe): the training data
        Xtest (series): the target labels for the training data
        ytrain (dataframe): the testing data
        ytest (series): the target labels for the testing data

    Returns:
        xgboost model object: the xgboost model
    """
    print("Training XGBoost model")
    # train XGBoost model
    xgb_full = xgboost.DMatrix(X, label=y)
    xgb_train = xgboost.DMatrix(Xtrain, label=ytrain)
    xgb_test = xgboost.DMatrix(Xtest, label=ytest)
    params = {
        "eta": 0.002,
        "max_depth": 3,
        "objective": "survival:cox",
        "subsample": 0.5,
    }
    model_train = xgboost.train(
        params, xgb_train, 10000, evals=[(xgb_test, "test")], verbose_eval=1000
    )

    # train final model on the full dataset
    params = {
        "eta": 0.002,
        "max_depth": 3,
        "objective": "survival:cox",
        "subsample": 0.5,
    }
    model = xgboost.train(
        params, xgb_full, 5000, evals=[(xgb_full, "test")], verbose_eval=1000
    )

    return model


def harmony_correction(X, meta, batch_keys):
    """Correct for intersample differences with Harmony

    Parameters:
        X (dataframe): the latent dimensions
        meta (dataframe): matrix of covariates + target to be used in the model
        batch_keys (list): list of column names to correct on if harmonize=True

    Returns:
        dataframe: the harmonized latent dimensions
    """

    print(f"Calling Harmony to correct the following effects: {batch_keys}")
    x_harmonized = harmonize(X=np.array(X), batch_mat=meta, batch_key=batch_keys)
    x_harmonized_df = pd.DataFrame(x_harmonized)
    x_harmonized_df.columns = X.columns
    return x_harmonized_df


def _generateDatasets(X, k_values):
    data = []
    description = []
    for k in k_values:
        description.append(f"NMF k={k}")
        nmfModel = NMF(n_components=k, init="random", random_state=11)
        W = nmfModel.fit_transform(X)
        H = nmfModel.components_
        data.append(H)
    return (data, description)


def compareModels(X, k_values, outcome, outpath="./results/"):
    """Plot ROC curve to compare different models

    Args:
        datasets (list): list of datasets to use to train the model
        details (list): list of short description for each dataset
        outcome (array?): labels
    """
    datasets, details = _generateDatasets(X, k_values)
    label_encoder = LabelEncoder()
    encoded_outcome = label_encoder.fit_transform(outcome)
    y = encoded_outcome
    # print("shape of y: ", y.shape)
    # Create a list to store the models and their corresponding datasets
    models = []

    # Split each dataset, train a model, and calculate ROC curve
    for idx, df in enumerate(datasets):
        X = df.T
        # print("shape of X: ", X.shape)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=44)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Make predictions on the test set
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Calculate AUC score
        auc_score = auc(fpr, tpr)
        accuracy = accuracy_score(y_test, y_pred)

        # Plot ROC curve
        plt.plot(
            fpr,
            tpr,
            label=f"{details[idx]} (AUC = {auc_score:.2f}; Accuracy = {accuracy:.2f})",
        )

        # Save the trained model for later use if needed
        models.append(model)

    # Set plot labels and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Model Comparison")
    plt.legend()
    plt.savefig(f"{outpath}modelComparison.png")
