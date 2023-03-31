import argparse
import json
import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def regression(X:str, y:str) -> None:
    
    """
    Trains a Logistic Regression model using the provided data and outputs the accuracy of the model and the trained model 
    to the files 'accuracy.json' and 'model_lr.joblib' respectively.

    Args:
    X (str): path to the input data file containing the features.
    y (str): path to the input data file containing the target.

    Returns:
    None
    """
        
    X_df = pd.read_pickle(X)
    y_df = pd.read_pickle(y)

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, solver='lbfgs', max_iter=10))
    ])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X=X_df, y=y_df, cv=kfold)

    logging.info("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))

    model = 'lr'
    score = scores.mean()
    
    with open('accuracy.json', 'w') as f:
        json.dump({model:score}, f)

    output_file = "model_lr.joblib"
    joblib.dump(pipeline, output_file)  # type: ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--X', help='Path to the input data file containing the features.')
    parser.add_argument('--y', help='Path to the input data file containing the target.')

    args = parser.parse_args()
    regression(
        X=args.X,
        y=args.y,
    )