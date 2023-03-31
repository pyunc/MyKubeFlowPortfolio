import sys
from sklearn.datasets import load_iris
from typing import Tuple
import pickle

def load_iris_data()-> Tuple:
    
    """
    Loads iris dataset from scikit-learn's datasets module, 
    saves it as two separate pickle files (X and y) in the current directory, 
    and returns a tuple containing the features and labels arrays.
    """
    
    iris = load_iris()
    X = iris.data
    y = iris.target

    # save X and y as separate pickle files
    with open('X', 'wb') as f:
        pickle.dump(X, f)
    with open('y', 'wb') as f:
        pickle.dump(y, f)

if __name__ == '__main__':
     
    print('Preprocessing data...')
    sys.exit(load_iris_data())