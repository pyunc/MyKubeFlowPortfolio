import pandas as pd
import argparse
import json
import pickle

def evaluation(accuracy_json: str) -> None:
    """
    Loads the accuracy score of a trained model from a JSON file and saves the result as a Pandas DataFrame. The DataFrame
    is sorted by the accuracy score and then pickled to a binary file.

    Args:
    accuracy_json (str): The path to the JSON file containing the accuracy score of the trained model.

    Returns:
    None
    """

    # Load the accuracy score from the JSON file
    accuracy_score = json.load(open(accuracy_json))

    # Create a Pandas DataFrame with the model name and accuracy score
    results = pd.DataFrame({
        'Model': ['logisticregression'],
        'Score': [accuracy_score]
    })

    # Sort the results DataFrame by the accuracy score in descending order
    result_df = results.sort_values(by='Score', ascending=False)

    # Dump the sorted DataFrame to a binary file
    with open('result_df', 'wb') as f:
        pickle.dump(result_df, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add an argument for the path to the JSON file containing the accuracy score
    parser.add_argument('--accuracy_json', type=str, help='The path to the JSON file containing the accuracy score.')
    args = parser.parse_args()

    # Call the evaluation function with the path to the accuracy JSON file
    evaluation(accuracy_json=args.accuracy_json)