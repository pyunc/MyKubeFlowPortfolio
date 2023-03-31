#!/usr/bin/env python3
import kfp
from kfp import dsl
import logging
import kfp.compiler as compiler

def preprocess_op() -> dsl.ContainerOp:
    """
    Returns a Kubeflow ContainerOp for preprocessing the data.

    Returns:
        dsl.ContainerOp: A Kubeflow ContainerOp object.
    """
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='pauloyuncha/iris-kubeflow-ml-preprocessing:latest',
        arguments=[],
        file_outputs={
            'X': '/app/X',
            'y': '/app/y',
        }
    )

def regression_op(X: dsl.InputArgumentPath, y: dsl.InputArgumentPath) -> dsl.ContainerOp:
    """
    Returns a Kubeflow ContainerOp for performing regression.

    Args:
        X (dsl.InputArgumentPath): Path to input X data.
        y (dsl.InputArgumentPath): Path to input y data.

    Returns:
        dsl.ContainerOp: A Kubeflow ContainerOp object.
    """
    return dsl.ContainerOp(
        name='regression',
        image='pauloyuncha/iris-kubeflow-ml-feature-lr:latest',
        arguments=[
            '--X', X,
            '--y', y,
        ],
        file_outputs={
            'model_regression': '/app/model_lr.joblib',
            'accuracy_json': '/app/accuracy.json'
        }
    )

def evaluation_model_op(accuracy_json: dsl.InputArgumentPath) -> dsl.ContainerOp:
    """
    Returns a Kubeflow ContainerOp for evaluating the model.

    Args:
        accuracy_json (dsl.InputArgumentPath): Path to the accuracy json file.

    Returns:
        dsl.ContainerOp: A Kubeflow ContainerOp object.
    """
    return dsl.ContainerOp(
        name='evaluation',
        image='pauloyuncha/iris-kubeflow-ml-evaluation:latest',
        arguments=[
            '--accuracy_json', accuracy_json,
        ],
        file_outputs={
            'result_df': '/app/result_df',
        }   
    )

@dsl.pipeline(
   name='iris-ml',
   description='Kubeflow pipeline local using iris '
)
def iris_pipeline():
    """
    Defines a Kubeflow pipeline for performing regression on the iris dataset.
    """
    _preprocess_op = preprocess_op().add_pod_label("iris-ml", "true")
    
    _regression_op = regression_op(
        X=dsl.InputArgumentPath(_preprocess_op.outputs['X']),
        y=dsl.InputArgumentPath(_preprocess_op.outputs['y']),
    ).after(_preprocess_op)

    evaluation_model_op(
        accuracy_json=dsl.InputArgumentPath(_regression_op.outputs['accuracy_json']),
    ).after(_regression_op)

# client = kfp.Client()
# client.create_run_from_pipeline_func(boston_pipeline, arguments={})

if __name__ == '__main__':
    compiler.Compiler().compile(iris_pipeline, __file__[:-3] + '.yaml')