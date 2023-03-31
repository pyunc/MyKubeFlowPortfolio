# kubeflow_example
This repo is made for kubeflow example using Iris dataset for local cluster spin and deployment 

![iris](https://static.vecteezy.com/system/resources/previews/000/145/921/non_2x/vector-iris-flower-banner-line-art.jpg)

# How to use it?
- install minikube
- run minikube start
- install kubeflow in minikube cluster
- install kubeflow pipeline in minikube cluster
- run kubectl port-forward svc/ml-pipeline-ui 3000:80 --namespace kubeflow
- run iris-ml-kfp.py pipeline to generate -> iris-ml-kfp.yaml
- In http://127.0.0.1:3000 -> click Pipelines and upload the generated yaml file
- click run -> start experiment
- you should expect something like the pic bellow

![result](https://i.imgur.com/VQlXhQ7.png)

