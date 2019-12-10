# Using Manifold Learning to Defend Against Adversarial Examples

### Kate Avery, MG Hirsch

### CS 5713 Computer Security | Fall 2019

## Abstract

In this project, we used manifold learning to detect and defend against adversarial examples. We used three different manifold learning methods: locally linear embeddings, isomaps, and t-SNE. We used the Fast Gradient Sign Method to generate our adversarial examples. Distances were measured with L<sub>1</sub>, L<sub>2</sub>, and L<sub>1/2</sub> norms. If an adversarial example was off the manifold, we measured the distance between the example and the center of the manifold to detect the example. If the example was on the manifold, we calculated the distance between the example and each label cluster and relabelled the point as the closest cluster. Using this on-manifold method, we found that using the L<sub>2</sub> norm and the isomap manifold performed the best compared to all other manifold learning and norm combinations. Isomap with L<sub>2</sub> corrected incorrect predictions about 15.5% of the time and changed correct predictions about 74.3% of the time. We concluded that these results are most likely because of the nature of on-manifold examples, which move into other class clusters on the manifold when perturbed. 

## How to use this code

1) To set up the environment for running the code, follow the instructions laid out in https://realpython.com/pipenv-guide/. In short, you will need to `pip install pipenv` and then run `pipenv shell` to give you a clean pipenv environment. Run `pipenv install` to install all the dependencies in the environment. 
2) Train the MNIST model by running mnist_model.py. This should output a model file stored in HDF5. 
3) Generate adversarial examples using the methods in fgsm.py. fsgm.generate_adversarial will generate a single adverarial example based on the input example. fgsm.generate_all will generate adversarial examples for a list of input images.
4) Run manifold learning techniques using mnist_generate_manifold.py. Change global parameters to control which method and how many examples to use.
5) Transform adversarial examples into the manifold space by running mnist_transformer.py.
6) Run similarity.py to get the adversarial examples' similarity to the manifold. 

