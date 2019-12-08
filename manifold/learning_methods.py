from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding


def lle(x):
    """
    https://stackoverflow.com/questions/42275922/setting-the-parameters-of-locally-linear-embedding-lle-method-in-scikit-learn
    """
    embedding = LocallyLinearEmbedding(n_components=2)  # 2D projection
    x_transformed = embedding.fit_transform(x)
    return embedding, x_transformed


def isomap(x):
    embedding = Isomap(n_components=2)
    x_transformed = embedding.fit_transform(x)
    return embedding, x_transformed


def laplacian(x):
    embedding = SpectralEmbedding(n_components=2)
    x_transformed = embedding.fit_transform(x)
    return embedding, x_transformed
