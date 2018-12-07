from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams


def get_params(tsne):
    return tsne.get_params(deep=True)


def plot_tsne(embedding, dimensions, num_features):
    rcParams['figure.figsize'] = 6, 10
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.xlabel('x reduction')
    plt.ylabel('y reduction')
    plot_title = "t-SNE Embedding of " + \
        str(num_features) + " Features in " + str(dimensions) + \
        " Dimensions"
    plt.title(plot_title)
    plt.show()


def run_tsne(data, dimensions=2, perplexity=5, early_exaggeration=12.0, learning_rate=60, n_iter=3000, n_iter_without_progress=400, random_state=10):
    """
    Runs t-SNE on data, reduce to # of dimensions passed in.
    :param data: DF of complete data, < 50 features
    :param dimensions: Number of dimensions to reduce to
    :param perplexity: Number of nearest neighbors, between 5 and 50
    :param early_exaggeration: How tight clusters
    """
    num_features = len(list(data))
    tsne = TSNE(n_components=dimensions, perplexity=perplexity,
                early_exaggeration=early_exaggeration)
    embedding = tsne.fit_transform(data)
    plot_tsne(embedding, dimensions, num_features)
    return embedding
