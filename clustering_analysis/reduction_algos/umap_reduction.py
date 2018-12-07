import umap
import matplotlib.pyplot as plt


def plot_umap(embedding, dimensions, num_features):
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.xlabel('x reduction')
    plt.ylabel('y reduction')
    plot_title = "UMAP projection of " + \
        str(num_features) + " Features in " + str(dimensions) + \
        " Dimensions"
    plt.title(plot_title)
    plt.show()


def run_umap(data):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.001, metric='correlation')
    embedding = reducer.fit_transform(data)
    plot_umap(embedding, 2, len(list(data)))
    return embedding
