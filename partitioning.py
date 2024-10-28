from sklearn.cluster import SpectralClustering

class PartitioningClustering:
    def __init__(self, data, n_clusters=2):
        self.data = data
        self.n_clusters = n_clusters
        self.model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

    def fit(self):
        self.labels = self.model.fit_predict(self.data)

    def plot(self):
        fig = go.Figure(data=go.Scatter(x=self.data[:, 0], y=self.data[:, 1],
                                        mode='markers', marker=dict(color=self.labels)))
        fig.update_layout(title=f"Partitioning Clustering (n_clusters: {self.n_clusters})",
                          xaxis_title="X", yaxis_title="Y")
        return fig
