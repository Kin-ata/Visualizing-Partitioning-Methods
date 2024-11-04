import numpy as np
import plotly.graph_objects as go
from copy import deepcopy

def _get_init_centers(K, n_samples):
    """Return random points as initial centers."""
    init_ids = []
    while len(init_ids) < K:
        _ = np.random.randint(0, n_samples)
        if _ not in init_ids:
            init_ids.append(_)
    return init_ids

def _get_distance(data1, data2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((data1 - data2)**2))

def _get_cost(X, centers_id, dist_func):
    """Calculate total cost and cost of each cluster."""
    dist_mat = np.zeros((len(X), len(centers_id)))
    for j, center_id in enumerate(centers_id):
        center = X[center_id, :]
        for i, point in enumerate(X):
            dist_mat[i, j] = 0. if i == center_id else dist_func(point, center)
    mask = np.argmin(dist_mat, axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))
    for i in range(len(centers_id)):
        mem_id = np.where(mask == i)
        members[mem_id] = i
        costs[i] = np.sum(dist_mat[mem_id, i])
    return members, costs, np.sum(costs), dist_mat

def _kmedoids_run(X, K, dist_func, max_iter=1000, tol=0.001, verbose=False):
    """Run the K-Medoids algorithm and return the centers, members, and costs."""
    n_samples = len(X)
    centers = _get_init_centers(K, n_samples)
    members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)
    for _ in range(max_iter):
        swapped = False
        for i in range(n_samples):
            if i not in centers:
                for j in range(len(centers)):
                    new_centers = deepcopy(centers)
                    new_centers[j] = i
                    new_members, new_costs, new_tot_cost, new_dist_mat = _get_cost(X, new_centers, dist_func)
                    if new_tot_cost - tot_cost < tol:
                        members, costs, tot_cost, dist_mat = new_members, new_costs, new_tot_cost, new_dist_mat
                        centers = new_centers
                        swapped = True
                        # if verbose:
                        #     print(f"Updated centers to: {centers}")
        if not swapped:
            break
    return centers, members, costs, tot_cost, dist_mat

class KMedoids:
    """
    K-Medoids clustering class.
    """
    def __init__(self,X, K, max_iter=1000, tol=0.001, dist_func=_get_distance):
        """Initialize the K-Medoids model."""
        try:
            self.X = X.values
        except:
            self.X = X
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.dist_func = dist_func
        self.centers_ = None
        self.labels_ = None
        self.costs_ = None

    def fit(self, verbose=False):
        """Fit the K-Medoids model."""
        self.centers_, self.labels_, self.costs_, self.tot_cost_, self.dist_mat_ = _kmedoids_run(
            self.X, self.K, self.dist_func, self.max_iter, self.tol, verbose
        )
    def plot(self, X):
        """Plot the clusters and centers using Plotly."""
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        fig = go.Figure()
        for i in range(self.K):
            cluster_points = X[self.labels_ == i]
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(color=colors[i % len(colors)], size=6),
                name=f"Cluster {i + 1}"
            ))
            fig.add_trace(go.Scatter(
                x=[X[self.centers_[i], 0]],
                y=[X[self.centers_[i], 1]],
                mode='markers',
                marker=dict(color=colors[i % len(colors)], size=12, symbol="star"),
                name=f"Center {i + 1}"
            ))
        fig.update_layout(
            title="K-Medoids Clustering",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True
        )
        return fig
