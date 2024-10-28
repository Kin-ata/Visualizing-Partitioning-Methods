import numpy as np
import pandas as pd
import templates
import plotly.graph_objects as go


class DBSCAN:
    """A class for computing DBSCAN from scratch

    Attributes
    ----------
    X : numpy array
        The input values for computing DBSCAN
    eps : float
        Maximum distance between two points to be considered neighbors
    min_pts : int
        Minimum number of points to form a dense region
    max_eps : int
        Maximum value of epsilon for interactive control
    max_min_pts : int
        Maximum value of MinPts for interactive control
    num_individuals : int
        Number of observations in the dataset
    _labels : 3D list
        Stores labels for each point for each eps-MinPts configuration

    Methods
    -------
    region_query(point_idx, eps)
        Find all neighbors within `eps` distance of a given point
    expand_cluster(labels, point_idx, cluster_idx, eps, min_pts)
        Expand cluster based on neighbors if it meets `min_pts`
    fit()
        Cluster data for each eps and MinPts combination and store in `_labels`
    """

    def __init__(self, X, max_eps, max_min_pts):
        self.X = X if isinstance(X, np.ndarray) else X.values
        self.num_individuals = len(self.X)
        self.max_eps = max_eps
        self.max_min_pts = max_min_pts
        self._labels = [[[None for _ in range(self.num_individuals)] 
                        for _ in range(self.max_min_pts)] 
                        for _ in range(self.max_eps)]

    def region_query(self, point_idx, eps):
        """Find points within `eps` distance from `point_idx`"""
        neighbors = []
        for idx in range(self.num_individuals):
            if np.linalg.norm(self.X[point_idx] - self.X[idx]) < eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(self, labels, point_idx, cluster_idx, eps, min_pts):
        """Expand cluster with density-reachable points"""
        neighbors = self.region_query(point_idx, eps)
        if len(neighbors) < min_pts:
            labels[point_idx] = -1  # Noise
            return False
        else:
            labels[point_idx] = cluster_idx
            queue = list(neighbors)
            while queue:
                neighbor_idx = queue.pop(0)
                if labels[neighbor_idx] is None:
                    labels[neighbor_idx] = cluster_idx
                    new_neighbors = self.region_query(neighbor_idx, eps)
                    if len(new_neighbors) >= min_pts:
                        queue.extend(new_neighbors)
                elif labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_idx
            return True

    def fit(self):
        """Cluster data over eps and min_pts ranges and store in `_labels`"""
        for eps in range(1, self.max_eps + 1):
            for min_pts in range(1, self.max_min_pts + 1):
                labels = [None] * self.num_individuals
                cluster_idx = 0
                for point_idx in range(self.num_individuals):
                    if labels[point_idx] is None:
                        if self.expand_cluster(labels, point_idx, cluster_idx, eps, min_pts):
                            cluster_idx += 1
                self._labels[eps - 1][min_pts - 1] = labels
        return self


def plot(model, max_eps, max_min_pts, velocity=2):
    """Plot animated DBSCAN clustering steps with eps and min_pts sliders

    Parameters
    ----------
    model : DBSCAN class
        An instance of the DBSCAN class after applying the `.fit()` method.
    max_eps : int
        Maximum value for epsilon (eps) slider
    max_min_pts : int
        Maximum value for MinPts slider
    velocity : int
        Transition speed in seconds between frames
    """

    # Convert velocity from seconds to milliseconds
    velocity *= 1000

    # Default color palette
    tab10_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {i: color for i, color in enumerate(tab10_colors)}
    
    layout = templates.plotly_dark
    layout.update(
        showlegend=False,
        updatemenus=[
            {
                "buttons": [
                    {"args": [None, {"frame": {"duration": velocity, "redraw": False},
                                     "fromcurrent": True}],
                     "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False}, 
                                       "mode": "immediate"}],
                     "label": "Pause", "method": "animate"}
                ],
                "type": "buttons", "showactive": True
            }
        ]
    )

    # Initialize frames and slider for interactive control
# Initialize frames and sliders for interactive control
    eps_slider = {
        "steps": [],
        "currentvalue": {"prefix": "eps: ", "font": {"size": 20}},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": -0.1  # Position the eps slider below the plot
    }

    min_pts_slider = {
        "steps": [],
        "currentvalue": {"prefix": "MinPts: ", "font": {"size": 20}},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": -0.3  # Position the MinPts slider below the eps slider
    }

    

    frames = []
    for eps in range(max_eps):
        for min_pts in range(max_min_pts):
            labels = model._labels[eps][min_pts]
            cluster_colors = pd.Series(labels).map(color_dict).fillna("gray").values
            scatter = go.Scatter(x=model.X[:, 0], y=model.X[:, 1],
                                mode="markers", marker=dict(color=cluster_colors))
            frame = go.Frame(data=[scatter], name=f"{eps + 1},{min_pts + 1}")
            frames.append(frame)
            if min_pts == 0:
                eps_slider["steps"].append({"args": [[f"{eps + 1},{min_pts + 1}"]],
                                            "label": f"{eps + 1}",
                                            "method": "animate"})
            if eps == 0:
                min_pts_slider["steps"].append({"args": [[f"{eps + 1},{min_pts + 1}"]],
                                                "label": f"{min_pts + 1}",
                                                "method": "animate"})

    layout["sliders"] = [eps_slider, min_pts_slider]
    fig = go.Figure(data=[frames[0].data[0]], layout=layout, frames=frames)

    return fig
