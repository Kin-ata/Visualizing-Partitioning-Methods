import numpy as np
import pandas as pd
import templates
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, fcluster

class HierarchicalClustering:
    """A class for computing hierarchical clustering from scratch.
    
    Attributes
    ----------
    X : numpy array
        Input values for hierarchical clustering
    method : str
        Linkage method ('single', 'complete', 'average', etc.)
    distance_threshold : float
        Distance at which clusters will be merged
    max_clusters : int
        Max number of clusters for animation
    clusters_over_steps : list
        Clusters formed at each step for animation
    _labels : list
        List of all labels calculated by each step
    labels : numpy array
        Final cluster labels after fitting
    """

    def __init__(self, X, method='ward', distance_threshold=None, max_clusters=10):
        self.X = X if isinstance(X, np.ndarray) else X.values
        self.method = method
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.clusters_over_steps = []

    def fit(self):
        """Perform hierarchical clustering and store clusters at each number of clusters. From 2 to `max_clusters`."""
        self._labels = []
        Z = linkage(self.X, method=self.method)
        
        for i in range(self.max_clusters+1):
            labels = fcluster(Z, i, criterion='maxclust')
            self._labels.append(labels)
            self.clusters_over_steps.append(labels)
        
        self.labels = self._labels[-1]
        return self

def plot(model, velocidade=2):
    """Plot Hierarchical Clustering's steps.
    
    Parameters
    ----------
    model : HierarchicalClustering
        Hierarchical clustering model after `.fit()`
    velocidade : int
        Transition speed between frames in seconds
    """
    # Convert velocity from seconds to milliseconds
    velocidade *= 1000 

    # Set default colors
    tab10_colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {i: color for i,color in enumerate(tab10_colors)}

    frames = []
    
    points_init = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color="gray"), name='points')
    centroids_init = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')

    points = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color="gray", size=10, opacity=0.8), name='points')
    # centroids = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')

    centroids_init = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Number of clusters:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": velocidade/2, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    def make_slider_step(i, label):
        return {
            "args": [
                [f"step_{i+1}"],
                {"frame": {"duration": velocidade/2, "redraw": True}, "mode": "immediate", "transition": {"duration": velocidade}}
            ],
            "label": label,
            "method": "animate"
        }

    for i in range(1, model.max_clusters+1):
        labels = model._labels[i]
        # colors = [color_dict[l] for l in labels]
        # points = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color=colors, size=10, opacity=0.8), name='points')
        
        colors = pd.Series(model._labels[i]).map(color_dict).values

        # Add centroids

        centroids = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')
        for k in range(0, model.max_clusters + 1):
            if k <= i + 2:
                points.update(dict(marker=dict(color=colors)))
                centroid = np.mean(model.X[labels == k], axis=0)
                centroids['x'] += (centroid[0],)
                centroids['y'] += (centroid[1],)
            # else:
            
                # centroids['x'] += (None,)
                # centroids['y'] += (None,)

        frame = go.Frame(data=[points, centroids], name=f"step_{i+1}")
        frames.append(frame)
        
        # Add slider step
        sliders_dict["steps"].append(make_slider_step(i, f"{i}"))

    # Initial data
    labels_init = model._labels[0]
    # colors_init = [color_dict[l] for l in labels_init]
    # points_init = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color=colors_init, size=10, opacity=0.8), name='points')


    for k in range(1, model.max_clusters + 1):
        if k <= 1:
            centroid = np.mean(model.X[labels_init == k], axis=0)
            centroids_init['x'] += (centroid[0],)
            centroids_init['y'] += (centroid[1],)
        else:
            centroids_init['x'] += (None,)
            centroids_init['y'] += (None,)

    # Build figure
    layout = {
        "sliders": [sliders_dict],
        "updatemenus": [{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": velocidade, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    }

    fig = go.Figure(data=[points_init, centroids_init], layout=layout, frames=frames)
    return fig