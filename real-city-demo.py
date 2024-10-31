import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


df = pd.read_csv(r'D:\Visualizing-Partitioning-Methods\clcs_corr.csv')

# Sidebar options for user inputs
st.sidebar.header("Clustering Options")
features = st.sidebar.multiselect("Select features for clustering", options=df.columns[1:])
method = st.sidebar.selectbox("Select clustering method", ("KMeans"))

# Check if features are selected for clustering
if len(features) < 2:
    st.warning("Please select at least two features for clustering.")
else:
    # Data preprocessing
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method for determining optimal clusters
    wcss = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    
    # Plot the Elbow Method
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 10), wcss, marker='o', color='gold')
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # User input for number of clusters based on elbow plot
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    
    # Apply clustering with chosen method and plot 3D clusters if features are ≥ 3
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
    # elif method == "KMedoids":
    #     model = KMedoids(n_clusters=n_clusters, random_state=0)

    df['Cluster'] = model.fit_predict(X_scaled)

    # 3D Plotting function
    def plot_3d_clusters(df, features):
        palette = sns.color_palette("husl", n_colors=n_clusters)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for cluster, color in zip(df['Cluster'].unique(), palette):
            subset = df[df['Cluster'] == cluster]
            ax.scatter(subset[features[0]], subset[features[1]], subset[features[2]],
                       label=f'Cluster {cluster}', color=color, s=100, alpha=0.8)

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.legend()
        plt.show()
        st.pyplot(fig)

    # Display 3D plot if enough features are selected
    if len(features) >= 3:
        st.write(f"3D Plot of clusters using {features[:3]}")
        plot_3d_clusters(df, features[:3])
    else:
        st.write("Please select at least 3 features for 3D visualization.")
