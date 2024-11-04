import kmeans, kmedoids
import templates

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from sklearn.cluster import DBSCAN
import streamlit as st	
import streamlit.components.v1 as components

import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, fcluster  # Import for hierarchical clustering
import plotly.figure_factory as ff  # Import for creating dendrogram

st.set_page_config(page_title="K-means Visualization App", page_icon="python.png", layout="wide")


# st.markdown("""
# **Author:** Cainã Max Couto da Silva  
# **LinkedIn:** [cmcouto-silva](https://www.linkedin.com/in/cmcouto-silva/)
# """)

st.header('**Visualizing Partitioning Methods**')
st.write("")

# components.html('<b>texto</b>')
# st.markdown("<h1 style='text-align: center; color: red;'>texto</h1>", unsafe_allow_html=True)

# st.sidebar.title('Parameters')
st.sidebar.title('Parameters')

with st.sidebar.container():
   _, slider_col, _ = st.columns([0.02, 0.96, 0.02])
   with slider_col:
        k = st.sidebar.select_slider(
				label='Number of simulated clusters (groups):',
				options=range(2,11), value=2
			)
        std = st.sidebar.slider(
				'Standard Deviation of simulated data:',
				0.1, 5.0, 1.0, 0.1
			)

# mode = st.sidebar.selectbox('Method for initialization of the centroids:', ["random", "kmeans++"])

_, central_button, _ = st.sidebar.columns([0.25, 0.5, 0.25])
with central_button:
	st.text("")
	st.button('Recompute')


tab1, tab2, tab3, tab4 = st.tabs(["K-means", "K-means++", "K-medoids", "K-means trap"])

with tab1:
	st.markdown("""
		## **K-means**
	""")	

	data = make_blobs(centers=k, cluster_std=std)
	df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

	model, wss = kmeans.calculate_WSS(data[0], k, 10, mode="random")
	raw_col, elbow_col = st.columns([0.5,0.5])

	_, kanimation_col, _ = st.columns([0.2,0.8,0.2])

	with kanimation_col:
		fig = kmeans.plot(model)
		fig = fig.update_layout(autosize=False, height=560,
			title_text="<b>Visualizing K-means - animated steps</b>", title_font=dict(size=24))
		st.plotly_chart(fig, use_container_width=True, key="kanimation_col_kmean")

	with raw_col:
		raw_fig_kmean = go.Figure(
			data=fig.data[0],
			layout=dict(
				template='seaborn', title='<b>Unlabeled data</b>',
				xaxis=dict({'title':'x'}), yaxis=dict({'title':'y'})
				)
			)
		st.plotly_chart(raw_fig_kmean, use_container_width=True, key="raw_col_kmean")

	with elbow_col:
		elbow_fig = go.Figure(
		data=go.Scatter(x=list(range(1,11)), y=wss),
		layout=dict(
			template='seaborn', title='<b>Elbow method</b>',
			xaxis=dict({'title':'k'}), yaxis=dict({'title':'wss'})
			)
		)
		st.plotly_chart(elbow_fig, use_container_width=True, key="elbow_col_kmean")

with tab2:
	st.markdown("""
		## **K-means++**
	""")	

	# data = make_blobs(centers=k, cluster_std=std)
	df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

	model, wss = kmeans.calculate_WSS(data[0], k, 10, mode="kmeans++")
	raw_col, elbow_col = st.columns([0.5,0.5])

	_, kanimation_col, _ = st.columns([0.2,0.8,0.2])

	with kanimation_col:
		fig = kmeans.plot(model)
		fig = fig.update_layout(autosize=False, height=560,
			title_text="<b>Visualizing K-means - animated steps</b>", title_font=dict(size=24))
		st.plotly_chart(fig, use_container_width=True, key="kanimation_col_kmean++")

	with raw_col:
		raw_fig_kmean = go.Figure(
			data=fig.data[0],
			layout=dict(
				template='seaborn', title='<b>Unlabeled data</b>',
				xaxis=dict({'title':'x'}), yaxis=dict({'title':'y'})
				)
			)
		st.plotly_chart(raw_fig_kmean, use_container_width=True, key="raw_col_kmean++")

	with elbow_col:
		elbow_fig = go.Figure(
		data=go.Scatter(x=list(range(1,11)), y=wss),
		layout=dict(
			template='seaborn', title='<b>Elbow method</b>',
			xaxis=dict({'title':'k'}), yaxis=dict({'title':'wss'})
			)
		)
		st.plotly_chart(elbow_fig, use_container_width=True, key="elbow_col_kmean++")

with tab3:
    st.markdown("""
        ## **K-medoids**
    """)

    # data, labels = make_blobs(centers=k, cluster_std=std)
    df = pd.DataFrame(data[0], columns=['x', 'y']).assign(label=data[1])

    model = kmedoids.KMedoids(data[0], k)
    model.fit(verbose=True)

    raw_col, kanimation_col = st.columns([0.5, 0.5])

    with kanimation_col:
        fig = model.plot(data[0])
        fig = fig.update_layout(autosize=False, height=560,
                                title_text="<b>Visualizing K-medoids - animated steps</b>", title_font=dict(size=24))
        st.plotly_chart(fig, use_container_width=True, key="kanimation_col_kmedoids")

    with raw_col:
        raw_fig_kmedoids = go.Figure(
            data=go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode="markers",
							marker=dict(color='gray')),
            layout=dict(
                template='seaborn', title='<b>Unlabeled data</b>',
                xaxis=dict({'title': 'x'}), yaxis=dict({'title': 'y'})
            )
        )
        st.plotly_chart(raw_fig_kmedoids, use_container_width=True, key="raw_col_kmedoids")

with tab4:
	st.markdown("""
		## **K-means trap**
	""")

	# Specific biased data
	raw_seed, kanimation_seed = st.columns([0.5,0.5])

	data_seed,labels_seed = make_blobs(centers=4, random_state=3)
	model_seed = kmeans.Kmeans(data_seed, 4, seed=2)
	model_seed.fit()

	with raw_seed:
		raw_fig_seed = go.Figure(
			data=go.Scatter(x=data_seed[:,0], y=data_seed[:,1], mode='markers', marker=dict(color=labels_seed)),
			layout=dict(title_text="<b>Labeled points according to the actual clusters</b>",
				template="simple_white", title_font=dict(size=18)))
		raw_fig_seed.update_layout(templates.simple_white, height=500, title_x=0.15, title_font_size=18)
		st.plotly_chart(raw_fig_seed, use_container_width=True, key="raw_seed")

	with kanimation_seed:
		fig_seed = kmeans.plot(model_seed)
		fig_seed = fig_seed.update_layout(autosize=False, height=500,
			title_text="<b>Visualizing bias in centroid initialization</b>", title_font=dict(size=21))
		st.plotly_chart(fig_seed, use_container_width=True, key="kanimation_seed")

# with tab2:
# 	st.markdown("""
# 		## **Hierarchical Clustering**
# 	""")	
# 	# data = make_blobs(centers=k, cluster_std=std)
# 	df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

# 	model = hierarchical.HierarchicalClustering(data[0])
# 	model.fit()
# 	raw_col, dendrogram_col = st.columns([0.5, 0.5])

# 	_, kanimation_col, _ = st.columns([0.2, 0.8, 0.2])

# 	with kanimation_col:
# 		fig = hierarchical.plot(model)
# 		fig = fig.update_layout(
# 			autosize=False, height=560,
# 			title_text="<b>Visualizing Hierarchical Clustering - animated steps</b>", title_font=dict(size=24)
# 		)
# 		st.plotly_chart(fig, use_container_width=True, key="kanimation_col_hierarchical")

# 	with raw_col:
# 		raw_fig_hierarchical = go.Figure(
# 			data=fig.data[0],
# 			layout=dict(
# 				template='seaborn', title='<b>Unlabeled data</b>',
# 				xaxis=dict({'title': 'x'}), yaxis=dict({'title': 'y'})
# 			)
# 		)
# 		st.plotly_chart(raw_fig_hierarchical, use_container_width=True, key="raw_col_hierarchical")

# 	with dendrogram_col:
# 		dendrogram_fig = go.Figure(
# 			data=ff.create_dendrogram(data[0]),
# 			layout=dict(
# 				template='seaborn', title='<b>Dendrogram</b>',
# 				xaxis=dict({'title': 'x'}),
# 				yaxis=dict({'title': 'Height'})
# 			)
# 		)
# 		st.plotly_chart(dendrogram_fig, use_container_width=True, key="dendrogram_col_hierarchical")

# with tab3:
# 	st.markdown("""
# 		## **DBSCAN Clustering**
# 	""")	

# 	# data = make_blobs(centers=k, cluster_std=std)
# 	df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

# 	model = dbscan.DBSCAN(data[0], max_eps=10, max_min_pts=10)  # Set max values as needed
# 	model.fit()

#     # Plot raw and clustered data with interactive control over eps and MinPts
# 	raw_col, dendrogram_col = st.columns([0.5,0.5])

# 	_, kanimation_col, _ = st.columns([0.2,0.8,0.2])

# 	with kanimation_col:
# 			fig = dbscan.plot(model, max_eps=10, max_min_pts=10)
# 			fig.update_layout(autosize=False, height=560,
# 							title_text="<b>Visualizing DBSCAN - Animated Steps</b>", title_font=dict(size=24))
# 			st.plotly_chart(fig, use_container_width=True, key="kanimation_col_dbscan")

# 	with raw_col:
# 		raw_fig_dbscan = go.Figure(
#             data=go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode="markers", 
#                             marker=dict(color='gray')),
#             layout=dict(
#                 template='seaborn', title='<b>Unlabeled Data</b>',
#                 xaxis=dict(title='x'), yaxis=dict(title='y')
#             )
#         )
# 		st.plotly_chart(raw_fig_dbscan, use_container_width=True, key="raw_col_dbscan")
    
   
# with tab4:
# 	st.markdown("""
# 		## **Partitioning Clustering**
# 	""")