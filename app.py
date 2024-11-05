import kmeans
import templates

import numpy as np
import pandas as pd

from sklearn import cluster, datasets
import streamlit as st	
import streamlit.components.v1 as components

import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, fcluster  # Import for hierarchical clustering
import plotly.figure_factory as ff  # Import for creating dendrogram

st.set_page_config(page_title="K-means Visualization App", page_icon="python.png", layout="wide")


st.header('**Visualizing Partitioning Methods**')
st.write("")

st.sidebar.title('Parameters')

type_ = st.sidebar.selectbox("Data type", ["blobs1", "blobs2", "moons", "circles"])

if type_ == "blobs1":
	with st.sidebar.container():
		k = st.sidebar.select_slider(
			label='Number of simulated clusters (groups):',
			options=range(2,11), value=2
		)
		std = st.sidebar.slider(
			'Standard Deviation:',
			0.1, 5.0, 1.0, 0.1
		)
	data = datasets.make_blobs(centers=k, cluster_std=std)
elif type_ == "blobs2":
	with st.sidebar.container():
		k = st.sidebar.select_slider(
			label='Number of simulated clusters (groups):',
			options=range(2,11), value=2
		)
		std = []
		for i in range(k):
			std.append(st.sidebar.slider(
				f'Standard Deviation for cluster {i+1}:',
				0.1, 5.0, 1.0, 0.1
			))
	data = datasets.make_blobs(centers=k, cluster_std=std)
elif type_ == "moons":
	with st.sidebar.container():
		k = st.sidebar.select_slider(
			label='Number of simulated clusters (groups):',
			options=range(2,11), value=2
		)
		std = st.sidebar.slider(
			'Noise level:',
			0.1, 5.0, 0.1, 0.1
		)
	data = datasets.make_moons(n_samples=100, noise=std)
elif type_ == "circles":
	with st.sidebar.container():
		k = st.sidebar.select_slider(
			label='Number of simulated clusters (groups):',
			options=range(2,11), value=2
		)
		std = st.sidebar.slider(
			'Noise level:',
			0.1, 5.0, 0.1, 0.1
		)
		factor = st.sidebar.slider(
			'Factor between inner and outer circle:',
			0.1, 5.0, 0.1, 0.1
		)
	data = datasets.make_circles(n_samples=100, noise=0.1, factor=factor)

_, central_button, _ = st.sidebar.columns([0.25, 0.5, 0.25])
with central_button:
	st.text("")
	st.button('Recompute')


tab1, tab2, tab3 = st.tabs(["K-means", "K-means++", "K-means trap"])

with tab1:
	st.markdown("""
		## **K-means**
	""")	
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

	# data = datasets.(centers=k, cluster_std=std)
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
		## **K-means trap**
	""")

	# Specific biased data
	raw_seed, kanimation_seed = st.columns([0.5,0.5])

	data_seed,labels_seed = datasets.make_blobs(centers=4, random_state=3)
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