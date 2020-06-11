#!/usr/bin/env python
# encoding: utf-8

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import plotly.offline as offline
import plotly.graph_objs as go
import plotly
from plotly import tools
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
import matplotlib.pyplot as plt
import chart_studio.plotly as py

#since it is my personal account for plotly, do not use it too often. QAQ
tools.set_credentials_file(username='yy.fy.cn',api_key='s4aYwqqYayPPjcqFxk7p')
#offline.init_notebook_mode()

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def embd_visual(disease_img, title, image_path, dim = 2):
	'''
		visualise embedding with bubble chart.  The area of bubble decreases with its rank.
	:param disease_img: shape: [batch_size, embedding_dim]
	:param vac_titles:
	:param jskr_id:
	:param image_path:
	:param dim:
	:return:
	'''

	tsne = TSNE(n_components=dim, init='pca', random_state=0)
	disease_img = disease_img.detach().numpy()
	result = tsne.fit_transform(disease_img)

	#x_min, x_max = np.min(result, 0), np.max(result, 0)
	#data = (result - x_min) / (x_max - x_min)
	data = result

	tsne3d = TSNE(n_components=3, init='pca', random_state=0)
	result3d = tsne3d.fit_transform(disease_img)

	#x_min, x_max = np.min(result3d, 0), np.max(result3d, 0)
	#data3d = (result3d - x_min) / (x_max - x_min)
	data3d = result3d

	# no using any GUI dependent backend. (the remove server(appbox) doesn't have GUI)
	plt.switch_backend('agg')

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(2, 1, 2)

	#plt.xlim(-0.3, 1.3)
	#plt.ylim(-0.3, 1.3)

	#area = (-1) * np.linspace(-disease_img.shape[0], -1, disease_img.shape[0])
	#area = (10 * area) ** 2
	#plt.scatter(data[:, 0], data[:, 1], s=40, lw=0, c=colors, alpha=0.5, edgecolors="grey", linewidths=2)
	#plt.scatter(data[:, 0], data[:, 1], s=40, lw=0, c=palette[colors.astype(np.int)])
	#colors = np.linspace(1, disease_img.shape[0], disease_img.shape[0])
	#plt.scatter(data[:, 0], data[:, 1], s=20, c=colors, alpha=0.5, edgecolors="grey", linewidths=2)
	ax.scatter(data[:, 0], data[:, 1], s=20, alpha=0.5, edgecolors="grey", linewidths=2)
	#plt.scatter(data[:, 0], data[:, 1], s = data[:, 1], c=palette[colors.astype(np.int)], camp='Blues', alpha=0.4, edgecolors="grey", linewidths=2)

	ax = fig.add_subplot(211, projection='3d')
	ax.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2], s=20, alpha=0.5, edgecolors="grey", linewidths=2)
	ax.view_init(4,-72)

	plt.title("{}".format(title))
	# plt.axis("off")
	plt.savefig(image_path)

def get_tsne(disease_img, dim):
	tsne = TSNE(n_components=dim, init='pca', random_state=0)
	result = tsne.fit_transform(disease_img)

	return result


def embd_visual_group(disease_img,normal_img,fake_normal_img, title, image_path):
	'''
		visualise embedding with bubble chart.  The area of bubble decreases with its rank.
	:param disease_img: shape: [batch_size, embedding_dim]
	:param vac_titles:
	:param jskr_id:
	:param image_path:
	:param dim:
	:return:
	'''
	disease_img = np.reshape(disease_img, (disease_img.shape[0], -1))
	normal_img = np.reshape(normal_img, (normal_img.shape[0], -1))
	fake_normal_img = np.reshape(fake_normal_img, (fake_normal_img.shape[0], -1))

	disease_img = disease_img
	real_tsne_3d = get_tsne(disease_img, 3)

	normal_tsne_3d = get_tsne(normal_img, 3)

	fake_normal_img = fake_normal_img
	fake_tsne_3d = get_tsne(fake_normal_img, 3)


	print (np.shape (real_tsne_3d))
	print (np.shape (fake_tsne_3d))

	trace1 = go.Scatter3d(
    x=real_tsne_3d[:,0], # それぞれの次元をx, y, zにセットするだけです．
    y=real_tsne_3d[:,1],
    z=real_tsne_3d[:,2],
	name = 'Original Alcoholism Distribution',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(255, 0, 0)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2 # ごちゃごちゃしないように小さめに設定するのがオススメです．
    	)
	)

	trace2 = go.Scatter3d(
    x=normal_tsne_3d[:,0], # それぞれの次元をx, y, zにセットするだけです．
    y=normal_tsne_3d[:,1],
    z=normal_tsne_3d[:,2],
    mode='markers',
	name = 'Original Control Distribution',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(0, 255, 0)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2 # ごちゃごちゃしないように小さめに設定するのがオススメです．
    	)
	)

	trace3 = go.Scatter3d(
    x=fake_tsne_3d[:,0], # それぞれの次元をx, y, zにセットするだけです．
    y=fake_tsne_3d[:,1],
    z=fake_tsne_3d[:,2],
	name = 'Generated Control Distribution',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(0, 0, 255)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2 # ごちゃごちゃしないように小さめに設定するのがオススメです．
    	)
	)

	data=[trace1, trace2, trace3]
	layout=dict(height=700, width=600, title='coil-20 tsne exmaple')
	fig=dict(data=data, layout=layout)
	py.iplot(fig, filename='tsne_example')


	assert (0)

	# no using any GUI dependent backend. (the remove server(appbox) doesn't have GUI)
	plt.switch_backend('agg')
	plt.legend(loc='upper right')

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(2, 1, 2)

	ax.scatter(real_tsne[:, 0], real_tsne[:, 1], s=20, c="red", alpha=0.5, label='original')
	ax.scatter(fake_tsne[:, 0], fake_tsne[:, 1], s=20, c="blue", alpha=0.5, label='filted')

	ax = fig.add_subplot(211, projection='3d')
	ax.scatter(real_tsne_3d[:, 0], real_tsne_3d[:, 1], real_tsne_3d[:, 2], s=20, c="red", alpha=0.5, label='original')
	ax.scatter(fake_tsne_3d[:, 0], fake_tsne_3d[:, 1], fake_tsne_3d[:, 2], s=20, c="blue", alpha=0.5, label='filted')
	ax.view_init(4,-45)

	plt.title("{}".format(title))
	# plt.axis("off")
	plt.savefig(image_path)
