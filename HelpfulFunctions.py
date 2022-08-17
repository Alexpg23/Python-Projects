# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:43:45 2017

@author: Dominic
"""
# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.ticker as mticker
from sklearn.metrics import plot_confusion_matrix

mpl.rcParams['pcolor.shading']

colorList = ['g','b','r','y']

def setcolor(y_list, colors = colorList):
	
	c = []
	for y in y_list:
		if y == 0:
			c.append(colors[0])
		elif y==1:
			c.append(colors[1])
		elif y==2:
			c.append(colors[2])
		else:
			c.append('k')
			
	return c

markerList = ['.','o','v','D']

def setmarker(y_list, markers = markerList):
	
	c = []
	for y in y_list:
		if y == 0:
			c.append(markers[0])
		elif y==1:
			c.append(markers[1])
		elif y==2:
			c.append(markers[2])
		else:
			c.append('o')
			
	return c


def plot_cm(clf, X, y, display_labels):
    
    mpl.rcParams.update({'font.size': 16})
    plot_confusion_matrix(clf, X, y, display_labels=display_labels,cmap=mpl.cm.Blues);


def print_cm(cm,labels,plt):      

    print(cm)
    plt.figure(figsize=(10, 10))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Greys)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)

#    ticks_loc = ax.get_yticks().tolist()
#    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#from sklearn.metrics import confusion_matrix
#from sklearn.linear_model import LogisticRegression

def featureSpacePlot(Xname,Yname,data,y,classifier,plt,titleName=""):

    h = .01  # step size in the mesh

    X = data[Xname]
    Y = data[Yname]
        
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X.min() - .05, X.max() + .05
    y_min, y_max = Y.min() - .05, Y.max() + .05
    
    ######################################
    
    nx = int((x_max - x_min)/ h)
    ny = int((y_max - y_min)/ h)
    
    g1 = np.linspace(x_min, x_max, nx)
    g2 = np.linspace(y_min, y_max, ny)

    xx, yy = np.meshgrid(g1,g2)

    ######################################
    
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Plot also the training points
    plt.scatter(X,Y, c=y, edgecolors='k', cmap=cmap_bold, alpha = 1.0)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_bold, alpha=0.1, shading='auto')

    plt.xlabel(Xname)
    plt.ylabel(Yname)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(titleName, fontsize=16)

    return plt

#####################################################################

import time as time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

#####################################################################

from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, plt, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#####################################################################
	
    