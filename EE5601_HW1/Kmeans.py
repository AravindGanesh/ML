import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class Kmeans():
    def __init__(self, X, K=3):
        self.K = K
        self.x = X
        pixels = np.unique(self.x, axis=0)
        self.means = pixels[np.random.choice(len(pixels), self.K, replace=False)]
        # initalize means by picking K random unique (non repeating) pixels from given image
    
    # update the means and calculate error
    def update(self):
        self.new_means = []
        # array with k values of pixels (in same order i.e. same index)
        self.assign_k = np.array([np.argmin(np.array([np.linalg.norm(pix-mean) for mean in self.means])) for pix in self.x])
        for k in np.arange(self.K):
            kth_cluster = self.x[np.argwhere(self.assign_k==k).flatten()] # inputs belonging to same cluster
            self.new_means.append(np.mean(kth_cluster, axis=0)) # updated mean
            
        self.error = np.linalg.norm(self.new_means - self.means)
        return np.array(self.new_means)
    
    # cluster the pixels by updating the means till error < ε 
    def cluster(self,n_iter):
        count = 0 # initial iteration
        self.means = self.update()
        print('Iteration count = ', count, '  -->  Error = ', self.error)
        count += 1
        while(count < n_iter): # interations untill error <= ε 
            self.means = self.update()
            print('Iteration count = ', count, '  -->  Error = ', self.error)
            count += 1
        print('Done')
    
    # print the means and clusters at convergence
    def get_clusters(self): 
        print('MEANS at covergence: ' , self.means)
        for k in range(self.K): 
            kth_cluster = self.x[np.argwhere(self.assign_k==k).flatten()]
            print('Cluster ', k+1, ' : ', kth_cluster)
            
    # pixel belonging to each cluster is show by the color of the centroid of the cluster
    # def show_clusters(self):
    #     cluster_img = self.means[self.assign_k].astype(np.int).reshape(self.img_shape)
    #     fig = plt.figure(figsize=self.img_shape[:2])
    #     fig.add_subplot(1,2,1) # show original image
    #     plt.imshow(self.img)
    #     fig.add_subplot(1,2,2) # show clusters
    #     plt.imshow(cluster_img)
    #     plt.show()

