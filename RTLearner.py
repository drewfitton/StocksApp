import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size

    def author(self):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        :return: The GT username of the student 		  	   		 	 	 			  		 			     			  	 
        :rtype: str  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        return "afitton3"
    
    def study_group(self):
        """
        : return: a comma separated string of GT_Name of each member of your study group
        : rtype: str
        """
        return "afitton3"   

    def add_evidence(self, data_x, data_y):
        """  		  	   		 	 	 			  		 			     			  	 
        Add training data to learner  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # Return if data shape is less than leaf size
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Return if all data.y are the same
        if np.all(data_y == data_y[-1]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        
        # Pick random feature, and take mean of 2 random points for SplitVal
        best_feat = np.random.randint(0, data_x.shape[1])
        rand_indices = np.random.choice(data_x.shape[0], 2, replace=False)

        # Set SplitVal, get masks for left and right trees with SplitVal
        split_val = np.mean(data_x[rand_indices, best_feat])
        left_mask = data_x[:, best_feat] <= split_val
        right_mask = ~left_mask

        # If mask has no split (all left or all right), return
        if not np.any(left_mask) or not np.any(right_mask):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Recursively build left tree and right tree with masks
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        # Set root, return root + left and right subtrees
        root = np.array([[best_feat, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))


    def query(self, points):
        """  		  	   		 	 	 			  		 			     			  	 
        Estimate a set of test points using the trained model.  		  	   		 	 	 			  		 			     			  	 
        
        :param points: A numpy array where each row is a query point.
        :return: A numpy array of predicted values.
        """
        # Initialize predictions array
        pred = np.zeros(points.shape[0])

        # Iterate over set of points to predict values
        for i, point in enumerate(points):
            # Start at root node
            node_ind = 0

            # Iterate over tree until we reach leaf node
            while self.tree[node_ind, 0] != -1: 
                # Get split val and feature index
                feat_ind = int(self.tree[node_ind, 0])
                split_val = self.tree[node_ind, 1]

                # Decide to go to left or right child node based on split_val
                if point[feat_ind] <= split_val:
                    node_ind += int(self.tree[node_ind, 2]) 
                else:
                    node_ind += int(self.tree[node_ind, 3])  
            
            # Add prediction value to array
            pred[i] = self.tree[node_ind, 1]

        return pred    	   		 	 	 			  		 			     			  	 
        
    

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")