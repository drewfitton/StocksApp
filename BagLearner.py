import numpy as np

class BagLearner(object):
    
    def __init__(self, learner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False):
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.kwargs = kwargs
        self.boost = boost

    def author(self):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        :return:The GT username of the student 		  	   		 	 	 			  		 			     			  	 
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

        # Add evidence for each learner (each bag)
        for learner in self.learners:
            mask = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            learner.add_evidence(data_x[mask], data_y[mask])

    def query(self, points):
        """  		  	   		 	 	 			  		 			     			  	 
        Estimate a set of test points using the trained model.  		  	   		 	 	 			  		 			     			  	 
        
        :param points: A numpy array where each row is a query point.
        :return: A numpy array of predicted values.
        """
        # Initialize predictions array
        preds = np.zeros((len(self.learners), len(points)))  

        # Iterate over learners, querying each bag with points
        for i, learner in enumerate(self.learners):
            preds[i] = learner.query(points) 
        
        # print(preds)
        # Take mean of all bag predictions
        if len(preds) > 0:
            return np.mean(preds, axis=0)
        
        return np.zeros((len(self.learners), len(points))) 