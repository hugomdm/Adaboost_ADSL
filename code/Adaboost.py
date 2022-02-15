#------------------------------- BINARY CLASSIFICATIONS ---------------------- #

import numpy as np 

from sklearn.tree import DecisionTreeClassifier



class BinaryClassAdaboost():
    """
    """
    
    def __init__(self, n_estimators:int, size:int):
        """
        Initialialisation of Adaboost class
        Parameters: 
            n_estimators: int:  number of weak learners 
            size: int: size de l'échantillon
        """
        self.n_estimators = n_estimators
        self.list_WL = [] #list with model
        self.list_alpha = [] #list with weight of model 
        self.estimator_errors = []
        self.size = size
        
        
    def fit(self, X, y):
        """
        Fit model 
        Parameters: 
            X: array: data
            y: array: vector of class labels where yi E Y= {1,..., k} and k = 2
        """
        ## Step 1: Initialize the weights to a constant
        n_samples = X.shape[0]      
        self.list_WL = [] 
        self.list_alpha = [] 
        self.estimator_errors = []          
    
    
        ##Weights are initialized to 1/Number of samples: 
        w_t = np.array([1/n_samples for x in range(n_samples)])
        
        ## Step 2: Classify with ramdom sampling of data using a weak learner
        #Construction des weaklearner
        
        #for each weak learner
        for t in range(self.n_estimators):

            #Choose and Call the Base/Weak learner
            
            X_sample, y_sample = self.sampling(X, y, w_t)          
            
            #A decision tree with one depth has one node and is called a stump or weak learner
            WL = DecisionTreeClassifier(max_depth=1)
            #Fit the stump model with the ramdom samples
            WL.fit(X_sample, y_sample) #, sample_weight=w_t_sample
            #Get the predicted classes
            y_pred = WL.predict(X)
            
            ##Step 3: Compute error of weak learner
            eps = BinaryClassAdaboost.error_wl(w_t, y_pred, y)
        
            # if the error of the weak learner is higher then 0.5 (worse then random guess) 
            #don't take into account this learner weight
            if eps > 0.5:
                break
            
            #Step 4: Calculate the performance of the weak learner
            #Performance of the weak learner(α) = 0.5* ln (1 – error/error)
            #Calculate alpha for this weak learner
            alpha_t = 0.5 * np.log((1- eps) / eps)
            

            #Step 5: Update weight
            #With the alpha performance (α) the weights of the wrongly classified records are increased
            #and the weights of the correctly classified records decreased.
            y_temp = -alpha_t * np.multiply(y, y_pred) 
            normalized_w_t = np.multiply(w_t, np.exp(y_temp))

            #normalizing the weigths for the sum to be equal do 1
            w_t = normalized_w_t / sum(normalized_w_t)
            
            #store the alpha performance of each weak learner
            self.list_alpha.append(alpha_t)
            #store each weak learner
            self.list_WL.append(WL)
            self.estimator_errors.append(eps)

            
            
        return self

    def predict(self, X):
        """
        predict output of Adaboost 
        Paramters: 
            X: array: data
        Return: 
            y_pred: array: data
        """
        #The final prediction is a compromise between all the weak learners predictions
        list_y_pred = []
        
        #for each weak learner get their prediction

        for WL, w in zip(self.list_WL, self.list_alpha):
            #Final prediction is obtained by the weighted by alpha sum of each weak learner prediction
            list_y_pred.append(WL.predict(X) * w)
         
        #the array of all the predictions

        arr_y_pred = np.array(sum(list_y_pred))
 
        #get -1 if y_pred < 0 or 1 if y_pred > 0
        y_pred = np.sign(arr_y_pred)
        
        return y_pred 
        
    def error_wl(w_t, y_pred, y):
        """
        error of current weaklearner
        Parameters:
            w_t: array:  weight of observation
            y_pred: array: output of wl 
            y: array: labels
        Return: 
            eps: float: error of wl 
        """
        
        ind_err = []
        for i in range(y_pred.shape[0]):
            if y_pred[i] != y[i]:
                ind_err.append(1) 
            else: 
                ind_err.append(0) 
    
        w_ind_err = np.multiply(w_t,ind_err)
        
        eps = np.sum(w_ind_err)
    
        return eps
    
    def sampling(self, X, y, w_t):
        """
        sampling X with w_t 
        Parameters:
            X: array: data
            y: array: labels
            w_t: array: weigth
        Return:
            X_sample: array: sample of X
            y_sample: array: labels corresponding to X_sample
        """
        #put X and y in same array to sample 
        y_temp = np.reshape(y, (y.shape[0], 1))
        data = np.hstack((X, y_temp))
        
        w_t_temp = np.reshape(w_t, (w_t.shape[0], 1))
        data = np.hstack((data,w_t_temp))
        
        #size of sample
        self.size = int(0.75*X.shape[0])
        
        #sample
        #sample = choice(data, size, w_t)
        ch = np.random.choice([x for x in range(data.shape[0])], size=self.size, p=w_t)
        
        sample = data[ch,:]
        
    
        y_sample = sample[:,-2]
        X_sample = sample[:,:-2]
        
        return X_sample, y_sample
    
    
#-------------------------------Multiclass LASSIFICATIONS ---------------------- #


class MultiClassAdaBoost(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.
    n_estimators: integer, optional(default=50)
        The maximum number of estimators
    learning_rate: float, optional(default=1)
    Attributes
    -------------
    estimators_: list of base estimators
    estimator_weights_: array of floats
        Weights for each base_estimator
    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.
    '''

    def __init__(self, n_estimators, learning_rate):
        self.n_estimators = n_estimators
        self.list_WL = [] #list with model
        self.list_alpha = [] #list with weight of model 
        self.learning_rate_ = learning_rate
        self.estimator_errors = []
        self.w = []

    def fit(self, X, y):
        
        ## Step 1: Initialize the weights to a constant
        n_samples = X.shape[0]                
        self.list_WL = [] #list with model
        self.list_alpha = [] #list with weight of model 
        self.estimator_errors = []
        self.w = []

        ##Weights are initialized to 1/Number of samples: 
        w_t = np.array([1/n_samples for x in range(n_samples)])
        
        # So in boost we have to ensure that the predict results have the same classes sort
        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)
        
        
        ## Step 2: Classify with ramdom sampling of data using a weak learner
        #Construction des weaklearner
        
        #for each weak learner
        for t in range(self.n_estimators):
          
            #Choose and Call the Base/Weak learner
            #A decision tree with one depth has one node and is called a stump or weak learner
            WL = DecisionTreeClassifier(max_depth=1)
            #Fit the stump model with the ramdom samples
            WL.fit(X, y, sample_weight=w_t)
            
            y_pred = WL.predict(X)
            
            ##Step 3: Compute error of weak learner
            incorrect = y_pred != y
            estimator_error = np.dot(incorrect, w_t) / np.sum(w_t, axis=0)
            
            # if worse than random guess, stop boosting
            if estimator_error >= 1 - 1 / self.n_classes_:
                break

            # update alphe performance
            alpha_t = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
            self.n_classes_ - 1)
        

            # update sample weight
            w_t *= np.exp(alpha_t * incorrect)
            sample_weight_sum = np.sum(w_t, axis=0)

            # normalize sample weight
            w_t /= sample_weight_sum
            
            #store the alpha performance of each weak learner
            self.list_alpha.append(alpha_t)
            #store each weak learner
            self.list_WL.append(WL)
            # append error
            self.estimator_errors.append(estimator_error)

        return self


    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.list_WL,
                                           self.list_alpha))

        pred /= sum(self.list_alpha)
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)