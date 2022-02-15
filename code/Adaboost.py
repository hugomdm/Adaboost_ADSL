#------------------------------- BINARY CLASSIFICATIONS ---------------------- #

import numpy as np 

from sklearn.tree import DecisionTreeClassifier



class BinaryClassAdaboost():
    """
    """
    
    def __init__(self, n_estimators:int):
        """
        Initialialisation of Adaboost class
        Parameters: 
            n_estimators: int:  number of weak learners 
        """
        self.n_estimators = n_estimators
        self.list_WL = [] #list with model
        self.list_alpha = [] #list with weight of model 
        self.estimator_errors = []

        
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

            #Sample of X
            X_sample, y_sample = BinaryClassAdaboost.sampling(X, y, w_t)                    

            #Choose and Call the Base/Weak learner
            #A decision tree with one depth has one node and is called a stump or weak learner
            WL = DecisionTreeClassifier(max_depth=1)
            #Fit the stump model with the ramdom samples
            WL.fit(X_sample, y_sample) 
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
            y_pred: array: prediction of Adaboost 
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
    
    def sampling( X, y, w_t):
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
        
        #size of sample
        size = int(0.75*X.shape[0])
                
        #get index of data kept
        ch = np.random.choice([x for x in range(data.shape[0])], size=size, p=w_t)
        
        sample = data[ch,:]
        
    
        y_sample = sample[:,-1]
        X_sample = sample[:,:-1]
        
        return X_sample, y_sample
    
    
    ## Function to use cross_validate of sklearn
    def get_params(self, deep=True):
        '''
        '''
        return {'n_estimators': self.n_estimators}

    def set_params(self, **parameters):
        '''
        '''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    
#-------------------------------Multiclass LASSIFICATIONS ---------------------- #


class MultiClassAdaBoost():
    """
    """

    def __init__(self, n_estimators: int):
        """
        Parameters:
            n_estimators: int: number of Weak Learner member of vote
        """
        self.n_estimators = n_estimators
        self.list_WL = [] #list with model
        self.list_beta = [] #list with weight of model 
        self.estimator_errors = []
        self.w = []

    def fit(self, X, y):
        """
        Parameters:
            X: array: data
            y: array: class
        """
        ## Step 1: Initialize the weights to a constant
        n_samples = X.shape[0]                
        self.list_WL = [] #list with model
        self.list_beta = [] #list with weight of model 
        self.estimator_errors = []
        self.w = []

        ##Weights are initialized to 1/Number of samples: 
        w_t = np.array([1/n_samples for x in range(n_samples)])
        
        # So in boost we have to ensure that the predict results have the same classes sort
        self.K = np.sort(np.unique(np.array(y)))
        self.n_K = len(self.K)
        
        
        ## Step 2: Classify with ramdom sampling of data using a weak learner
        #Construction des weaklearner
        
        #for each weak learner
        for t in range(self.n_estimators):
            
            #Do a sample of data to train WL
            X_sample, y_sample = MultiClassAdaBoost.sampling(X, y, w_t)   
            
            #Choose and Call the Base/Weak learner
            #A decision tree with one depth has one node and is called a stump or weak learner
            WL = DecisionTreeClassifier(max_depth=1)
            #Fit the stump model with the ramdom samples
            WL.fit(X_sample, y_sample)
            y_pred = WL.predict(X)
            
            ##Step 3: Compute error of weak learner
            incorrect = y_pred != y
            index = np.where(incorrect)
            eps = sum(w_t[index])
            # if worse than random guess, stop boosting
            if eps > 1/2:
                break
            
            #Compute beta_t the weight of weak learner
            beta_t = eps/(1 - eps)

            w_t_temp = np.array([w_t[i] if incorrect[i] else w_t[i]*beta_t for i in range(w_t.shape[0])])
            w_t = w_t_temp / sum(w_t_temp)

            
            #store the alpha performance of each weak learner
            self.list_beta.append(beta_t)
            #store each weak learner
            self.list_WL.append(WL)
            # append error
            self.estimator_errors.append(eps)


        return self



    def predict(self, X):
        """
        Parameters: 
            X: array: data
        returns:
            y_pred: array: prediction of Adaboost
        """
        #initialise matrix of score row: nrow of X, col: count of class 
        #score will be the sum of 1/beta_t when a class is predicr by WL
        y_pred_score = np.zeros((X.shape[0], self.n_K))
        
        #For each WL
        for beta_t, WL in zip(self.list_beta, self.list_WL):
            #Predict of WL
            y_pred_t = WL.predict(X)
            #For each id
            for e, y_pred_t_i in enumerate(y_pred_t):
                #we increase the y_predict columns by log(1/beta_t)
                y_pred_score[e, int(y_pred_t_i)] += np.log(1/beta_t)
        
        #We get for each id the number of column with max value 
        y_pred = np.argmax(y_pred_score, axis=1)
            
        return y_pred #, y_pred_score
    
    
    def sampling( X, y, w_t):
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
        
        #size of sample
        size = int(0.75*X.shape[0])
                
        #get index of data kept
        ch = np.random.choice([x for x in range(data.shape[0])], size=size, p=w_t)
        
        sample = data[ch,:]
        
    
        y_sample = sample[:,-1]
        X_sample = sample[:,:-1]
        
        return X_sample, y_sample
    
    ## Function to use cross_validate of sklearn
    def get_params(self, deep=True):
        """
        """
        return {'n_estimators': self.n_estimators}


    def set_params(self, **parameters):
        """
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        return self