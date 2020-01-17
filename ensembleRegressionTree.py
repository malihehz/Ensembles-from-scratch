import numpy as np
import time
       
class RegressionTreeNode(object):
    # Constructor
    def __init__(self, att, thr, left, right):  
        self.attribute = att
        self.threshold = thr
        # left and right are either binary classifications or references to
        # decision tree nodes
        self.left = left    
        self.right = right  
       
class DecisionTreeRegressor(object):
    # Constructor
    def __init__(self, max_depth=20, min_samples_split=10, max_mse =0.001):  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_mse = max_mse
       
    def fit(self,x,y):  
        self.root = self._id3(x,y,depth=0)
        
    def fit_randomization(self,x,y):  
        self.root = self._id3_randomization(x,y,depth=0)
       
    def predict(self,x_test):
        pred = np.zeros(len(x_test),dtype=np.float32)
        for i in range(len(x_test)):
            pred[i] = self._predict(self.root,x_test[i])
        return pred
       
    def _id3(self,x,y,depth):
        orig_mse = np.var(y)
        #print('original mse:',orig_mse)
        mean_val = np.mean(y)
        if depth >= self.max_depth or len(y) <= self.min_samples_split or orig_mse <=self.max_mse:
            return mean_val
       
        thr = np.mean(x,axis=0)
        mse_attribute = np.zeros(len(thr))
       
        #x.shape[1]= num of attributes
        for i in range(x.shape[1]):
            less = x[:,i] <= thr[i]
            more = ~ less
            mse_attribute[i] = self._mse(y[less], y[more])
         
        gain = orig_mse - mse_attribute
        #print('Gain:',gain)
        best_att = np.argmax(gain)
        #print('mse best attribute:',mse_attribute[best_att])
        less = x[:,best_att] <= thr[best_att]
        more = ~ less
           
        leftNode = self._id3(x[less,:],y[less],depth+1)#less than thr
        rightNode = self._id3(x[more,:],y[more],depth+1)#more than thr
       
        return RegressionTreeNode(best_att, thr[best_att],leftNode,rightNode)
    def _id3_randomization(self,x,y,depth):
        orig_mse = np.var(y)
        #print('original mse:',orig_mse)
        mean_val = np.mean(y)
        if depth >= self.max_depth or len(y) <= self.min_samples_split or orig_mse <=self.max_mse:
            return mean_val
       
        thr = np.mean(x,axis=0)
        mse_attribute = np.zeros(len(thr))
       
        #x.shape[1]= num of attributes
        for i in range(x.shape[1]):
            less = x[:,i] <= thr[i]
            more = ~ less
            mse_attribute[i] = self._mse(y[less], y[more])
            #part 2 : multiple thr
            for j in range(0,4):
                multi_thr = np.min(x)+(np.max(x)-np.min(x))*np.random.rand(1)
                #multi_thr = thr[i]+ np.random.uniform(-1,1)
                less = x[:,i] <= multi_thr
                more = ~ less
                gain = self._mse(y[less], y[more])
                if mse_attribute[i] > gain:
                    mse_attribute[i] = gain        
                 
           
        gain = orig_mse - mse_attribute
        #print('Gain:',gain)
        best_att = np.argmax(gain)
        #print('mse best attribute:',mse_attribute[best_att])
        less = x[:,best_att] <= thr[best_att]
        more = ~ less
       
     
        leftNode = self._id3_randomization(x[less,:],y[less],depth+1)#less than thr
        rightNode = self._id3_randomization(x[more,:],y[more],depth+1)#more than thr
       
        return RegressionTreeNode(best_att, thr[best_att],leftNode,rightNode)
   
       
    def _mse(self,l,m):
        err = np.append(l - np.mean(l),m-np.mean(m)) #It will issue a warning if either l or m is empty
        return np.mean(err*err)
   
    def _predict(self, dt_node, x):
        if isinstance(dt_node, np.float32):
            return dt_node
        if x[dt_node.attribute] <= dt_node.threshold:
            return self._predict(dt_node.left, x)
        else:
            return self._predict(dt_node.right, x)
   
   
print('\nSolar particle dataset')
skip = 10

x_train = np.load('x_ray_data_train.npy')[::skip]
y_train = np.load('x_ray_target_train.npy')[::skip]
x_test = np.load('x_ray_data_test.npy')[::skip]
y_test = np.load('x_ray_target_test.npy')[::skip]

model = DecisionTreeRegressor()

for k in range(3,12,4): 
    print('number of trees:',k) 
    #print('------------------Decision tree-------------------------')
    start = time.time()
    model.fit(x_train, y_train)
    
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))  
    print('{0:.4f}'.format(np.mean(np.square(train_pred-y_train))))
    print('{0:.4f}'.format(np.mean(np.square(test_pred-y_test))),"\n")
    #print('----------------------Randomization---------------------------')
    test_pred=[]
    train_pred=[]
    #k = 9
    start = time.time()
    for i in range(k):
        model.fit_randomization(x_train, y_train)
        train_pred.append(model.predict(x_train))
        test_pred.append(model.predict(x_test))
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    
    train_pred_bagging=np.array(train_pred)
    train_pred= np.mean(train_pred_bagging, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(train_pred-y_train))))
    
    test_pred_bagging=np.array(test_pred)
    test_pred= np.mean(test_pred_bagging, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(test_pred-y_test))),"\n")
    
    #print('----------------------Bagging---------------------------')
    test_pred=[]
    train_pred=[]
    #k = 9
    start = time.time()
    for i in range(k):
        idx = np.random.randint(x_train.shape[0], size=x_train.shape[0])
        #idxx = np.random.choice(idx,len(idx),replace=True)
        x_train_rand = x_train[idx,:]
        y_train_rand = y_train[idx]
       
        model.fit(x_train_rand, y_train_rand)
        train_pred.append(model.predict(x_train))
        test_pred.append(model.predict(x_test))
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    
    train_pred_bagging=np.array(train_pred)
    train_pred= np.mean(train_pred_bagging, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(train_pred-y_train))))
    
    test_pred_bagging=np.array(test_pred)
    test_pred= np.mean(test_pred_bagging, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(test_pred-y_test))),"\n")
    
    #print('----------------------Boosting---------------------------')
    p = np.ones(x_train.shape[0])*(1/(x_train.shape[0]))
    
    #1. produsing probabilities
    def prob_select(prob,n):
        cp = np.cumsum(prob)
        R = np.sort(np.random.random_sample((n)))* cp[-1]
        i = 0
        S = []
        for r in R:
            while r > cp[i]:
                i += 1
            S.append(i)
        return S
    
    train_pred_boosting=[]
    test_pred_boosting=[]
    start = time.time()
    for i in range(k):
        #2. producing x star
        S = prob_select(p,x_train.shape[0])
        SS = np.random.choice(S,len(S),replace=True) # randomly chosen from S
       
        x_train_S = x_train[SS,:]
        y_train_S = y_train[SS]
       
        model.fit(x_train_S, y_train_S)
    
        #train_pred_S = boosting_model.predict(x_train_S)
        train_pred = model.predict(x_train)
       
        #3,4. increasing probabilitiy of missclassifing ones and recalculating probabilities
        for j in range(len(train_pred)):
            if train_pred[j] != y_train[j]:
                p[j] *= 1.5
           # else:
               #p[j] *= 0.5
        #normalizing
        p /= np.sum(p)
        train_pred_boosting.append(train_pred)
        test_pred_boosting.append(model.predict(x_test))
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    
    train_pred_boosting=np.array(train_pred_boosting)
    
    train_pred= np.mean(train_pred_boosting, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(train_pred-y_train))))
    
    test_pred_boosting=np.array(test_pred_boosting)
    test_pred= np.mean(test_pred_boosting, axis=0)
    print('{0:.4f}'.format(np.mean(np.square(test_pred-y_test))))
    
    #print('---------------------------------------------------------')
    #print('number of trees:',k) 
