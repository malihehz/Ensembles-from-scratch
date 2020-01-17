import numpy as np
import time
       
class DecisionTreeNode(object):
    # Constructor
    def __init__(self, att, thr, left, right):  
        self.attribute = att
        self.threshold = thr
        # left and right are either binary classifications or references to
        # decision tree nodes
        self.left = left    
        self.right = right  

class DecisionTreeClassifier(object):
    # Constructor
    def __init__(self, max_depth=10, min_samples_split=10, min_accuracy =1):  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_accuracy = min_accuracy
       
    def fit(self,x,y):  
        self.root = self._ID3(x,y,depth=0)
        
    def fit_randomization(self,x,y):  
        self.root = self._ID3_randomization(x,y,depth=0)
         
    def predict(self,x_test):
        pred = np.zeros(len(x_test),dtype=int)
        for i in range(len(x_test)):
            pred[i] = self._classify(self.root,x_test[i])
        return pred
       
    def _ID3(self,x,y,depth):
        mean_val = np.mean(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or max([mean_val,1-mean_val])>=self.min_accuracy:
            return int(round(mean_val))
        #mean of each attribute,col
        thr = np.mean(x,axis=0)
 
        #x.shape[1]= num of attributes
        entropy_attribute = np.zeros(len(thr))
        #x.shape[1]= num of attributes
        for i in range(x.shape[1]):
            less = x[:,i] <= thr[i]
            more = ~ less
            entropy_attribute[i] = self._entropy(y[less], y[more])
           
        #print(entropy_attribute)
        best_att = np.argmin(entropy_attribute)  
        less = x[:,best_att] <= thr[best_att]
        more = ~ less
       
        leftNode = self._ID3(x[less,:],y[less],depth+1)#less than thr
        rightNode = self._ID3(x[more,:],y[more],depth+1)#more than thr
        return DecisionTreeNode(best_att, thr[best_att],leftNode,rightNode)
    
    def _ID3_randomization(self,x,y,depth):
        mean_val = np.mean(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or max([mean_val,1-mean_val])>=self.min_accuracy:
            return int(round(mean_val))
        #mean of each attribute,col
        thr = np.mean(x,axis=0)
 
        #x.shape[1]= num of attributes
        entropy_attribute = np.zeros(len(thr))
        #x.shape[1]= num of attributes
        for i in range(x.shape[1]):
            less = x[:,i] <= thr[i]
            more = ~ less
            entropy_attribute[i] = self._entropy(y[less], y[more])
            #part 2 : multiple thr
            for j in range(0,5):
                multi_thr = np.min(x)+(np.max(x)-np.min(x))*np.random.rand(1)
                #multi_thr = thr[i]+ np.random.uniform(-1,1)
                less = x[:,i] <= multi_thr
                more = ~ less
                gain = self._entropy(y[less], y[more])
                if entropy_attribute[i] > gain:
                    entropy_attribute[i] = gain
   
        #print(entropy_attribute)
        best_att = np.argmin(entropy_attribute)  
        less = x[:,best_att] <= thr[best_att]
        more = ~ less
         
        leftNode = self._ID3_randomization(x[less,:],y[less],depth+1)#less than thr
        rightNode = self._ID3_randomization(x[more,:],y[more],depth+1)#more than thr
       
        return DecisionTreeNode(best_att, thr[best_att],leftNode,rightNode)
   
    def _entropy(self,l,m):
        ent = 0
        for p in [l,m]:
            if len(p)>0:
                pp = sum(p)/len(p)
                pn = 1 - pp
                if pp<1 and pp>0:
                    ent -= len(p)*(pp*np.log2(pp)+pn*np.log2(pn))
        ent = ent/(len(l)+len(m))
        return ent  
   
    def _classify(self, dt_node, x):
        if dt_node in [0,1]:
            return dt_node
        if x[dt_node.attribute] < dt_node.threshold:
            return self._classify(dt_node.left, x)
        else:
            return self._classify(dt_node.right, x)
   
x = []
y = []
infile = open("magic04.txt","r")
for line in infile:
    y.append(int(line[-2:-1] =='g'))
    x.append(np.fromstring(line[:-2], dtype=float,sep=','))
infile.close()
   
x = np.array(x).astype(np.float32)
y = np.array(y)

#Split data into training and testing
ind = np.random.permutation(len(y))
split_ind = int(len(y)*0.8)
x_train = x[ind[:split_ind]]
x_test = x[ind[split_ind:]]
y_train = y[ind[:split_ind]]
y_test = y[ind[split_ind:]]


for k in range(3,12,4):     
    print('number of trees:',k) 
    #print('------------------Decision tree-------------------------')
    start = time.time()
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    
    train_pred = model.predict(x_train)
    train_acc = np.sum(train_pred==y_train)/len(train_pred)
    
    test_pred = model.predict(x_test)
    test_acc = np.sum(test_pred==y_test)/len(test_pred)
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))  
    print('{0:.2f}'.format(train_acc*100))
    print('{0:.2f}'.format(test_acc*100),"\n")
    #print('----------------------Randomization---------------------------')
    test_pred=[]
    train_pred=[]
    #k = 3
    start = time.time()
    for i in range(k):
        model.fit_randomization(x_train, y_train)
        train_pred.append(model.predict(x_train))
        test_pred.append(model.predict(x_test))
    
    train_pred=np.array(train_pred)
    train_pred_max= np.round(np.mean(train_pred, axis=0)) 
    train_acc = np.sum(train_pred_max==y_train)/len(y_train)
    
    test_pred=np.array(test_pred)
    test_pred_max= np.round(np.mean(test_pred, axis=0))  
    test_acc_bagg = np.sum(test_pred_max==y_test)/len(y_test)
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    print('{0:.2f}'.format(train_acc*100))
    print('{0:.2f}'.format(test_acc_bagg*100),"\n")
    
    #print('----------------------Bagging---------------------------')
    test_pred=[]
    train_pred=[]
    #k = 3
    start = time.time()
    for i in range(k):
        idx = np.random.randint(x_train.shape[0], size=x_train.shape[0])
        #idxx = np.random.choice(idx,len(idx),replace=True)
        x_train_rand = x_train[idx,:]
        y_train_rand = y_train[idx]
       
        model.fit(x_train_rand, y_train_rand)
        train_pred.append(model.predict(x_train))
        test_pred.append(model.predict(x_test))
    
    train_pred=np.array(train_pred)
    train_pred_max= np.round(np.mean(train_pred, axis=0)) 
    train_acc = np.sum(train_pred_max==y_train)/len(y_train)
    
    test_pred=np.array(test_pred)
    test_pred_max= np.round(np.mean(test_pred, axis=0))  
    test_acc_bagg = np.sum(test_pred_max==y_test)/len(y_test)
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    print('{0:.2f}'.format(train_acc*100))
    print('{0:.2f}'.format(test_acc_bagg*100),"\n")
    
    #print('----------------------Boosting---------------------------')
    #boosting_model = DecisionTreeClassifier()
    
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
    
    
    train_pred_boosting=np.array(train_pred_boosting)
    train_pred_max= np.round(np.mean(train_pred_boosting, axis=0))  
    train_acc_boos = np.sum(train_pred_max==y_train)/len(train_pred_max)
    
    test_pred_boosting=np.array(test_pred_boosting)
    test_pred_max= np.round(np.mean(test_pred_boosting, axis=0))  
    test_acc_boos = np.sum(test_pred_max==y_test)/len(test_pred_max)
    
    elapsed_time = time.time()-start
    print('{0:.2f}'.format(elapsed_time))
    print('{0:.2f}'.format(train_acc_boos*100))
    print('{0:.2f}'.format(test_acc_boos*100))
    #print('---------------------------------------------------------')
    #print('number of trees:',k) 
    k = k+2
    
