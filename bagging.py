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
    def __init__(self, max_depth=10, min_samples_split=5, min_accuracy =1):  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_accuracy = min_accuracy
        
    def fit_partOne(self,x,y):  
        self.root = self._ID3_partOne(x,y,depth=0)
              
    def predict(self,x_test):
        pred = np.zeros(len(x_test),dtype=int)
        for i in range(len(x_test)):
            pred[i] = self._classify(self.root,x_test[i])
        return pred
       
    def _ID3_partOne(self,x,y,depth):
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
       
        leftNode = self._ID3_partOne(x[less,:],y[less],depth+1)#less than thr
        rightNode = self._ID3_partOne(x[more,:],y[more],depth+1)#more than thr
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
#x=x[1:5,:]
#y=y[1:5]

#Split data into training and testing
ind = np.random.permutation(len(y))
split_ind = int(len(y)*0.8)
x_train = x[ind[:split_ind]]
x_test = x[ind[split_ind:]]
y_train = y[ind[:split_ind]]
y_test = y[ind[split_ind:]]

model = DecisionTreeClassifier()

#i is number of classifiers in ensemble which are trees here
for i in range(5):
    #x_train_rand = x_train[np.random.choice(x_train.shape[0],x_train.shape[0],replace=True),:]
    idx = np.random.randint(x_train.shape[0], size=x_train.shape[0])
    idxx = np.random.choice(idx,len(idx),replace=True)
    x_train_rand = x_train[idxx,:]
    y_train_rand = y_train[idxx]
    
    start = time.time()
    model.fit_partOne(x_train_rand, y_train_rand)
    elapsed_time = time.time()-start
    #print('\n','Elapsed_time training partOne {0:.6f} '.format(elapsed_time))  

    train_pred = model.predict(x_train_rand)

    start = time.time()
    test_pred = model.predict(x_test)
    elapsed_time = time.time()-start
    #print('Elapsed_time testing partOne  {0:.6f} '.format(elapsed_time))  

    train_acc = np.sum(train_pred==y_train_rand)/len(train_pred)
    print('\n','train accuracy partOne:', train_acc)

    test_acc = np.sum(test_pred==y_test)/len(test_pred)
    print('test accuracy partOne:', test_acc)
