# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.sparse import rand
import statsmodels.tsa.api as smt
import os
os.chdir('D:/Ruiyu/坚果云/徐瑞宇/3d打印变点/data')

#%%
def generate_x_IC(n_variable,n_time):
    x_layer=[]
    x_initial=np.random.uniform(-1,1,n_variable)
    x_layer.append(x_initial)
    for i in range(n_time-1):
        x_change=0.03*np.random.randn(n_variable)
        x_current=x_initial+x_change
        x_current[x_current<-1]=x_current[x_current<-1]*(-1)-2
        x_current[x_current>1]=x_current[x_current>1]*(-1)+2
        x_layer.append(x_current)
        x_initial=x_current
    x_layer=np.array(x_layer)
    return x_layer

#%%
import tensorflow as tf
from tensorflow.keras import Model
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

batch_size = 512*4

class MyModel(Model):
    # 可以传入一些超参数，用以动态构建模型
    # __init_——()方法在创建模型对象时被调用
    # input_shape: 输入层和输出层的节点个数（输入层实际要比这多1，因为有个bias）
    # 使用方法：直接传入实际的input_shape即可，在call中也直接传入原始Input_tensor即可
    # 一切关于数据适配模型的处理都在模型中实现
    def __init__(self,**kwargs):
        # 调用父类__init__()方法
        super(MyModel, self).__init__()
        
        self.Adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.1)
        self.train_loss = None
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        # 输入数据
        output_dense_1 = self.dense_1(input_tensor)
        output_dense_2 = self.dense_2(output_dense_1)
        output_dense_3 = self.dense_3(output_dense_2)
        output_dense_4 = self.dense_4(output_dense_3)
        output_dropout_1 = self.dropout_1(output_dense_4)
        output = self.dense_5(output_dropout_1)
        return output
    
    def get_loss(self, input_tensor, y_true, cov_left, lambda1, lambda2):
        # print("get_loss")
        output_dense_1 = self.dense_1(input_tensor)
        output_dense_2 = self.dense_2(output_dense_1)
        output_dense_3 = self.dense_3(output_dense_2)
        output_dense_4 = self.dense_4(output_dense_3)
        output_dropout_1 = self.dropout_1(output_dense_4)
        output = self.dense_5(output_dropout_1)
        
        # 计算loss
        gamma = y_true-tf.cast(output,dtype=tf.double)
        loss1 = tf.reduce_mean (tf.square (gamma))*lambda1
        loss2 = tf.reduce_mean (tf.multiply (gamma,cov_left))*lambda2
        loss = loss1+loss2
        return loss
    
    def get_grad(self, input_tensor, y_true, cov_left, lambda1, lambda2):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(input_tensor, y_true, cov_left, lambda1, lambda2)
            # 保存一下loss，用于输出
            self.train_loss = L
            g = tape.gradient(L, self.variables)
        return g

    def train_step(self, input_tensor, y_true, cov_left, lambda1, lambda2):
        g = self.get_grad(input_tensor, y_true, cov_left, lambda1, lambda2)
        self.Adam.apply_gradients(zip(g, self.variables))
        train_loss.update_state(self.train_loss)
        return self.train_loss
    
    def test_step(self, input_tensor, y_true, cov_left, lambda1, lambda2):
        L = self.get_loss(input_tensor, y_true, cov_left, lambda1, lambda2)
        test_loss.update_state(L)
#%%
#Case 1
n_variable=10
n_time=30000
n_IC=100
sigma_xi=1
alpha=0.5
sigma_gamma=0.25

np.random.seed(412)
beta_1 = np.random.uniform(-1,1,n_variable)
beta_2 = np.random.uniform(-1,1)
beta_3 = np.random.uniform(-1,1,n_variable)
beta_4 = rand(n_variable,n_variable,density=0.2)
beta_4 = beta_4.todense()

data_x_IC=[]
data_y1_IC=[]
data_y2_IC=[]
data_y3_IC=[]
data_y1_IC_noise=[]
data_y2_IC_noise=[]
data_y3_IC_noise=[]

np.random.seed(0)
xi_IC_true=np.random.normal(size=n_IC)*np.sqrt(sigma_xi)
xi_IC_true=(np.array(xi_IC_true)-np.mean(xi_IC_true))/np.std(xi_IC_true)
#plt.hist(xi_IC_true)

np.random.seed(901)
for i in range(n_IC):
    data_x_IC_layer=generate_x_IC(n_variable,n_time)
    data_x_IC.append(data_x_IC_layer)
    data_y1_IC_layer=np.dot(data_x_IC_layer,beta_1)
    data_y2_IC_layer=data_y1_IC_layer+beta_2*np.sin(np.dot(data_x_IC_layer,beta_3))
    a=np.multiply(np.dot(data_x_IC_layer,beta_4),data_x_IC_layer)
    data_y3_IC_layer=data_y2_IC_layer+np.sum(a.A,axis = 1)
    data_y1_IC.append(data_y1_IC_layer.copy())
    data_y2_IC.append(data_y2_IC_layer.copy())
    data_y3_IC.append(data_y3_IC_layer.copy())
    #plt.plot(data_y1_IC_layer[:3000])
    xi_temp=xi_IC_true[i]
    
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha, scale=np.sqrt(sigma_gamma*(1-alpha**2)))
        gamma_previous=gamma_temp[i]
    
    data_y1_IC_layer+=xi_temp+gamma_temp
    data_y2_IC_layer+=xi_temp+gamma_temp
    data_y3_IC_layer+=xi_temp+gamma_temp
    #plt.plot(data_y1_IC_layer[:3000])
    '''
    plt.plot(data_y1_IC_layer)
    plt.plot(data_y2_IC_layer)
    plt.plot(data_y3_IC_layer)
    '''
    data_y1_IC_noise.append(data_y1_IC_layer)
    data_y2_IC_noise.append(data_y2_IC_layer)
    data_y3_IC_noise.append(data_y3_IC_layer)
#plt.hist(xi_IC_true)

#%%
results=[]
#%%
#1
num_train = 80
num_test = 20
Y_train_original = data_y1_IC_noise[:80]
Y_test_original = data_y1_IC_noise[-20:]
X_train_original = data_x_IC[:80]
X_test_original = data_x_IC[-20:]

#%%
import time
from tensorflow.keras import metrics
MyModel1 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)

for i in range(80):
    xi_train[i] = np.mean(Y_train_original[i])
for i in range(20):
    xi_test[i] = np.mean(Y_test_original[i])
xi_test=xi_test-np.mean(xi_train) 
xi_train=xi_train-np.mean(xi_train) 

Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(80)
for iter_0 in range(5):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel1.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel1.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel1.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel1.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel1.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel1.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    Y_train_diff_total = []
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel1.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel1.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred-xi_train[layer_temp]-mean_xi_train
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel1.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp-xi_test[layer_temp]

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)

    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel1.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    '''
    y_pred = MyModel1.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    
    y_pred = MyModel1.predict(X_test_original[0]).reshape(-1)
    y_truth = data_y1_IC[80].reshape(-1)
    plt.figure()
    plt.plot(abs(y_truth[:3000]-y_pred[:3000]))
    plt.ylim([0,0.45])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel1.save('../code-paper-v2/checkpoints_cov_case_1_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel1.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  


MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel1.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y1_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_1_DMEM.npy',results)

#%%
MyModel1_2 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10
exp_lambda_exp_pre=-1

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
for iter_0 in range(30):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel1_2.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel1_2.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel1_2.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel1_2.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel1_2.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel1_2.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update 
    Y_train_diff_total = []
    acf_total=[]
  
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel1_2.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel1_2.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)


    '''
    y_pred = MyModel1_2.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel1_2.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    if abs(exp_lambda_exp-exp_lambda_exp_pre)<0.0001:
        break
    exp_lambda_exp_pre=exp_lambda_exp


#%%
#MyModel1_2.save('../code-paper-v2/checkpoints_cov_case_12_v2')


alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel1_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  


MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel1_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred
    MSE+=np.mean((data_y1_IC[layer_temp]-y_pred)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[0,0,0,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_1_SM1.npy',results)


#%%
MyModel1_3 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)
Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(80)
for iter_0 in range(100):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel1_3.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel1_3.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel1_3.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel1_3.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel1_3.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel1_3.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel1_3.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel1_3.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)


    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel1_3.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel1_3.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel1_3.save('../code-paper-v2/checkpoints_cov_case_13_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel1_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)

MSE=0
rRr_est=0    
for layer_temp in range(100):
    y_pred = MyModel1_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y1_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y1_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    rRr_est+=np.sum(gamma_temp**2)

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,0,MSE]
#np.save('../code-paper-v2/simu_I_1_SM2.npy',results)


#%%
num_train = 80
num_test = 20
Y_train_original = data_y2_IC_noise[:80]
Y_test_original = data_y2_IC_noise[-20:]
X_train_original = data_x_IC[:80]
X_test_original = data_x_IC[-20:]

MyModel2 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)

for i in range(80):
    xi_train[i] = np.mean(Y_train_original[i])
for i in range(20):
    xi_test[i] = np.mean(Y_test_original[i])
xi_test=xi_test-np.mean(xi_train) 
xi_train=xi_train-np.mean(xi_train) 

Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(80)
for iter_0 in range(100):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel2.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel2.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel2.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel2.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel2.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel2.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    Y_train_diff_total = []
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel2.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel2.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred-xi_train[layer_temp]-mean_xi_train
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel2.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp-xi_test[layer_temp]

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)

    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel2.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel2.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel2.save('../code-paper-v2/checkpoints_cov_case_2_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  


MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y2_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_2_DMEM.npy',results)

#%%
MyModel2_2 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10
exp_lambda_exp_pre=-1

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
for iter_0 in range(30):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel2_2.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel2_2.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel2_2.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel2_2.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel2_2.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel2_2.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update 
    Y_train_diff_total = []
    acf_total=[]
  
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel2_2.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel2_2.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)


    '''
    y_pred = MyModel2_2.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel2_2.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    if abs(exp_lambda_exp-exp_lambda_exp_pre)<0.0001:
        break
    exp_lambda_exp_pre=exp_lambda_exp


#%%
#MyModel2_2.save('../code-paper-v2/checkpoints_cov_case_22_v2')


alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel2_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  


MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel2_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred
    MSE+=np.mean((data_y2_IC[layer_temp]-y_pred)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[0,0,0,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_2_SM1.npy',results)


#%%
MyModel2_3 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)
Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(80)
for iter_0 in range(30):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel2_3.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel2_3.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel2_3.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel2_3.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel2_3.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel2_3.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel2_3.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel2_3.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)


    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel2_3.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel2_3.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel2_3.save('../code-paper-v2/checkpoints_cov_case_23_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel2_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)

MSE=0
rRr_est=0    
for layer_temp in range(100):
    y_pred = MyModel2_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y2_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y2_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    rRr_est+=np.sum(gamma_temp**2)

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,0,MSE]
#np.save('../code-paper-v2/simu_I_2_SM2.npy',results)


#%%
num_train = 80
num_test = 20
Y_train_original = data_y3_IC_noise[:80]
Y_test_original = data_y3_IC_noise[-20:]
X_train_original = data_x_IC[:80]
X_test_original = data_x_IC[-20:]

MyModel3 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)

for i in range(80):
    xi_train[i] = np.mean(Y_train_original[i])
for i in range(20):
    xi_test[i] = np.mean(Y_test_original[i])
xi_test=xi_test-np.mean(xi_train) 
xi_train=xi_train-np.mean(xi_train) 
    
Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
    

xi_train_pre=np.zeros(80)
for iter_0 in range(100):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel3.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel3.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel3.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel3.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel3.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel3.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    Y_train_diff_total = []
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel3.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel3.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred-xi_train[layer_temp]-mean_xi_train
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel3.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp-xi_test[layer_temp]

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)

    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel3.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel3.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel3.save('../code-paper-v2/checkpoints_cov_case_3_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  

MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y3_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_3_DMEM.npy',results)

#%%
MyModel3_2 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10
exp_lambda_exp_pre=-1

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
for iter_0 in range(30):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
         
    Y_train = np.array([i for k in Y_train for i in k])
    if iter_0>=1:
        Y_train_diff_total = np.array([i for k in Y_train_diff_total for i in k])
    ##
    Y_test = Y_test_original.copy()
        
    Y_test = np.array([i for k in Y_test for i in k])
    if iter_0>=1:
        Y_test_diff_total = np.array([i for k in Y_test_diff_total for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel3_2.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel3_2.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel3_2.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel3_2.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel3_2.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel3_2.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update 
    Y_train_diff_total = []
    acf_total=[]
  
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel3_2.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        
        index_path_change=np.where(np.array(Y_train_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)   
        
        gamma_temp = gamma_temp.tolist()
        acf = smt.stattools.acf(gamma_temp, nlags=100)
        #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
        acf_total.append(acf.tolist())
        
        Y_train_diff_right = gamma_temp[1:]
        Y_train_diff_right.extend([0])
        Y_train_diff_right=np.array(Y_train_diff_right)
        Y_train_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_train_diff_left = [0]
        Y_train_diff_left.extend(gamma_temp[:-1])
        Y_train_diff_left=np.array(Y_train_diff_left)
        Y_train_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_train_diff = Y_train_diff_right+Y_train_diff_left
        Y_train_diff = Y_train_diff.tolist()
        Y_train_diff_total.append(Y_train_diff)
    
    
    acf_mean = np.mean(np.array(acf_total),0)
    exp_lambda_exp = acf_mean[1]
    print('alpha=%.4f'%exp_lambda_exp)
    lambda_b = -exp_lambda_exp/(1-exp_lambda_exp**2)
    lambda_c = (1+exp_lambda_exp**2)/(1-exp_lambda_exp**2)
    
    lambda1 = lambda_c
    lambda2 = lambda_b
    
    Y_test_diff_total = []
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel3_2.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred
        xi_test[layer_temp] = np.mean(gamma_temp)
        gamma_temp = gamma_temp

        index_path_change=np.where(np.array(Y_test_original[layer_temp])<800)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)        
        
        gamma_temp = gamma_temp.tolist()
        
        Y_test_diff_right = gamma_temp[1:]
        Y_test_diff_right.extend([0])
        Y_test_diff_right=np.array(Y_test_diff_right)
        Y_test_diff_right[[i-1 for i in index_path_change_list[1:]]] = 0
        Y_test_diff_left = [0]
        Y_test_diff_left.extend(gamma_temp[:-1])
        Y_test_diff_left=np.array(Y_test_diff_left)
        Y_test_diff_left[[i for i in index_path_change_list[1:]]] = 0
        Y_test_diff = Y_test_diff_right+Y_test_diff_left
        Y_test_diff = Y_test_diff.tolist()
        Y_test_diff_total.append(Y_test_diff)


    '''
    y_pred = MyModel3_2.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel3_2.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    if abs(exp_lambda_exp-exp_lambda_exp_pre)<0.0001:
        break
    exp_lambda_exp_pre=exp_lambda_exp


#%%
#MyModel3_2.save('../code-paper-v2/checkpoints_cov_case_32_v2')


alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel3_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred

    gamma_temp = gamma_temp.tolist()
    acf = smt.stattools.acf(gamma_temp, nlags=10)
    #pacf = smt.stattools.pacf(gamma_temp, nlags=100)
    alpha_total.extend([acf[1]])
    
alpha_est=np.mean(alpha_total)
lambda_a_est = 1/(1-alpha_est**2)
lambda_b_est = -alpha_est/(1-alpha_est**2)
lambda_c_est = (1+alpha_est**2)/(1-alpha_est**2)  


MSE=0
rRr_est=0    
for layer_temp in range(100):
    rRr_est_temp=0
    y_pred = MyModel3_2.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred
    MSE+=np.mean((data_y3_IC[layer_temp]-y_pred)**2)  
    for kk in range(len(y_pred)):
        if (kk>0) and (kk<len(y_pred)-1):
            rRr_est_temp+=lambda_c_est*gamma_temp[kk]**2 \
                +lambda_b_est*gamma_temp[kk+1]*gamma_temp[kk] \
                +lambda_b_est*gamma_temp[kk-1]*gamma_temp[kk]
    
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    rRr_est_temp+=lambda_a_est*gamma_temp[0]**2+lambda_b_est*gamma_temp[1]*gamma_temp[0]
    print(rRr_est_temp)
    rRr_est+=rRr_est_temp

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[0,0,0,sigma_gamma_est,alpha_est,MSE]
#np.save('../code-paper-v2/simu_I_3_SM1.npy',results)


#%%
MyModel3_3 = MyModel()
#cov_left = np.zeros(batch_size)

#%%
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


xi_train = np.zeros(num_train)
xi_test = np.zeros(num_test)
Y_train_diff_total = np.zeros(num_train*n_time)
Y_test_diff_total = np.zeros(num_test*n_time)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(80)
for iter_0 in range(30):
    test_loss_0 = 1000000
    num_no_decrease = 0
    ##
    Y_train = Y_train_original.copy()
    for layer_temp in range(len(Y_train_original)):
        Y_train[layer_temp] = (np.array(Y_train[layer_temp]) - xi_train[layer_temp]).tolist()
         
    Y_train = np.array([i for k in Y_train for i in k])
    ##
    Y_test = Y_test_original.copy()
    for layer_temp in range(len(Y_test_original)):
        Y_test[layer_temp] = (np.array(Y_test[layer_temp]) - xi_test[layer_temp]).tolist()
        
    Y_test = np.array([i for k in Y_test for i in k])
    

    
    X_train = np.array([i for k in X_train_original for i in k])
    X_test = np.array([i for k in X_test_original for i in k])
    
    Y_train=Y_train.reshape(-1,1)
    Y_train_diff_total=Y_train_diff_total.reshape(-1,1)
    np.random.seed(116) 
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)
    np.random.seed(116)
    np.random.shuffle(Y_train_diff_total)

    Y_test=Y_test.reshape(-1,1)
    Y_test_diff_total=Y_test_diff_total.reshape(-1,1)

    data_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train,Y_train_diff_total)).batch(batch_size)
    data_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test,Y_test_diff_total)).batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        print('Iter %s / %s ,Epoch %s / %s' % (iter_0,10,epoch, epochs))
        for batch, (x_batch_train, y_batch_train,y_batch_train_diff) in enumerate(data_train, 1):
            MyModel3_3.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel3_3.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
            #print(MyModel3_3.get_loss(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double)))
            MyModel3_3.test_step(x_batch_test,y_batch_test,y_batch_test_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))

        end = time.time()
        run_epoch_time = int((end - start) % 60)
        print('ETA : %ss, loss : %s| val_loss : %s'
            % ( run_epoch_time, train_loss.result().numpy(), test_loss.result().numpy() ))
        test_loss_temp = test_loss.result().numpy()
        
        train_loss.reset_states()
        test_loss.reset_states()
        '''
        y_pred = MyModel3_3.predict(X_test[:3000])
        plt.figure()
        plt.plot(y_pred)
        plt.plot(Y_test[:3000])
        np.var(y_pred.reshape((-1))-np.array(Y_test[:3000]))
        plt.show()
        
        for layer_temp in range(100):
            y_pred = MyModel3_3.predict(X_train_original[layer_temp])
            #print(np.mean((y_pred.reshape((-1))+xi_train[0]-np.array(Y_train_original[layer_temp]))**2))
            print(np.mean((y_pred.reshape((-1))-np.array(Y_train_original[layer_temp]))**2))
        '''


        if test_loss_temp<test_loss_0:
            test_loss_0 = test_loss_temp
        else:
            num_no_decrease+=1
        if num_no_decrease>=3:
            break
    
    ##update xi_train and xi_test
    acf_total=[]
    for layer_temp in range(len(X_train_original)):
        y_pred = MyModel3_3.predict(X_train_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_train_original[layer_temp]-y_pred
        #print(np.mean(gamma_temp))
        xi_train[layer_temp] = np.mean(gamma_temp)
        
    mean_xi_train = np.mean(xi_train)
    xi_train = xi_train - mean_xi_train
        
    
    for layer_temp in range(len(X_test_original)):
        y_pred = MyModel3_3.predict(X_test_original[layer_temp])
        y_pred = y_pred.reshape((-1))
        gamma_temp = Y_test_original[layer_temp]-y_pred- mean_xi_train
        xi_test[layer_temp] = np.mean(gamma_temp)


    xi_test = xi_test - mean_xi_train

    '''
    y_pred = MyModel3_3.predict(X_train_original[layer_temp])
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp][:3000])
    plt.plot(data_y1_IC[layer_temp][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    plt.figure()
    plt.plot(data_y1_IC_noise[layer_temp])
    plt.plot(y_pred)
    plt.show()    
    '''
    
    y_pred = MyModel3_3.predict(X_test_original[0])
    plt.figure()
    plt.plot(data_y1_IC_noise[80][:3000])
    plt.plot(data_y1_IC[80][:3000])
    plt.plot(y_pred[:3000])
    plt.show()
    '''
    plt.figure()
    plt.plot(Y_test_original[0])
    plt.plot(y_pred)
    plt.show()
    '''
    
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    print(max(abs(xi_train-xi_train_pre)))
    print(np.mean(xi_train),np.mean(xi_train_pre))
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()

#%%
#MyModel3_3.save('../code-paper-v2/checkpoints_cov_case_33_v2')


xi_est=np.zeros(100)
alpha_total=[]
for layer_temp in range(100):
    y_pred = MyModel3_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred
    xi_est[layer_temp] = np.mean(gamma_temp)

    
mean_xi_est = np.mean(xi_est)
xi_est = xi_est - mean_xi_est
sigma_xi_est=np.var(xi_est)

MSE=0
rRr_est=0    
for layer_temp in range(100):
    y_pred = MyModel3_3.predict(data_x_IC[layer_temp])
    y_pred = y_pred.reshape((-1))
    gamma_temp = data_y3_IC_noise[layer_temp]-y_pred-xi_est[layer_temp]-mean_xi_est
    MSE+=np.mean((data_y3_IC[layer_temp]-y_pred-mean_xi_est)**2)  
    rRr_est+=np.sum(gamma_temp**2)

MSE=MSE/100
sigma_gamma_est=rRr_est/100/len(y_pred)
    
results=[xi_est,mean_xi_est,sigma_xi_est,sigma_gamma_est,0,MSE]
#np.save('../code-paper-v2/simu_I_3_SM2.npy',results)






