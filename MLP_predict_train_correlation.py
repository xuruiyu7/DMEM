import h5py
import os
#import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

os.chdir('D:/Ruiyu/坚果云/徐瑞宇/3d打印变点/data')
#%%
def distance_point(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)
    

def preprocessing():
    layer_num=379
        
    data_total=[]
    
    layer_list = [n for n in range(1,layer_num+1)]
    for layer_temp in layer_list:
        layer_name='1/layer'+str(layer_temp)+'.hdf5'
        sample_dataset_object = h5py.File(layer_name, 'r')
        sample_dataset = sample_dataset_object['OpenData'][:]
        sample_dataset = sample_dataset.transpose()
        data_total.append(sample_dataset)
    
    index_unexposed_layer=[]
    index_anomaly_layer=[26,27,28]
    index_normal_layer=[]
    for layer_temp in range(379):
        data_layer=data_total[layer_temp]
        if len(data_layer)<15000:
            index_unexposed_layer.extend([layer_temp])
        elif np.max(data_layer[:,8])>0:
            index_anomaly_layer.extend([layer_temp])
        else:
            index_normal_layer.extend([layer_temp])
    index_normal_layer.remove(26)      
    index_normal_layer.remove(27)      
    index_normal_layer.remove(28)      
        
    random.seed(116) 
    train_index = random.sample(index_normal_layer,250)
    test_index = list(set(index_normal_layer) - set(train_index))
    
    data_layer=data_total[0]
    x_min=np.min(data_layer[:,0])
    x_max=np.max(data_layer[:,0])
    y_min=np.min(data_layer[:,1])
    y_max=np.max(data_layer[:,1])
    
    x_normal_total_train=[]
    x_normal_total_test=[]
    y_normal_total_train=[]
    y_normal_total_test=[]
    for layer_temp in index_normal_layer:
        print(layer_temp)
        data_layer=data_total[layer_temp]
        len_data=len(data_layer)
        power=data_layer[:,5]
        temperature=data_layer[:,6]
        
        dis_point=np.zeros((len_data,1))
        x_old=data_layer[0,0]
        y_old=data_layer[0,1]
        for i in range(1,len_data):
            x_new=data_layer[i,0]
            y_new=data_layer[i,1]
            dis_point[i,0]=distance_point(x_new,y_new,x_old,y_old)
            x_old=x_new
            y_old=y_new
                
        #index_path_change_new=np.where(dis_point>0.1)[0]
        
        index_path_change=np.where(data_layer[:,5]<4000)[0]
        index_path_change_list=[index_path_change[0]]
        for i in range(len(index_path_change)-1):
            if index_path_change[i+1]-index_path_change[i]==1:
                continue
            index_path_change_list.extend([index_path_change[i+1]])
        index_path_change=np.array(index_path_change_list)
        
        num_previous_temperature=10
        previous_temperature=np.zeros((len_data,num_previous_temperature))
        for i in range(num_previous_temperature):
            previous_temperature[i:,i]=temperature[:len_data-i]
        for i in index_path_change:
            for j in range(num_previous_temperature-1):
                if i+j>=len_data:
                    break
                previous_temperature[i+j,j+1:]=0
                
        num_previous_power=10
        previous_power=np.zeros((len_data,num_previous_power))
        for i in range(num_previous_power):
            previous_power[i:,i]=power[:len_data-i]
        for i in index_path_change:
            for j in range(num_previous_power-1):
                if i+j>=len_data:
                    break
                previous_power[i+j,j+1:]=0

        last_seg_length=np.zeros((len_data,1))
        length_present_seg=np.zeros((len_data,1))
        last_seg_length_temp=0
        for j in range(len(index_path_change)-1):
            last_seg_length[index_path_change[j]:index_path_change[j+1]]=last_seg_length_temp
            last_seg_length_temp=index_path_change[j+1]-index_path_change[j]
            length_present_seg[index_path_change[j]:index_path_change[j+1],0]=np.arange(last_seg_length_temp)+1
        last_seg_length[index_path_change[-1]:]=last_seg_length_temp
        length_present_seg[index_path_change[-1]:,0]=np.arange(len_data-index_path_change[-1])+1
                

        x=data_layer[:,0]
        y=data_layer[:,1]
        distance=np.min(np.array([x-x_min,x_max-x,y-y_min,y_max-y]),0)
        distance=distance.reshape((-1,1))
        
        num_seg=np.zeros((len_data,1))
        num_seg_temp=1
        for j in range(len(index_path_change)-1):
            if j>1:
                if dis_point[index_path_change[j]]>5:
                    #print(index_path_change[j],dis_point[index_path_change[j]])
                    num_seg_temp=1
                else:
                    num_seg_temp+=1
            num_seg[index_path_change[j]:index_path_change[j+1]]=num_seg_temp
        
        x_layer=np.concatenate((previous_power,distance,last_seg_length,length_present_seg,dis_point,num_seg),1)
        

        y_layer=temperature

        x_layer=x_layer.tolist()
        y_layer=y_layer.tolist()
        
        if layer_temp in train_index:
            x_normal_total_train.append(x_layer)
            y_normal_total_train.append(y_layer)
        if layer_temp in test_index:
            x_normal_total_test.append(x_layer)
            y_normal_total_test.append(y_layer)
    return x_normal_total_train,y_normal_total_train,x_normal_total_test,y_normal_total_test
#%%

X_train_original,Y_train_original,X_test_original,Y_test_original=preprocessing()
'''
i=0
X_temp = np.array(X_train_original[i])
Y_temp = np.array(Y_train_original[i])

'''

#%%
import tensorflow as tf
from tensorflow.keras import Model
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

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
        self.test_loss = None
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
        self.test_loss = L
        test_loss.update_state(self.test_loss)
        return self.test_loss
        
#%%
import time
from tensorflow.keras import metrics
#%%
MyModel1 = MyModel()
#cov_left = np.zeros(batch_size)
epochs = 10

train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')


num_train = sum([len(Y_train_original[i]) for i in range(len(Y_train_original))])
num_test = sum([len(Y_test_original[i]) for i in range(len(Y_test_original))])

xi_train = np.zeros(len(Y_train_original))
xi_test = np.zeros(len(Y_test_original))

for i in range(len(xi_train)):
    xi_train[i] = np.mean(Y_train_original[i])
for i in range(len(xi_test)):
    xi_test[i] = np.mean(Y_test_original[i])
xi_test=xi_test-np.mean(xi_train) 
xi_train=xi_train-np.mean(xi_train) 


Y_train_diff_total = np.zeros(num_train)
Y_test_diff_total = np.zeros(num_test)

lambda1 = 1
lambda2 = 0
xi_train_pre=np.zeros(len(Y_train_original))
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
            MyModel1.train_step(x_batch_train,y_batch_train,y_batch_train_diff,tf.constant(lambda1,dtype=tf.double),tf.constant(lambda2,dtype=tf.double))
            if batch % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (batch, train_loss.result().numpy()))
                #print(MyModel1.Adam.lr.numpy().item())

        for batch, (x_batch_test, y_batch_test,y_batch_test_diff) in enumerate(data_test, 1):
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


    y_pred = MyModel1.predict(X_test[:3000])
    plt.figure()
    plt.plot(y_pred)
    plt.plot(Y_test[:3000])
    plt.show()
    plt.figure()
    plt.hist(xi_train)
    plt.show()
    
    if max(abs(xi_train-xi_train_pre))<0.0001:
        break
    xi_train_pre=xi_train.copy()    
    


#MyModel1.save('../code-paper-v2/checkpoints_AM_EXP_1')





    
