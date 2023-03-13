# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import rand

from scipy import stats

import os
os.chdir('D:/Ruiyu/坚果云/徐瑞宇/3d打印变点/data')
#%%
def cal_T2(gamma,mean_xi,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c):
    residual_layer = gamma
    T2_list_layer = []
    rRr = 0
    rR1 = 0
 
    for kk in range(len(gamma)):
        if kk==0:
            T2_list_layer.extend([residual_layer[kk]**2/(sigma_xi+sigma_gamma)]) 
            rRr+=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk+1]*residual_layer[kk]
            rR1+=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk+1]
        elif kk<len(gamma)-1:
            temp1=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk-1]*residual_layer[kk]
            temp2=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk-1]
            rRr+=temp1
            rR1+=temp2
            sum1r1 = (lambda_b*2+lambda_c)*(kk-1)+(lambda_a+lambda_b)*2
            T2_layer = rRr/sigma_gamma-(rR1/sigma_gamma)**2/(1/sigma_xi+sum1r1/sigma_gamma)
            T2_layer /= (kk+1)
            T2_list_layer.extend([T2_layer])
            
            rRr-=temp1
            rR1-=temp2
            rRr+=lambda_c*residual_layer[kk]**2 \
                +lambda_b*residual_layer[kk+1]*residual_layer[kk] \
                +lambda_b*residual_layer[kk-1]*residual_layer[kk]
            rR1+=lambda_c*residual_layer[kk] \
                +lambda_b*residual_layer[kk+1] \
                +lambda_b*residual_layer[kk-1]   
        else:
            rRr+=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk-1]*residual_layer[kk]
            rR1+=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk-1]
            sum1r1 = (lambda_b*2+lambda_c)*(kk-1)+(lambda_a+lambda_b)*2
            T2_layer = rRr/sigma_gamma-(rR1/sigma_gamma)**2/(1/sigma_xi+sum1r1/sigma_gamma)
            T2_layer /= (kk+1)
            T2_list_layer.extend([T2_layer])
    return T2_list_layer
#%%
def cal_GLR(gamma,mean_xi,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c):
    residual_layer = gamma
    GLR_list_layer = []
    mu_xi_best_layer = []
    sigma_gamma_best_layer = []
    
    log_1=np.log(sigma_gamma+sigma_xi)
    rRr=0
    rR1=0
    
    for kk in range(len(gamma)):
        if kk==0:
            temp_exp_inside=(residual_layer[0])**2/(sigma_gamma+sigma_xi)  
            log_pro=-1/2*log_1-temp_exp_inside/2
            log_pro_best=-1/2*np.log(sigma_xi)
            GLR_layer=-log_pro+log_pro_best
            GLR_list_layer.extend([GLR_layer])
            sigma_gamma_best=0
            mu_xi_best_layer.extend([gamma[0]])
            sigma_gamma_best_layer.extend([0])
            rRr+=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk+1]*residual_layer[kk]
            rR1+=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk+1]
        elif kk<len(gamma)-1:
            temp1=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk-1]*residual_layer[kk]
            temp2=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk-1]
            rRr+=temp1
            rR1+=temp2
            sum1r1 = (lambda_b*2+lambda_c)*(kk-1)+(lambda_a+lambda_b)*2
            mu_xi_best=rR1/sum1r1
            sigma_gamma_temp=1/(kk+1-1)*(rRr-rR1**2/sum1r1)
            sigma_gamma_best=sigma_gamma_temp-1/(kk+1-1)*sigma_gamma_best**2/(sigma_gamma_best+sigma_xi*sum1r1)
            sigma_gamma_best=max(0,sigma_gamma_best)+0.0001
            
            mu_xi_best_layer.extend([mu_xi_best])
            sigma_gamma_best_layer.extend([sigma_gamma_best])
            temp_2_1 = sigma_gamma_best/sigma_gamma
            temp_2_2_1 = sigma_gamma_best+sigma_xi*sum1r1
            temp_2_2_2 = sigma_gamma+sigma_xi*sum1r1
            temp_2_2 = temp_2_2_1/temp_2_2_2
            diff_2 = (kk+1-1)*(temp_2_1-1-np.log(temp_2_1))+temp_2_2-1-np.log(temp_2_2)+mu_xi_best**2*sum1r1/temp_2_2_1+sigma_xi*(sigma_gamma_best-sigma_gamma)**2*sum1r1/(sigma_gamma*temp_2_2_1*temp_2_2_2)
            GLR_list_layer.extend([diff_2/2])
            rRr-=temp1
            rR1-=temp2
            rRr+=lambda_c*residual_layer[kk]**2 \
                +lambda_b*residual_layer[kk+1]*residual_layer[kk] \
                +lambda_b*residual_layer[kk-1]*residual_layer[kk]
            rR1+=lambda_c*residual_layer[kk] \
                +lambda_b*residual_layer[kk+1] \
                +lambda_b*residual_layer[kk-1]   
            
        else:
            rRr+=lambda_a*residual_layer[kk]**2+lambda_b*residual_layer[kk-1]*residual_layer[kk]
            rR1+=lambda_a*residual_layer[kk]+lambda_b*residual_layer[kk-1]
            sum1r1 = (lambda_b*2+lambda_c)*(kk-1)+(lambda_a+lambda_b)*2
            mu_xi_best=rR1/sum1r1
            sigma_gamma_temp=1/(kk+1-1)*(rRr-rR1**2/sum1r1)
            sigma_gamma_best=sigma_gamma_temp-1/(kk+1-1)*sigma_gamma_best**2/(sigma_gamma_best+sigma_xi*sum1r1)
            
            mu_xi_best_layer.extend([mu_xi_best])
            sigma_gamma_best_layer.extend([sigma_gamma_best])
            temp_2_1 = sigma_gamma_best/sigma_gamma
            temp_2_2_1 = sigma_gamma_best+sigma_xi*sum1r1
            temp_2_2_2 = sigma_gamma+sigma_xi*sum1r1
            temp_2_2 = temp_2_2_1/temp_2_2_2
            diff_2 = (kk+1-1)*(temp_2_1-1-np.log(temp_2_1))+temp_2_2-1-np.log(temp_2_2)+mu_xi_best**2*sum1r1/temp_2_2_1+sigma_xi*(sigma_gamma_best-sigma_gamma)**2*sum1r1/(sigma_gamma*temp_2_2_1*temp_2_2_2)
            GLR_list_layer.extend([diff_2/2])
    return GLR_list_layer#,mu_xi_best_layer,sigma_gamma_best_layer
#%%
def cal_EWMA(gamma,mean_xi,sigma_xi,sigma_gamma):
    lambda_EWMA=0.2
    EWMA_list_layer = []
    
    statistics_EWMA=0
    for kk in range(len(gamma)):
        if kk==1:
            statistics_EWMA=gamma[kk]**2/(sigma_xi+sigma_gamma)
        else:
            statistics_EWMA=gamma[kk]**2/(sigma_xi+sigma_gamma)*lambda_EWMA+statistics_EWMA*(1-lambda_EWMA)
        EWMA_list_layer.extend([statistics_EWMA])
    return EWMA_list_layer
#%%
def cal_CUSUM(gamma,mean_xi,sigma_xi,sigma_gamma,minmiss=2.69):
    minn1 = 10
    minn2 = 10
    summax = 0
    summin = 0
    
    sumsave1=[]
    sumsave2=[]
    CUSUM_list_layer = []
    for i in range(len(gamma)):
        summax = max(0,summax+(gamma[i]-mean_xi-minmiss)) 
        summin = min(0,summin+(gamma[i]-mean_xi+minmiss))
        sumsave1.extend([summax])
        sumsave2.extend([summin])
        
        if sumsave1[i]<minn1:    
            minn1 = sumsave1[i]
        if sumsave2[i]>minn2:    
            minn2 = sumsave2[i]
        d = max(sumsave1[i]-minn1,-sumsave2[i]+minn2)
        CUSUM_list_layer.extend([d])
    return CUSUM_list_layer

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


#%%
def print_error(value):
    print("error: ", value)

#%%
def gen1_combine(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,beta_1,beta_2,beta_3,beta_4):
    TFModel=tf.keras.models.load_model('../code-paper-v2/checkpoints_cov_case_1_v2')
    np.random.seed(i)
    data_x_layer=generate_x_IC(n_variable,n_time)
    data_y1_IC_layer=np.dot(data_x_layer,beta_1)
    xi_IC=np.random.normal()*np.sqrt(sigma_xi_IC)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
        gamma_previous=gamma_temp[i]
    data_y1_IC_layer+=xi_IC+gamma_temp
    beta_1_OC = np.random.uniform(-1,1,n_variable)
    g_delta=np.dot(data_x_layer,beta_1_OC)
    
    data_y1_OC_layer=data_y1_IC_layer+change_delta*g_delta
    
    beta_1_OC = np.random.uniform(-1,1,n_variable)
    y_pred = TFModel.predict(data_x_layer)
    y_pred = y_pred.reshape((-1))
    
    residual_layer = data_y1_OC_layer-y_pred-0.09
    
    sigma_xi_est=1.00169
    sigma_gamma_est=0.25665
    alpha_est=0.511926
    lambda_a = 1/(1-alpha_est**2)
    lambda_b = -alpha_est/(1-alpha_est**2)
    lambda_c = (1+alpha_est**2)/(1-alpha_est**2)  

    T2_layer=cal_T2(residual_layer,0,sigma_xi_est,sigma_gamma_est,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_est,sigma_gamma_est,lambda_a,lambda_b,lambda_c)
    EWMA_layer=cal_EWMA(residual_layer,0,sigma_xi_est,sigma_gamma_est)
    CUSUM_layer=cal_CUSUM(residual_layer,0,sigma_xi_est,sigma_gamma_est)
        
    return [T2_layer,GLR_layer,EWMA_layer,CUSUM_layer]

#%%
sigma_xi_IC=1
sigma_gamma_IC=0.25
alpha_IC=0.5
n_variable=10

#%%
n_time=100000
num_OC=100

#%%
import multiprocessing
if __name__ == '__main__':
    MyModel1=tf.keras.models.load_model('../code-paper-v2/checkpoints_cov_case_1_v2')
    
    np.random.seed(412)
    beta_1 = np.random.uniform(-1,1,n_variable)
    beta_2 = np.random.uniform(-1,1)
    beta_3 = np.random.uniform(-1,1,n_variable)
    beta_4 = rand(n_variable,n_variable,density=0.2)
    beta_4 = beta_4.todense()
    
    data_gen1=[]
    pool = multiprocessing.Pool(processes=10)
    for i in range(num_OC):
        pool.apply_async(gen1_combine, args=(i,n_time,sigma_xi_IC,sigma_gamma_IC,alpha_IC,beta_1,beta_2,beta_3,beta_4,),callback=data_gen1.append,error_callback=print_error) 
    pool.close()
    pool.join()
    
    
    for change_delta in [-0.15,-0.1,-0.05,0,0.05,0.1,0.15]:
        for layer_temp in range(num_OC):
            beta_1_OC = np.random.uniform(-1,1,n_variable)
            y_pred = MyModel1.predict(data_x_layer)
            y_pred = y_pred.reshape((-1))
            gamma_temp = data_y1_OC_layer-y_pred
        
            gamma_temp = gamma_temp.tolist()
