# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:16:21 2022

@author: A
"""
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

def print_error(value):
    print("error: ", value)


def gen0(i,n_time,sigma_xi_IC,sigma_gamma_IC,alpha_IC):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    np.random.seed(i)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    residual_layer=xi_temp+gamma_temp
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_without_omega=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_without_omega=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    return [T2_layer,GLR_layer,T2_layer_without_omega,GLR_layer_without_omega]


def gen1(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    np.random.seed(i)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    residual_layer=xi_temp+gamma_temp+change_delta
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]


def gen2(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    np.random.seed(i)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC*(1+change_delta), scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC*(1+change_delta))**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    residual_layer=xi_temp+gamma_temp
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]



def gen3(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    np.random.seed(i)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1+change_delta)*(1-alpha_IC**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    residual_layer=xi_temp+gamma_temp
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]



def gen4(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    n_variable=10
    np.random.seed(i)

    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC)**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    x_layer=generate_x_IC(n_variable,n_time)
    beta_1 = np.random.uniform(-1,1,n_variable)
    g_delta=np.dot(x_layer,beta_1)
    residual_layer=xi_temp+gamma_temp+g_delta*change_delta
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]



def gen5(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    n_variable=10
    np.random.seed(i)

    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC)**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    x_layer=generate_x_IC(n_variable,n_time)
    beta_1 = np.random.uniform(-1,1,n_variable)
    g_delta=np.sin(np.dot(x_layer,beta_1))
    residual_layer=xi_temp+gamma_temp+g_delta*change_delta
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]


def gen6(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w):
    lambda_a = 1/(1-alpha_IC**2)
    lambda_b = -alpha_IC/(1-alpha_IC**2)
    lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
    n_variable=10
    np.random.seed(i)
    gamma_temp=np.zeros(n_time)
    gamma_previous=0
    for i in range(n_time):
        gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC)**2)))
        gamma_previous=gamma_temp[i]    
    xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
    x_layer=generate_x_IC(n_variable,n_time)
    beta_4 = rand(n_variable,n_variable,density=0.2)
    beta_4 = beta_4.todense()
    a=np.multiply(np.dot(x_layer,beta_4),x_layer)
    g_delta=np.sum(a.A,axis = 1)

    residual_layer=xi_temp+gamma_temp+g_delta*change_delta
    T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
    T2_layer_w=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    GLR_layer_w=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,1,0,1)
    '''
    plt.plot(residual_layer)
    plt.plot(g_delta*change_delta)
    
  
    plt.plot(T2_layer)
    plt.plot(up_bound_T2)
    plt.plot(lower_bound_T2)
    plt.ylim([0.5,1.5])
    
    plt.plot(T2_layer_w)
    plt.plot(up_bound_T2_w)
    plt.plot(lower_bound_T2_w)
    plt.ylim([0.5,1.5])
    
    plt.plot(T2_layer[:1000])
    plt.plot(T2_layer_w[:1000])
    '''
    
    T2_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
            T2_delay=i+1
            T2_TF=1
            break    
    if T2_TF==0:
        T2_delay=len(residual_layer)
    
    GLR_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer[i]>up_bound_GLR[i]:
            GLR_delay=i+1
            GLR_TF=1
            break    
    if GLR_TF==0:
        GLR_delay=len(residual_layer)
    '''
    T2_offline_TF=0
    if  (T2_layer[-1]>stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1) or (T2_layer[-1]<-stats.norm.ppf(q=1-0.01/2)*np.sqrt(2/(n_time))+1):
        T2_offline_TF=1
    
    GLR_offline_TF=0
    if  GLR_layer[-1]>stats.chi2.ppf(q=1-0.01,df=2)/2:
        GLR_offline_TF=1
    '''
    T2_w_TF=0
    for i in range(len(residual_layer)):
        if  (T2_layer_w[i]>up_bound_T2_w[i]) or (T2_layer_w[i]<lower_bound_T2_w[i]):
            T2_w_delay=i+1
            T2_w_TF=1
            break    
    if T2_w_TF==0:
        T2_w_delay=len(residual_layer)
    
    GLR_w_TF=0
    for i in range(len(residual_layer)):
        if  GLR_layer_w[i]>up_bound_GLR_w[i]:
            GLR_w_delay=i+1
            GLR_w_TF=1
            break    
    if GLR_w_TF==0:
        GLR_w_delay=len(residual_layer)
        
    return [T2_TF,GLR_TF,T2_w_TF,GLR_w_TF,T2_delay,GLR_delay,T2_w_delay,GLR_w_delay]



#%%
#experiment 1 T2
sigma_xi_IC=1
sigma_gamma_IC=0.25
alpha_IC=0.5
lambda_a = 1/(1-alpha_IC**2)
lambda_b = -alpha_IC/(1-alpha_IC**2)
lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  

#%%
n_time=100000
up_bound_T2=np.zeros(n_time)
lower_bound_T2=np.zeros(n_time)
alpha_error=0.00023
#alpha_error=0.00034
#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=stats.norm.ppf(q=1-alpha_error)
for i in range(n_time):
    up_bound_T2[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(1000):
    up_bound_T2[i]=stats.chi2.ppf(q=1-alpha_error,df=i+1)/(i+1)

UCL_GLR=stats.chi2.ppf(q=1-0.00023,df=2)/2
#UCL_GLR=stats.chi2.ppf(q=1-0.00034,df=2)/2
up_bound_GLR=np.ones(n_time)*UCL_GLR



up_bound_T2_w=np.zeros(n_time)
lower_bound_T2_w=np.zeros(n_time)
alpha_error=6e-6
#lambda1_T2=5
lambda1_T2=stats.norm.ppf(q=1-alpha_error/2)

for i in range(n_time):
    up_bound_T2_w[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(1000):
    up_bound_T2_w[i]=stats.chi2.ppf(q=1-alpha_error/2,df=i+1)/(i+1)
for i in range(n_time):
    lower_bound_T2_w[i]=-lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(1000):
    lower_bound_T2_w[i]=stats.chi2.ppf(q=alpha_error/2,df=i+1)/(i+1)
for i in range(100):
    lower_bound_T2_w[i]=0

#UCL_GLR=stats.chi2.ppf(q=1-0.0000172,df=2)/2
up_bound_GLR_w=np.ones(n_time)*10.97





#%%
import multiprocessing
if __name__ == '__main__':
    #%%
    print(1)
    '''
    num_OC=1000
    n_variable=10
    results_ini=[]
    pool = multiprocessing.Pool(processes=20)
    for i in range(num_OC):
        pool.apply_async(gen0, args=(i,n_time,sigma_xi_IC,sigma_gamma_IC,alpha_IC,),callback=results_ini.append,error_callback=print_error) 
    pool.close()
    pool.join()
    print('done')
        
    T2_layer_list=[]
    GLR_layer_list=[]
    T2_layer_without_omega_list=[]
    GLR_layer_without_omega_list=[]
    
    for i in range(num_OC):
        T2_layer_list.append(results_ini[i][0])
        GLR_layer_list.append(results_ini[i][1])
        T2_layer_without_omega_list.append(results_ini[i][2])
        GLR_layer_without_omega_list.append(results_ini[i][3])

    num_TF=0
    delay_list=[]
    for j in range(1000):
        T2_layer=T2_layer_list[j]
        for i in range(n_time):
            if  (T2_layer[i]>up_bound_T2[i]) or (T2_layer[i]<lower_bound_T2[i]):
                delay_list.extend([i+1])
                print(j+1,i+1,T2_layer[i])
                num_TF+=1
                break
    print('............')
    num_TF=0
    delay_list=[]
    for j in range(1000):
        GLR_layer=GLR_layer_list[j]
        for i in range(n_time):
            if  (GLR_layer[i]>up_bound_GLR[i]):
                delay_list.extend([i+1])
                print(j+1,i+1,GLR_layer[i])
                num_TF+=1
                break
    print('............')
    num_TF=0
    delay_list=[]
    for j in range(1000):
        T2_layer_without_omega=T2_layer_without_omega_list[j]
        for i in range(n_time):
            if  (T2_layer_without_omega[i]>up_bound_T2_w[i]) or (T2_layer_without_omega[i]<lower_bound_T2_w[i]):
                delay_list.extend([i+1])
                print(j+1,i+1,T2_layer_without_omega[i])
                num_TF+=1
                break
    print('............')
    num_TF=0
    delay_list=[]
    for j in range(1000):
        GLR_layer_without_omega=GLR_layer_without_omega_list[j]
        for i in range(n_time):
            if  (GLR_layer_without_omega[i]>up_bound_GLR_w[i]):
                delay_list.extend([i+1])
                print(j+1,i+1,GLR_layer_without_omega[i])
                num_TF+=1
                break
    
    '''
    #%%
    '''
    
    num_OC=1000
    n_variable=10
    
    results_total=[]
    print(1)
    for change_delta in [-4,-2,-1,0,1,2,4]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen1, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])

    for change_delta in [-0.03,-0.02,-0.01,0,0.01,0.02,0.03]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen2, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])


    for change_delta in [-0.02,-0.01,-0.005,0,0.005,0.01,0.02]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen3, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])

    for change_delta in [-0.15,-0.1,-0.05,0,0.05,0.1,0.15]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen4, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])

    
    for change_delta in [-0.2,-0.15,-0.1,0,0.1,0.15,0.2]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen5, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])

    for change_delta in [-0.15,-0.1,-0.05,0,0.05,0.1,0.15]:
        results=[]
        pool = multiprocessing.Pool(processes=20)
        for i in range(num_OC):
            pool.apply_async(gen6, args=(i,n_time,change_delta,sigma_xi_IC,sigma_gamma_IC,alpha_IC,up_bound_T2,lower_bound_T2,up_bound_GLR,up_bound_T2_w,lower_bound_T2_w,up_bound_GLR_w,),callback=results.append,error_callback=print_error) 
        pool.close()
        pool.join()
        
        T2_TF_list=[]
        T2_delay_list=[]
        GLR_TF_list=[]
        GLR_delay_list=[]
        T2_w_TF_list=[]
        GLR_w_TF_list=[]
        T2_w_delay_list=[]
        GLR_w_delay_list=[]
        for i in range(num_OC):
            T2_TF_list.extend([results[i][0]])
            GLR_TF_list.extend([results[i][1]])
            T2_w_TF_list.extend([results[i][2]])
            GLR_w_TF_list.extend([results[i][3]])
            T2_delay_list.append(results[i][4])
            GLR_delay_list.append(results[i][5])
            T2_w_delay_list.append(results[i][6])
            GLR_w_delay_list.append(results[i][7])
        print(change_delta,num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list))
        results_total.append([num_OC/(np.sum(T2_TF_list)+1e-8),num_OC/(np.sum(GLR_TF_list)+1e-8),num_OC/(np.sum(T2_w_TF_list)+1e-8),num_OC/(np.sum(GLR_w_TF_list)+1e-8),np.mean(T2_delay_list),np.mean(GLR_delay_list),np.mean(T2_w_delay_list),np.mean(GLR_w_delay_list)])

    '''
    #%%
    n_variable=10
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-16,-9,0,9,16]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=1.4
        residual_layer=xi_temp+gamma_temp+change_delta
        T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        #GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(T2_layer,label=r'$\delta$=%.0f'%(change_delta))

    plt.plot(up_bound_T2,'black',ls='--')
     
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks([0.95,1.00,1.05,1.1,1.15],fontsize=12)
    plt.ylim([0.95,1.15])
    plt.savefig('../code-paper-v2/fig/monitor_11.png',doi="300",bbox_inches='tight')
        
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-6,-4,0,4,6]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        residual_layer=xi_temp+gamma_temp+change_delta
        #T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(GLR_layer,label=r'$\delta$=%.0f'%(change_delta))

    plt.plot(up_bound_GLR,'black',ls='--')
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0,25])
    plt.savefig('../code-paper-v2/fig/monitor_12.png',doi="300",bbox_inches='tight')
        
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-0.04,-0.02,0,0.02,0.04]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC*(1+change_delta), scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC*(1+change_delta))**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        residual_layer=xi_temp+gamma_temp
        T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        #GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(T2_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_T2,'black',ls='--')
     
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks([0.95,1.00,1.05,1.1,1.15],fontsize=12)
    plt.ylim([0.95,1.15])
    plt.savefig('../code-paper-v2/fig/monitor_21.png',doi="300",bbox_inches='tight')
        
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-0.04,-0.02,0,0.02,0.04]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC*(1+change_delta), scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC*(1+change_delta))**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        residual_layer=xi_temp+gamma_temp
        #T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(GLR_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_GLR,'black',ls='--')
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0,25])
    plt.savefig('../code-paper-v2/fig/monitor_22.png',doi="300",bbox_inches='tight')    
    
    
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-0.04,-0.02,0,0.02,0.04]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1+change_delta)*(1-alpha_IC**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        residual_layer=xi_temp+gamma_temp
        T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        #GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(T2_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_T2,'black',ls='--')
     
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks([0.95,1.00,1.05,1.1,1.15],fontsize=12)
    plt.ylim([0.95,1.15])
    plt.savefig('../code-paper-v2/fig/monitor_31.png',doi="300",bbox_inches='tight')
        
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [-0.04,-0.02,0,0.02,0.04]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1+change_delta)*(1-alpha_IC**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        residual_layer=xi_temp+gamma_temp
        #T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(GLR_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_GLR,'black',ls='--')
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0,25])
    plt.savefig('../code-paper-v2/fig/monitor_32.png',doi="300",bbox_inches='tight')    
    
    
    
    
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [0,0.1,0.15,0.2]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC)**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        x_layer=generate_x_IC(n_variable,n_time)
        beta_1 = np.random.uniform(-1,1,n_variable)
        g_delta=np.dot(x_layer,beta_1)
        residual_layer=xi_temp+gamma_temp+g_delta*change_delta
        T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        #GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(T2_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_T2,'black',ls='--')
     
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks([0.95,1.00,1.05,1.1,1.15],fontsize=12)
    plt.ylim([0.95,1.15])
    plt.savefig('../code-paper-v2/fig/monitor_41.png',doi="300",bbox_inches='tight')
        
    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(4.5,4.5))
    for change_delta in [0,0.1,0.15,0.2]:
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(904)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-(alpha_IC)**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=0.4
        x_layer=generate_x_IC(n_variable,n_time)
        beta_1 = np.random.uniform(-1,1,n_variable)
        g_delta=np.dot(x_layer,beta_1)
        residual_layer=xi_temp+gamma_temp+g_delta*change_delta
        #T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        
        plt.plot(GLR_layer,label=r'$\delta$=%.2f'%(change_delta))

    plt.plot(up_bound_GLR,'black',ls='--')
    plt.legend(loc='upper right',fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0,25])
    plt.savefig('../code-paper-v2/fig/monitor_42.png',doi="300",bbox_inches='tight')       
    
    '''
    import time
    time1=[]
    time2=[]
    for j in range(100):
        print(j)
        lambda_a = 1/(1-alpha_IC**2)
        lambda_b = -alpha_IC/(1-alpha_IC**2)
        lambda_c = (1+alpha_IC**2)/(1-alpha_IC**2)  
        np.random.seed(i)
        gamma_temp=np.zeros(n_time)
        gamma_previous=0
        for i in range(n_time):
            gamma_temp[i]=np.random.normal(loc=gamma_previous*alpha_IC, scale=np.sqrt(sigma_gamma_IC*(1-alpha_IC**2)))
            gamma_previous=gamma_temp[i]    
        xi_temp=np.random.normal()*np.sqrt(sigma_xi_IC)
        residual_layer=xi_temp+gamma_temp
        time_begin=time.time()
        T2_layer=cal_T2(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        time1.extend([time.time()-time_begin])
        time_begin=time.time()
        GLR_layer=cal_GLR(residual_layer,0,sigma_xi_IC,sigma_gamma_IC,lambda_a,lambda_b,lambda_c)
        time2.extend([time.time()-time_begin])
        
    np.mean(time1)
    np.mean(time2)
    '''
