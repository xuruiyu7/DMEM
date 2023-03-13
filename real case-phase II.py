import numpy as np
import pandas as pd
import h5py
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import statsmodels.tsa.api as smt
from scipy import stats



os.chdir('D:/Ruiyu/坚果云/徐瑞宇/3d打印变点/data')

#%%

layer_num=379
data_total=[]

layer_list = [n for n in range(1,layer_num+1)]
columns=['X','Y','NominalPower','NominalSpeed','NominalSpotDiameter','LaserPowerCurrent','SignalInGaAs','BulkLayer','IDoocLayer','c_1_cost','c_2_cost']
for layer_temp in layer_list:
    layer_name='1/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

def distance_point(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

model = tf.keras.models.load_model('../code-paper-v2/checkpoints_AM_EXP_1')
data_layer=data_total[0].values
x_min=np.min(data_layer[:,0])
x_max=np.max(data_layer[:,0])
y_min=np.min(data_layer[:,1])
y_max=np.max(data_layer[:,1])

def predict_temperature(model,data_layer,x_min,x_max,y_min,y_max):    
    #model.summary()
    
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
            if dis_point[index_path_change[j]]>1:
                #print(index_path_change[j],dis_point[index_path_change[j]])
                num_seg_temp=1
            else:
                num_seg_temp+=1
        num_seg[index_path_change[j]:index_path_change[j+1]]=num_seg_temp
        
    x_layer=np.concatenate((previous_power,distance,last_seg_length,length_present_seg,dis_point,num_seg),1)
    x_layer=np.array(x_layer,np.float32)
                    
    y_predict_layer=model.predict(x_layer)
    
    return y_predict_layer
#%%
index_unexposed_layer=[]
index_anomaly_layer=[26,27,28]
index_normal_layer=[]
for layer_temp in range(379):
    data_layer=data_total[layer_temp].values
    if len(data_layer)<15000:
        index_unexposed_layer.extend([layer_temp])
    elif np.max(data_layer[:,8])>0:
        index_anomaly_layer.extend([layer_temp])
    else:
        index_normal_layer.extend([layer_temp])
index_normal_layer.remove(26)      
index_normal_layer.remove(27)      
index_normal_layer.remove(28)      

#%%
data_layer=data_total[0].values
index_path_change=np.where(data_layer[:,5]<4000)[0]
index_path_change_list=[index_path_change[0]]
for i in range(len(index_path_change)-1):
    if index_path_change[i+1]-index_path_change[i]==1:
        continue
    index_path_change_list.extend([index_path_change[i+1]])
index_path_change=np.array(index_path_change_list)

plt.rc('font',family='Times New Roman')

plt.figure(figsize=(6,4.5))
plt.plot(data_layer[:3000,6],'#1A6FDF')
#plt.vlines(index_path_change[:11],200,1900,'black','--')
plt.ylim([150,1900])
#plt.xlim([0,3000])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Sample index',fontsize=12)
plt.ylabel('Temperature',fontsize=12)
#plt.savefig('../code-paper/fig/signal.png',doi="300",bbox_inches='tight')

plt.figure(figsize=(6,4.5))
data_layer=data_total[0].values
plt.plot(data_layer[:,6],'#1A6FDF')
data_layer=data_total[100].values
#plt.plot(data_layer[:,6],'black')
data_layer=data_total[200].values
#plt.plot(data_layer[:,6],'black')
#plt.vlines(index_path_change[:11],200,1900,'black','--')
plt.ylim([150,2000])
#plt.xlim([0,3000])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Sample index',fontsize=12)
plt.ylabel('Temperature',fontsize=12)
#plt.savefig('../code-paper/fig/signal.png',doi="300",bbox_inches='tight')

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
y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
y_pred = y_pred.reshape(-1)
#%%
plt.figure(figsize=(6,4.5))
plt.plot(data_layer[:3000,6],'#1A6FDF')
plt.plot(y_pred[:3000],'orange')
#plt.vlines(index_path_change[:11],200,1900,'black','--')
plt.ylim([150,1900])
#plt.xlim([0,3000])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Sample index',fontsize=12)
plt.ylabel('Temperature',fontsize=12)
#plt.savefig('../code-paper/fig/signal.png',doi="300",bbox_inches='tight')

plt.figure(figsize=(6,4.5))
plt.plot(data_layer[:3000,6]-y_pred[:3000],'#1A6FDF')
#plt.vlines(index_path_change[:11],200,1900,'black','--')
#plt.ylim([150,1900])
#plt.xlim([0,3000])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Sample index',fontsize=12)
plt.ylabel('Residuals',fontsize=12)
#plt.savefig('../code-paper/fig/signal.png',doi="300",bbox_inches='tight')

#%%
alpha_list=[]
xi_list=[]
sigma_gamma_list=[]
for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    gamma_temp=data_layer[:,6]-y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_temp[change_index]=0
    alpha_list.append(smt.stattools.acf(gamma_temp, nlags=10)[1])
    xi_list.append(np.mean(gamma_temp))
    sigma_gamma_list.append(np.var(gamma_temp))

print(np.mean(alpha_list))
print(np.mean(sigma_gamma_list))

alpha=np.mean(alpha_list)
sigma_gamma=np.mean(sigma_gamma_list)
sigma_xi=np.var(xi_list)

lambda_a = 1/(1-alpha**2)
lambda_b = -alpha/(1-alpha**2)
lambda_c = (1+alpha**2)/(1-alpha**2)


#%%
T2_layer_good_list = []
for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_good_list.append(log_diff_list_layer)
    
#%% 
coef_T2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2.append(np.min(coef_temp[5000:]))

coef_T2_2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2_2.append(np.max(coef_temp[5000:]))


#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    plt.plot(T2_list_layer,'#1A6FDF',lw=1)
    plt.ylim([0,2])
  
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)

#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=16.3
alpha_error=1-stats.norm.cdf(lambda1_T2)
change_index_T2=3000
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1

  
''' 
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] = 120/ (np.sqrt(i+1)-0.6)+0.5
lower_bound = np.zeros(32000)
for i in range(32000):
    lower_bound[i] = -20/ (np.sqrt(i+1)-0.95)+0.93
'''
plt.ylim([0,2])
plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#1A6FDF',lw=1,label='IC layers')
plt.legend(loc='upper right',fontsize=10) 

plt.savefig('../code-paper-v2/fig/AM_Phase_II_1.png',doi="300",bbox_inches='tight')
upper_bound[:1000]=1000
lower_bound[:1000]=-1000
#%%
num_1=0
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>upper_bound[j]) or (T2_list_layer[j]<lower_bound[j]):
            print(i,j)
            num_1+=1
            break

#%%
T2_layer_bad_list = []
for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_bad_list.append(log_diff_list_layer)

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_bad_list)):
    T2_list_layer=T2_layer_bad_list[i]
    plt.plot(T2_list_layer,'#B22222',lw=1)
    plt.ylim([0,2])
    
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)
#alpha_error=0.00034
#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=16.3
alpha_error=1-stats.norm.cdf(lambda1_T2)
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1



plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#B22222',lw=1,label='OC layers')

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_2.png',doi="300",bbox_inches='tight')

upper_bound[:1000]=1000
lower_bound[:1000]=-1000


#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(T2_layer_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=T2_layer_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]) or (log_diff_list_layer[j]<lower_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))



#%%
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)
    log_diff_good_list.extend([log_diff_list_layer])
#%%
coef_GLR=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_GLR.append(np.max(log_diff_list_layer))

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =150
plt.plot(log_diff_list_layer,'#1A6FDF',lw=1,label='IC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_3.png',doi="300",bbox_inches='tight')

#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)

    log_diff_bad_list.extend([log_diff_list_layer])
#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_bad_list)):
    plt.plot(log_diff_bad_list[i],'#B22222',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =150
plt.plot(log_diff_list_layer,'#B22222',lw=1,label='OC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_4.png',doi="300",bbox_inches='tight')

#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))


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
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_good_list.extend([log_diff_list_layer])

#%%
coef_EWMA=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_EWMA.append(np.max(log_diff_list_layer))

#%%

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>57):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))


#%%
def cal_CUSUM(gamma,mean_xi,sigma_xi,sigma_gamma,minmiss):
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
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_good_list.extend([log_diff_list_layer])


for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%

coef_CUSUM=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_CUSUM.append(np.max(log_diff_list_layer))

#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#plt.ylim([0,100000])

#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>180000):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))



#%%
layer_num=379
data_total=[]

layer_list = [n for n in range(1,layer_num+1)]
columns=['X','Y','NominalPower','NominalSpeed','NominalSpotDiameter','LaserPowerCurrent','SignalInGaAs','BulkLayer','IDoocLayer','c_1_cost','c_2_cost']
for layer_temp in layer_list:
    layer_name='2/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)
    
#%%
index_unexposed_layer_test=[]
index_anomaly_layer_test=[26,27,28]
index_normal_layer_test=[]
for layer_temp in range(379):
    data_layer=data_total[layer_temp].values
    if len(data_layer)<15000:
        index_unexposed_layer_test.extend([layer_temp])
    elif np.max(data_layer[:,8])>0:
        index_anomaly_layer_test.extend([layer_temp])
    else:
        index_normal_layer_test.extend([layer_temp])
index_normal_layer.remove(26)      
index_normal_layer.remove(27)      
index_normal_layer.remove(28)      

#%%
T2_layer_good_list = []
for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_good_list.append(log_diff_list_layer)

#%% 
coef_T2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2.append(np.min(coef_temp[5000:]))

coef_T2_2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2_2.append(np.max(coef_temp[5000:]))


#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    plt.plot(T2_list_layer,'#1A6FDF',lw=1)
    plt.ylim([0,2])
  
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)
#alpha_error=0.00034
#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=17
alpha_error=1-stats.norm.cdf(lambda1_T2)
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1

  
''' 
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] = 120/ (np.sqrt(i+1)-0.6)+0.5
lower_bound = np.zeros(32000)
for i in range(32000):
    lower_bound[i] = -20/ (np.sqrt(i+1)-0.95)+0.93
'''
plt.ylim([0,2])
plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#1A6FDF',lw=1,label='IC layers')
plt.legend(loc='upper right',fontsize=10) 

plt.savefig('../code-paper-v2/fig/AM_Phase_II_5.png',doi="300",bbox_inches='tight')
upper_bound[:1000]=1000
lower_bound[:1000]=-1000
#%%
num_1=0
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>upper_bound[j]) or (T2_list_layer[j]<lower_bound[j]):
            print(i,j)
            num_1+=1
            break

#%%
T2_layer_bad_list = []
for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_bad_list.append(log_diff_list_layer)

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_bad_list)):
    T2_list_layer=T2_layer_bad_list[i]
    plt.plot(T2_list_layer,'#B22222',lw=1)
    plt.ylim([0,2])
    
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)
#alpha_error=0.00034
#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=17
alpha_error=1-stats.norm.cdf(lambda1_T2)
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1



plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#B22222',lw=1,label='OC layers')

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_6.png',doi="300",bbox_inches='tight')

upper_bound[:1000]=1000
lower_bound[:1000]=-1000


#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(T2_layer_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=T2_layer_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]) or (log_diff_list_layer[j]<lower_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))



#%%
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)
    log_diff_good_list.extend([log_diff_list_layer])

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =175
plt.plot(log_diff_list_layer,'#1A6FDF',lw=1,label='IC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_7.png',doi="300",bbox_inches='tight')

#%%
num_1=0
for i in range(len(log_diff_good_list)):
    T2_list_layer=log_diff_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>upper_bound[j]):
            print(i,j)
            num_1+=1
            break

#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)

    log_diff_bad_list.extend([log_diff_list_layer])
#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_bad_list)):
    plt.plot(log_diff_bad_list[i],'#B22222',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =175
plt.plot(log_diff_list_layer,'#B22222',lw=1,label='OC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/AM_Phase_II_8.png',doi="300",bbox_inches='tight')

#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))



#%%
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_good_list.extend([log_diff_list_layer])

#%%
num_1=0
for i in range(len(log_diff_good_list)):
    T2_list_layer=log_diff_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>50):
            print(i,j)
            num_1+=1
            break

#%%

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>50):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))

#%%
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_good_list.extend([log_diff_list_layer])


for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%
num_1=0
for i in range(len(log_diff_good_list)):
    T2_list_layer=log_diff_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>180000):
            print(i,j)
            num_1+=1
            break

#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#plt.ylim([0,100000])

#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>180000):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))







#%%
layer_num=379
data_total=[]

layer_list = [n for n in range(1,layer_num+1)]
columns=['X','Y','NominalPower','NominalSpeed','NominalSpotDiameter','LaserPowerCurrent','SignalInGaAs','BulkLayer','IDoocLayer','c_1_cost','c_2_cost']
for layer_temp in layer_list:
    layer_name='1/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)

for layer_temp in layer_list:
    layer_name='2/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)



#%%
index_unexposed_layer=[]
index_anomaly_layer=[26,27,28,405,406,407]
index_normal_layer=[]
for layer_temp in range(len(data_total)):
    data_layer=data_total[layer_temp].values
    if len(data_layer)<15000:
        index_unexposed_layer.extend([layer_temp])
    elif np.max(data_layer[:,8])>0:
        index_anomaly_layer.extend([layer_temp])
    else:
        index_normal_layer.extend([layer_temp])
index_normal_layer.remove(26)      
index_normal_layer.remove(27)      
index_normal_layer.remove(28)      

index_normal_layer.remove(405)      
index_normal_layer.remove(406)      
index_normal_layer.remove(407)      


#%%
T2_layer_good_list = []
for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_good_list.append(log_diff_list_layer)
    
#%% 
coef_T2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2.append(np.min(coef_temp[5000:]))

coef_T2_2=[]
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    coef_temp=np.zeros(len(T2_list_layer))
    for j in range(len(T2_list_layer)):
        coef_temp[j]=(T2_list_layer[j]-1)/np.sqrt(2/(j+1))
    coef_T2_2.append(np.max(coef_temp[5000:]))


#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    plt.plot(T2_list_layer,'#1A6FDF',lw=1)
    plt.ylim([0,2])
  
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)

#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=16.33
alpha_error=1-stats.norm.cdf(lambda1_T2)
change_index_T2=3000
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1

  
''' 
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] = 120/ (np.sqrt(i+1)-0.6)+0.5
lower_bound = np.zeros(32000)
for i in range(32000):
    lower_bound[i] = -20/ (np.sqrt(i+1)-0.95)+0.93
'''
plt.ylim([0,2])
plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#1A6FDF',lw=1,label='IC layers')
plt.legend(loc='upper right',fontsize=10) 


upper_bound[:1000]=1000
lower_bound[:1000]=-1000
#%%
num_1=0
for i in range(len(T2_layer_good_list)):
    T2_list_layer=T2_layer_good_list[i]
    for j in range(len(T2_list_layer)):
        if (T2_list_layer[j]>upper_bound[j]) or (T2_list_layer[j]<lower_bound[j]):
            print(i,j)
            num_1+=1
            break

#%%
T2_layer_bad_list = []
for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_T2(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)    
    #plt.show()
    T2_layer_bad_list.append(log_diff_list_layer)

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   
for i in range(len(T2_layer_bad_list)):
    T2_list_layer=T2_layer_bad_list[i]
    plt.plot(T2_list_layer,'#B22222',lw=1)
    plt.ylim([0,2])
    
upper_bound=np.zeros(32000)
lower_bound=np.zeros(32000)
#alpha_error=0.00034
#1-stats.norm.cdf(lambda1_T2)=0.0001487478574404566

lambda1_T2=16.33
alpha_error=1-stats.norm.cdf(lambda1_T2)
for i in range(32000):
    upper_bound[i]=lambda1_T2*np.sqrt(2/(i+1))+1
for i in range(32000):
    lower_bound[i]=-lambda1_T2*np.sqrt(2/(i+1))+1



plt.plot(upper_bound,color='black',ls='--')
plt.plot(lower_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(T2_list_layer,'#B22222',lw=1,label='OC layers')

plt.legend(loc='upper right',fontsize=10)


upper_bound[:1000]=1000
lower_bound[:1000]=-1000


#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(T2_layer_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=T2_layer_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]) or (log_diff_list_layer[j]<lower_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))



#%%
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)
    log_diff_good_list.extend([log_diff_list_layer])
#%%
coef_GLR=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_GLR.append(np.max(log_diff_list_layer))

#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =155
plt.plot(log_diff_list_layer,'#1A6FDF',lw=1,label='IC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)


#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_GLR(gamma_layer,0,sigma_xi,sigma_gamma,lambda_a,lambda_b,lambda_c)

    log_diff_bad_list.extend([log_diff_list_layer])
#%%
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,4.5))   

for i in range(len(log_diff_bad_list)):
    plt.plot(log_diff_bad_list[i],'#B22222',lw=1)

plt.ylim([0,500])
    
upper_bound = np.zeros(32000)
for i in range(32000):
    upper_bound[i] =155
plt.plot(log_diff_list_layer,'#B22222',lw=1,label='OC layers')

plt.plot(upper_bound,color='black',ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='upper right',fontsize=10)


#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>upper_bound[j]):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))


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
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_good_list.extend([log_diff_list_layer])

#%%
coef_EWMA=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_EWMA.append(np.max(log_diff_list_layer))

#%%

for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_EWMA(gamma_layer,0,sigma_xi,sigma_gamma)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>55):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))


#%%
def cal_CUSUM(gamma,mean_xi,sigma_xi,sigma_gamma,minmiss):
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
log_diff_good_list = []

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_good_list.extend([log_diff_list_layer])


for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)


#%%

coef_CUSUM=[]
for i in range(len(log_diff_good_list)):
    log_diff_list_layer=log_diff_good_list[i]
    coef_CUSUM.append(np.max(log_diff_list_layer))

#%%
log_diff_bad_list = []

for layer_temp in index_anomaly_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    log_diff_list_layer = cal_CUSUM(gamma_layer,0,sigma_xi,sigma_gamma,100)
    log_diff_bad_list.extend([log_diff_list_layer])


for i in range(len(log_diff_bad_list)):
    log_diff_list_layer=log_diff_bad_list[i]
    plt.plot(log_diff_list_layer,'#1A6FDF',lw=1)
    
#plt.ylim([0,100000])

#%%
time_delay_list=[]
time_delay_list_2=[]
for i in range(len(log_diff_bad_list)):
    if i%3==0:
        print("%d-th anomaly"%(int(i/3+1)))
        num_block=0
        is_new=1
    log_diff_list_layer=log_diff_bad_list[i]
    for j in range(len(log_diff_list_layer)):
        if (log_diff_list_layer[j]>180000):
            print("layer: %d, detection time: %d"%(i+1,j))
            time_delay_list.extend([j+1])
            num_block+=j+1
            if is_new==1:
                time_delay_list_2.extend([num_block])
                is_new=0
            break
    if len(time_delay_list)<i+1:
        time_delay_list.extend([len(log_diff_list_layer)])
    
    if is_new==1:
        num_block+=j+1
        if i%3==2:
            time_delay_list_2.extend([num_block])
print(np.mean(time_delay_list))
print(np.mean(time_delay_list_2))





#%%
plt.rc('font',family='Times New Roman')

layer_plot_list=[0,92,197,311]
for layer_temp in layer_plot_list:    
    data_layer = data_total[layer_temp].values
    plt.figure(figsize=(4,3))
    plt.plot(data_layer[:,6],color='#1A6FDF',lw=1)
    plt.ylim([0,2500])
    plt.xticks([0,10000,20000,30000],fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y',ls='--')
    plt.savefig('../code-paper-v2/fig/AM_traces_%d.png'%(layer_temp),doi="300",bbox_inches='tight')


#%%
xi_test_list=[]

layer_num=379
data_total=[]
layer_list = [n for n in range(1,layer_num+1)]
columns=['X','Y','NominalPower','NominalSpeed','NominalSpotDiameter','LaserPowerCurrent','SignalInGaAs','BulkLayer','IDoocLayer','c_1_cost','c_2_cost']
for layer_temp in layer_list:
    layer_name='1/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    xi_test_list.extend([np.mean(gamma_layer)])

#%%
xi_test_list=np.array(xi_test_list)-np.mean(xi_test_list)

from scipy import stats

plt.rc('font',family='Times New Roman')
normalDistribution=stats.norm(0,np.sqrt(sigma_xi))
x=np.arange(-35,40,0.1)
y=normalDistribution.pdf(x)

plt.figure(figsize=(6,4.5))
plt.hist(xi_test_list,bins=20,density=True,color='gray',edgecolor='white',label=r'Estimated $\xi_i$')
plt.plot(x,y,label='Estimated distribution',color='orange',linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([-35,40])
plt.ylim([0,0.040])
plt.legend(loc='upper right',fontsize=10)

plt.savefig('../code-paper-v2/fig/xi_AM_1.png',doi="300",bbox_inches='tight')


#%%

import statsmodels.api as sm

sm.qqplot(np.array(xi_test_list), line='45',scale=np.sqrt(sigma_xi),marker='.', markerfacecolor='gray', markeredgecolor='k',markeredgewidth=.5,markersize=14)
ax = plt.gca()
ax.grid(linestyle="--",linewidth=.5,color="black")
ax.set_aspect(1)
ax.set_xlim([-40,40])
ax.set_ylim([-40,40])
ax.set_yticks([-40,-20,0,20,40])
plt.savefig('../code-paper-v2/fig/xi_AM_1_qq.png',doi="300",bbox_inches='tight')

#%%
xi_test_list2=[]

layer_num=379
data_total=[]
layer_list = [n for n in range(1,layer_num+1)]
columns=['X','Y','NominalPower','NominalSpeed','NominalSpotDiameter','LaserPowerCurrent','SignalInGaAs','BulkLayer','IDoocLayer','c_1_cost','c_2_cost']
for layer_temp in layer_list:
    layer_name='2/layer'+str(layer_temp)+'.hdf5'
    sample_dataset_object = h5py.File(layer_name, 'r')
    sample_dataset = sample_dataset_object['OpenData'][:]
    sample_dataset = sample_dataset.transpose()
    sample_dataset = pd.DataFrame(sample_dataset,columns=columns)

    data_total.append(sample_dataset)

for layer_temp in index_normal_layer:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true = data_layer[:,6]
    gamma_layer = y_true - y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    gamma_layer[change_index]=0
    xi_test_list2.extend([np.mean(gamma_layer)])

#%%
#xi_test_list2=xi_test_list
xi_test_list2=np.array(xi_test_list2)-np.mean(xi_test_list2)
from scipy import stats



plt.rc('font',family='Times New Roman')
normalDistribution=stats.norm(0,np.sqrt(sigma_xi))
x=np.arange(-35,40,0.1)
y=normalDistribution.pdf(x)

plt.figure(figsize=(6,4.5))
plt.hist(xi_test_list2,bins=20,density=True,color='gray',edgecolor='white',label=r'Estimated $\xi_i$')
plt.plot(x,y,label='Estimated distribution',color='orange',linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([-35,40])
plt.ylim([0,0.040])
plt.legend(loc='upper right',fontsize=10)
plt.savefig('../code-paper-v2/fig/xi_AM_2.png',doi="300",bbox_inches='tight')


#%%

import statsmodels.api as sm

sm.qqplot(np.array(xi_test_list2), line='45',scale=np.sqrt(sigma_xi),marker='.', markerfacecolor='gray', markeredgecolor='k',markeredgewidth=.5,markersize=14)
ax = plt.gca()
ax.grid(linestyle="--",linewidth=.5,color="black")
ax.set_aspect(1)
ax.set_xlim([-40,40])
ax.set_ylim([-40,40])
ax.set_yticks([-40,-20,0,20,40])
plt.savefig('../code-paper-v2/fig/xi_AM_2_qq.png',doi="300",bbox_inches='tight')





#%%

for layer_temp in [0]:
    print(layer_temp)
    data_layer = data_total[layer_temp].values
    y_pred = predict_temperature(model,data_layer,x_min,x_max,y_min,y_max)
    y_true=data_layer[:,6]
    y_pred=y_pred.reshape(-1)
    change_index=np.where(data_layer[:,5]<4000)
    y_true=np.delete(y_true,change_index)
    y_pred=np.delete(y_pred,change_index)
    
    plt.figure(figsize=(4,3))
    plt.plot(y_true[:1000],color='#1A6FDF',lw=1,label='Observed signals')
    plt.plot(y_pred[:1000],color='orange',lw=1,label='Estimated g(x)')
    plt.ylim([500,2500])
    plt.xticks(fontsize=12)
    plt.yticks([500,1000,1500,2000,2500],fontsize=12)
    plt.grid(axis='y',ls='--')
    plt.legend(loc='upper right',fontsize=10)
    plt.xlabel('Sample index',fontsize=12)
    plt.ylabel('Temperature',fontsize=12)
    plt.savefig('../code-paper-v2/fig/AM_traces_2_%d.png'%(layer_temp),doi="300",bbox_inches='tight')


    gamma_temp=y_true-y_pred
    
    import statsmodels.api as sm
    fig = plt.figure(figsize=(4,3))  

    """ACF"""
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(gamma_temp, lags=20,ax=ax1,color='#1A6FDF')
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    ax1.set_ylim(0,1.1)
    """PACF"""
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(gamma_temp, lags=20, ax=ax2,color='#1A6FDF')
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    ax2.set_ylim(0,1.1)
    plt.savefig('../code-paper-v2/fig/AM_traces_3_%d.png'%(layer_temp),doi="300",bbox_inches='tight')
    


