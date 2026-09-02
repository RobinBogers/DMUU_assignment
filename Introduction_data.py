import numpy as np
import scipy.stats as stats


# Given data
p_i=np.array([0.05,0.25,0.15,0.10,0.20,0.08,0.12,0.04,0.17,0.22])
y_i=np.array([0.80,0.40,0.50,0.70,0.40,0.60,0.75,0.90,0.65,0.85])
c_i=np.array([40,25,35,30,21,55,45,37,52,42])
c_i_eff=np.array([39.6,21.3,32.4,29.1,18.5,53.2,43.7,36.9,48.9,40.6])
CR_i=np.array([0.40,0.70,0.50,0.60,0.78,0.10,0.30,0.46,0.16,0.36])
Exp_R_i=np.array([0.99,0.85,0.93,0.97,0.88,0.97,0.97,1.00,0.94,0.97])
Var_R_i=np.array([0.00,0.07,0.03,0.01,0.06,0.01,0.01,0.00,0.02,0.00])

#Initial parameters
k=15
s=45
r=10
group_number=3

#Seed generation
L=range(1,11)
seeds=[1000*group_number+l for l in L]

#New supplier data for 10 instances
for i in range(10):
    np.random.seed(seeds[i])

    p_i_new=np.random.uniform(np.maximum(0.02,p_i-0.05),np.minimum(0.35,p_i+0.05))
    y_i_new=np.random.uniform(np.maximum(0.30,y_i-0.15),np.minimum(0.95,y_i+0.15))
    c_i_new=np.random.uniform(np.maximum(r+1,0.90*c_i),np.minimum(s+k-1,1.10*c_i))

    #Demand parameters
    mu=np.random.uniform(350,450)
    #Coefficient of variation
    v=np.random.uniform(0.20,0.29)
    sigma=v*mu

    #Supplier specific quantities
    Exp_R_i_new=1-p_i_new*(1-y_i_new)
    Var_R_i_new=p_i_new*(1-p_i_new)*(1-y_i_new)**2
    c_i_eff_new=c_i_new*Exp_R_i_new
    CR_i_new=(s+k-c_i_new)/(s+k-r)

    #Tabulation of new data
