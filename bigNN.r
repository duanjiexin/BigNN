
###############################################################
#  Simulation 1: Regret and CIS for bigNN and oracle kNN for k with theorem setting in paper  
#  Simulation 2: Regret and CIS for bigNN and oracle kNN for fixed k=5   
#  By Jiexin Duan, 
#  All rights reserved  
#  Department of Statistics Purdue University   10/24/2019
###############################################################

library(snn)

##################################################################
#  Setup of parameters
##################################################################

# setup of parameters
# original N for simulation
NO = 1000  # 1000,2000,4000,8000,12000,16000,20000,32000  
# gamma 
gamma = 0.2  # 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
# get approximate s and n to distributed calculation
if(((floor(NO^gamma))%%2)==1) s=floor(NO^gamma) else s=ceiling(NO^gamma) #majority voting ensemble times, nearest odd number
if(NO-s*floor(NO/s)<s*ceiling(NO/s)-NO)   n=floor(NO/s)  else n=ceiling(NO/s) #find floor(N/s),ceiling(N/s) which is nearer to N
# new N after DaC
N=s*n  
# set up distribution of simulated data
d=5; mu=1; portion=1/2   

# number of observation in test data
ntest=1000

# choose oracle
threshold=1   # threshold in the paper
k0=1   #constant in the paper
k_O = as.integer(k0*N^((2*threshold)/(2*threshold+1))) # k for oracle NN

# simulation 1 for k with theorem setting in paper  
k_temp=(k0*N^((2*threshold)/(2*threshold+1))/s)  # k_temp for knn
if((floor(k_temp)%%2)==1) k=floor(k_temp) else k=ceiling(k_temp)
k=min(k,n)   #chose the smaller of k and n

# simulated 2 for fixed k=5
# k=5

# voting threshold for majority voting
threshold = 0.5 


#########################################################

##estimate Bayes risk 
library(mnormt) 
NN=10000000  
prop = portion
n1=floor(prop*NN)   
n2=NN-n1        

xx1 = matrix(rnorm(n1*d),n1,d)   
xx2 = matrix(rnorm(n2*d),n2,d) + mu  
myx = rbind(xx1,xx2)  
eta=function(x){    
  f1=dmnorm(x,rep(0,d), diag(d),log=FALSE) 
  f2=dmnorm(x,rep(mu,d), diag(d),log=FALSE)
  prop*f1/(prop*f1+(1-prop)*f2)  
}
obj = function(x){
  ifelse(eta(x)>1/2,1-eta(x),eta(x))
}
risk_Bayes = mean(obj(myx)) 
risk_Bayes
rm(myx)


###################################################################################
# Oracle Training data for classifier \phi_1 and \phi_2, we generate 2N observations
# DATA1 is used for both CIS and Regret, DATA2 is used for CIS
####################################################################################

# generate the oracle data
DATA_oracle1 = mydata(N,d,mu=mu,portion=portion)
DATA_oracle2 = mydata(N,d,mu=mu,portion=portion)

# genelize testing dataset
TEST = mydata(ntest,d,mu=mu,portion=portion)

#initialize prediect.sum list to store sum of prediction before majority voting
predict1.sum = rep(0,ntest)
predict2.sum = rep(0,ntest)

####################################################
#  Divide & Conquer Process
####################################################
time_dac_divide = 0

# Step 1: Dividing process, shuffle the index
permIndex_1 = sample(nrow(DATA_oracle1))
DATA_oracle1 = DATA_oracle1[permIndex_1,]
permIndex_2 = sample(nrow(DATA_oracle2))
DATA_oracle2 = DATA_oracle2[permIndex_2,]

# Step 2: run kNN on subsets
index = 1
for(j in 1:s){
  time_dac_temp0 = Sys.time()
  
  # Training data for classifier \phi_1 and \phi_2 in CIS, we generate 2n observations
  DATA1 = DATA_oracle1[index:(index+n-1),]
  DATA2 = DATA_oracle2[index:(index+n-1),]

  #get preditive value from trained classfier on testing data
  predict1 = myknn(DATA1, TEST[,1:d], k)
  predict2 = myknn(DATA2, TEST[,1:d], k)

  #sum all predict for majority voting
  predict1.sum = predict1.sum + predict1
  predict2.sum = predict2.sum + predict2

  #moving index to beginning of next subset
  index = index + n
  
  # calculated the longest time of each subsets
  time_dac_temp1 = Sys.time()
  time_dac_temp = time_dac_temp1 - time_dac_temp0
  time_dac_temp = round(as.numeric(time_dac_temp, units = "secs"),digits=2) 
  if(time_dac_temp >= time_dac_divide) time_dac_divide = time_dac_temp 
  
}
  
# Step3: Majority voting process 
time_dac_combine0 = Sys.time()

# Majority voting for results from all subsets
predict1.mv = ifelse(s^(-1)*predict1.sum > 2-threshold, 2, 1)
predict2.mv = ifelse(s^(-1)*predict2.sum > 2-threshold, 2, 1)

# Store CIS value for bigNN
cis_big = mycis(predict1.mv, predict2.mv)

# Store risk value for bigNN
risk_big = myerror(predict1.mv, TEST[,d+1])

# calculate combing time
time_dac_combine1 = Sys.time()
time_dac_combine = time_dac_combine1 - time_dac_combine0
time_dac_combine = round(as.numeric(time_dac_combine, units="secs"),digits=2) 
time_dac = time_dac_divide + time_dac_combine 


#####################################################################################
time_oracle0 = Sys.time()    #start to record oracle time

# Calculate oracle prediction 
predict_oracle_1 = myknn(DATA_oracle1, TEST[,1:d], k_O)
predict_oracle_2 = myknn(DATA_oracle2, TEST[,1:d], k_O)

# Store CIS value for oracle kNN
cis_oracle = mycis(predict_oracle_1, predict_oracle_2)

# Store regret and risk value for oracle kNN
risk_oracle = myerror(predict_oracle_1, TEST[,d+1])

# record oracle kNN time
time_oracle1 = Sys.time()  
time_oracle = time_oracle1 - time_oracle0    
time_oracle = round(as.numeric(time_oracle,units="secs"),digits=3)    

#Print results together
print(paste(NO,gamma,N,s,n,d,portion,mu,k,k_O,cis_big,cis_oracle,risk_big,risk_oracle,risk_Bayes,time_dac,time_oracle))     









