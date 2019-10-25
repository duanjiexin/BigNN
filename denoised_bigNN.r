
###############################################################
#  Simulation 3: Regret and CIS for denoised bigNN 
#  By Jiexin Duan, 
#  All rights reserved  
#  Department of Statistics Purdue University   10/24/2019
###############################################################

library(snn)
library(MASS)

# parameter of DDNN
# number of selected subsamples (I) in denoised algorithm
N_sub = 5 # 5,9,13,17,21
# size exponent of the selected subsample to denoise 
# eg. theta=0.5 means subset of size N^theta is used to find 1NN of x
theta = 0.2 # 0.1,0.2,0.3,0.4,0.5,0.6,0.7  

# parameters of data
# dimension
d = 8 
# portion of first class (mu=0)
portion = 1/3 
# number of observation in test data
ntest = 1000 
# original N for simulation
NO = 27000 
# gamma 
gamma = 0.2 # 0.2,0.3  
# get approximate s and n to distributed calculation
if(((floor(NO^gamma))%%2)==1) s=floor(NO^gamma) else s=ceiling(NO^gamma) #majority voting ensemble times, nearest odd number
if(NO-s*floor(NO/s)<s*ceiling(NO/s)-NO)   n=floor(NO/s)  else n=ceiling(NO/s) #find floor(N/s),ceiling(N/s) which is nearer to N
# new N after DaC
N=s*n   #size for the oracle Big Data

threshold =0.5 

# choose oracle k
k_O = N^0.7
if((floor(k_O)%%2)==1) k_O=floor(k_O) else k_O=ceiling(k_O)
k_O = min(N,k_O)  #make sure K_O <= N

# oracle kNN case
if(gamma==0){ 
  k=k_O/s
}

# bigNN case
k0 = 1.351284
if(gamma!=0){  
  k=k0*k_O/s
}
if((floor(k)%%2)==1) k=floor(k) else k=ceiling(k)
k=min(k,n) 


###########################################################
#  Calculation of Bayes Risk by Monte Carlo Simulation
###########################################################
library(mnormt)   
NN=round(1000000/d) 
prop = portion
n1=floor(prop*NN)   
n2=NN-n1        

myx = rbind(mvrnorm(floor(n1/2), rep(0,d), diag(d)),mvrnorm(n1 - floor(n1/2), rep(3,d), 2*diag(d)),mvrnorm(floor(n2/2), rep(1.5,d), diag(d)),mvrnorm(n2 - floor(n2/2), rep(4.5,d), 2*diag(d)))
eta=function(x){      
  f1=log(0.5*dmnorm(x,rep(0,d), diag(d),log=FALSE)+0.5*dmnorm(x,rep(3,d), 2*diag(d),log=FALSE))  
  f2=log(0.5*dmnorm(x,rep(1.5,d), diag(d),log=FALSE)+0.5*dmnorm(x,rep(4.5,d), 2*diag(d),log=FALSE)) 
  1/(1+((1-prop)/prop)*exp(f2-f1))
}
obj = function(x){
  ifelse(eta(x)>1/2,1-eta(x),eta(x))
}
risk_Bayes = mean(obj(myx))  
risk_Bayes = round(risk_Bayes, digits=6) 
risk_Bayes
rm(myx)
 
#######################################################################
# data generating function for sim3
#######################################################################
mydata_sim2 = function(n, d, portion){
  #data generation function	
  n1 = floor(n*portion)
  n2 = n - n1	
  X1 = rbind(mvrnorm(floor(n1/2), rep(0,d), diag(d)),mvrnorm(n1 - floor(n1/2), rep(3,d), 2*diag(d)))     				
  X2 = rbind(mvrnorm(floor(n2/2), rep(1.5,d), diag(d)),mvrnorm(n2 - floor(n2/2), rep(4.5,d), 2*diag(d)))     					
  data1 = rbind(X1,X2)
  y = c(rep(1,n1),rep(2,n2))
  DATA=cbind(data1,y)
  DATA
}

#######################################################################


###################################################################################
# Oracle Training data for classifier \phi_1 and \phi_2, we generate 2N observations
# DATA1 is used for both CIS and Regret, DATA2 is used for CIS
####################################################################################

# generalized training set as dataframe
DATA_oracle1 = as.data.frame(mydata_sim2(N, d, portion))  
DATA_oracle2 = as.data.frame(mydata_sim2(N, d, portion))  

# genelize testing dataset
TEST = cbind(mydata_sim2(ntest, d, portion))

####################################################
#  Divide & Conquer Process
####################################################

# initialize prediect.sum list to store sum of prediction before mv
predict1.sum_d = rep(0,ntest)
predict2.sum_d = rep(0,ntest)

# Step 1: Dividing process
permIndex_1 = sample(nrow(DATA_oracle1))
DATA_oracle1 = DATA_oracle1[permIndex_1,]
permIndex_2 = sample(nrow(DATA_oracle2))
DATA_oracle2 = DATA_oracle2[permIndex_2,]

# Step 1.5 find sub.test sample

# define function to get test sample of 1nn from N_sub subsets
creat.TEST.sub = function(train,test,N_sub,s,n,theta){
  # initialize a matrix to store N_sub matrix which are 1NN vector of TEST
  # save matrix 1 first, until matrix N_sub
  TEST.Nsub = matrix(nrow=(N_sub*nrow(test)), ncol=ncol(test))
  for(i in 1:N_sub) {
    for(j in 1:nrow(test)){
      sub_numbers = ceiling(((s*n)^theta)/n)   # calculate how many subsets need to sample enough values
      sub_index = sample(1:s,sub_numbers)   # sample subsample index to find 1NN test point
      obs_index = rep((sub_index-1)*n, each=n) + rep(1:n,sub_numbers)   # observation index correponedent ot subsample index
      train.sub = as.matrix(train[sample(obs_index, ceiling((s*n)^theta)),])  # sample to find 1NN to denoise

      n.sub = dim(train.sub)[1]   # get n for selected subsample
      d = dim(train.sub)[2] - 1  # get d for selected subsample
      X.sub = as.matrix(train.sub[,1:d])
      Y.sub = train.sub[, d+1]
      dist = function(x){
        sqrt(t(x - TEST[j,1:d]) %*% (x - TEST[j,1:d]))
      }
      Dis = apply(X.sub, 1, dist)
      Y_1NN = Y.sub[order(Dis)[1]]   #find Y for 1NN in subset for a query point in TEST
      X_1NN = X.sub[order(Dis)[1],]  #find X for 1NN in subset for a query point in TEST
      TEST.Nsub[j+nrow(test)*(i-1),1:d] = X_1NN
      TEST.Nsub[j+nrow(test)*(i-1),d+1] = Y_1NN
    }  
  }  
  return(TEST.Nsub) 
}


#Step 2: run base NN on subsets
time_dac_divide_d = 0
time_dac_temp0_d = Sys.time()


# generate the testing dataset
TEST_denoised1 = creat.TEST.sub(DATA_oracle1,TEST,N_sub,s,n,theta)
TEST_denoised2 = creat.TEST.sub(DATA_oracle2,TEST,N_sub,s,n,theta)

# record denoising time
time_dac_temp1_d = Sys.time()
time_dac_temp_d = time_dac_temp1_d - time_dac_temp0_d
time_dac_temp_d = round(as.numeric(time_dac_temp_d, units = "secs"), digits=2) 
time_dac_temp_d = time_dac_temp_d/N_sub   #parallel calculation to save time
if(time_dac_temp_d >= time_dac_divide_d) time_dac_divide_d = time_dac_temp_d 

index = 1
for(j in 1:s){

  # Training data for classifier \phi_1 and \phi_2 in CIS, we generate 2n observations
  DATA1 = DATA_oracle1[index:(index+n-1),]
  DATA2 = DATA_oracle2[index:(index+n-1),]

  # get preditive value from trained classfier on testing data
  predict1_d = myknn(DATA1, TEST_denoised1[,1:d], k)
  predict2_d = myknn(DATA2, TEST_denoised1[,1:d], k)

  # sum all predict for majority voting
  predict1.sum_d = predict1.sum_d + predict1_d
  predict2.sum_d = predict2.sum_d + predict2_d

  # moving index to beginning of next subset
  index = index + n
}
  
# Step3: Majority voting process 
time_dac_combine0_d = Sys.time()

for(i in 1:length(threshold)){
  # Majority voting for bigNN
  predict1.mv_d = ifelse(s^(-1)*predict1.sum_d > 2-threshold[i], 2, 1)
  predict2.mv_d = ifelse(s^(-1)*predict2.sum_d > 2-threshold[i], 2, 1)

  # Majority voting for N_sub results for denoised bigNN
  predict1.sum.denoised = 0
  predict2.sum.denoised = 0
  for(l in 1:N_sub){  
    predict1.sum.denoised = predict1.sum.denoised + predict1.mv_d[(1+(l-1)*nrow(TEST)):(l*nrow(TEST))] 
    predict2.sum.denoised = predict2.sum.denoised + predict2.mv_d[(1+(l-1)*nrow(TEST)):(l*nrow(TEST))] 
  }
  predict1.mv.denoised = ifelse(N_sub^(-1)*predict1.sum.denoised > 2-threshold, 2, 1) 
  predict2.mv.denoised = ifelse(N_sub^(-1)*predict2.sum.denoised > 2-threshold, 2, 1)

  # Store CIS value for denoised bigNN
  cis_big_d = mycis(predict1.mv.denoised, predict2.mv.denoised)
  # Store regret and risk value for denoised bigNN
  risk_big_d = myerror(predict1.mv.denoised, TEST[,d+1])

}

time_dac_combine1_d = Sys.time()
time_dac_combine_d = time_dac_combine1_d - time_dac_combine0_d
time_dac_combine_d = time_dac_combine_d/N_sub   # parallel process to save time
time_dac_combine_d = round(as.numeric(time_dac_combine_d, units="secs"),digits=2)  
time_dac_d = time_dac_divide_d + time_dac_combine_d 

# rounding decimals to be 2
portion = round(portion,digits=2)

# Print results together
print(paste(NO,gamma,N,s,n,d,portion,k,k_O,threshold,theta,N_sub,cis_big_d,risk_big_d,risk_Bayes,time_dac_d)) 




