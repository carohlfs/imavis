library(data.table)
library(tidyr)
library(scales)
library(lmtest)
library(aod)

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
}

# logit.path <- "/Volumes/T7/effort/probs/"
# datasets <- c("mnist","kmnist","fashionmnist","svhn","cifar")
# factors <- c(100,100,100,300,300)
# logit.files <- c(paste0(logit.path,datasets,"1_train_lprobs",factors,".RDS"),
# 	paste0(logit.path,datasets,"1_test_lprobs",factors,".RDS"))
# logit.data <- lapply(logit.files,readRDS)
# row.nums <- sapply(logit.data,nrow)
# data.names <- rep(rep(datasets,2),row.nums)
# sample.names <- rep(c(rep("train",5),rep("test",5)),row.nums)
# logit.data <- rbindlist(logit.data)
# logit.data[,dataset:=data.names]
# logit.data[,sample:=sample.names]
# logit.data[,model:="logit"]
# logit.data[,id:=.I]
# logit.data[,minid:=min(id),by=list(dataset,sample)]
# logit.data[,id:=id-minid+1]
# logit.data[,minid:=NULL]
# actuals <- logit.data[,list(dataset,sample,id,y)]
# setkey(actuals,dataset,sample,id)

# neural.paths <- c(rep("",5),"pytorch-svhn-master/","pytorch-cifar-master/","pytorch-svhn-resnet/","pytorch-cifar-resnet/")
# neural.paths <- paste0("/Volumes/T7 Shield/Effort_Jul2022/",neural.paths)
# datasets <- c("mnist","kmnist","fashionmnist","svhn","cifar","svhn","cifar","svhn","cifar")
# neural.train.ends <- c(rep("_train_probs.csv",5),"_train_dlaprobs.csv","_train_dlaprobs.csv","_train_resprobs.csv","_train_resprobs.csv") 
# neural.test.ends <- c(rep("_test_probs.csv",5),"_test_dlaprobs.csv","_test_dlaprobs.csv","_test_resprobs.csv","_test_resprobs.csv")
# neural.files <- c(paste0(neural.paths,datasets,neural.train.ends),paste0(neural.paths,datasets,neural.test.ends))
# neural.data <- lapply(neural.files,fread)
# row.nums <- sapply(neural.data,nrow)
# data.names <- rep(rep(datasets,2),row.nums)
# models <- rep(c(rep("lenet",5),"dla","dla","resnet","resnet"),2)
# models <- rep(models,row.nums)
# samples <- rep(c(rep("train",9),rep("test",9)),row.nums)
# neural.data <- rbindlist(neural.data)
# neural.data[,dataset:=data.names]
# neural.data[,model:=models]
# neural.data[,sample:=samples]
# Vs <- paste0("V",1:10)
# neural.data[model!="lenet",(Vs):=lapply(.SD,exp),.SDcols=Vs]
# neural.data[,total:=V1+V2+V3+V4+V5+V6+V7+V8+V9+V10]
# neural.data[,(Vs):=lapply(.SD,`*`,1/total),.SDcols=Vs]
# preds <- sapply(1:nrow(neural.data),argpmax,neural.data[,mget(Vs)])
# neural.data[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
# neural.data[,yhat:=preds]
# neural.data <- neural.data[,list(yhat,phat,dataset,sample,model)]
# neural.data[,id:=.I]
# neural.data[,minid:=min(id),by=list(model,sample,dataset)]
# neural.data[,id:=id-minid+1]
# neural.data[,minid:=NULL]
# setkey(neural.data,dataset,sample,id)
# neural.data <- actuals[neural.data]
# result.cols <- c("dataset","sample","id","model","y","yhat","phat")
# setcolorder(neural.data,result.cols)
# setcolorder(logit.data,result.cols)

# small.results <- rbind(logit.data,neural.data)
# setkey(small.results,dataset,sample,model,id)

# small.results[model=="logit" & grepl("mnist",dataset),params:=101*10e-6]
# small.results[model=="logit" & !grepl("mnist",dataset),params:=301*10e-6]
# small.results[model=="logit",gflops:=as.numeric(0)]
# small.results[model=="logit",gmacs:=as.numeric(0)]
# small.results[model=="lenet",params:=61710e-6]
# small.results[model=="lenet",gflops:=0.00041652]
# small.results[model=="lenet",gmacs:=0.00042934]
# small.results[model=="dla",params:=15.14]
# small.results[model=="dla",gflops:=0.9156905]
# small.results[model=="dla",gmacs:=0.91576]
# small.results[model=="resnet",params:=11.17]
# small.results[model=="resnet",gflops:=0.55665152]
# small.results[model=="resnet",gmacs:=0.55665]

# small.results[,sample:=factor(sample,levels=c("train","test"))]
# small.results[,dataset:=factor(dataset,levels=c("mnist","kmnist","fashionmnist","svhn","cifar"))]
# small.results[,model:=factor(model,levels=c("logit","lenet","resnet","dla"))]
# small.results[,correct:=as.numeric(y==yhat)]

# saveRDS(small.results,"small.results.RDS")
small.results <- readRDS('small.results.RDS')

max.results <- copy(small.results)
max.results[,maxp:=max(phat),by=list(dataset,sample,id)]
max.results <- max.results[phat==maxp]
max.results[,mean(correct),by=list(dataset,sample)]

models <- c("logit","lenet","resnet","dla")
models <- factor(models,levels=models)
datasets <- c("mnist","kmnist","fashionmnist","svhn","cifar")
datasets <- factor(datasets,levels=datasets)
specs <- data.table(expand.grid(dataset=datasets,model=models))
specs <- specs[!(model %in% c("resnet","dla")) | dataset %in% c("svhn","cifar")]
setkey(specs,dataset,model)
specs[,spec:=.I]
