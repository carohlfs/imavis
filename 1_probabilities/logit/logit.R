
library(data.table)
library(tidyr)
library(lmtest)
library(aod)

formula.j <- function(j){
	xset <- paste0("V",1:j,collapse=' + ')
	yset <- paste0("y",0:9)
	return(paste(yset,xset,sep=' ~ '))
}

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
}

forecast.outcome <- function(pca,regs){
	regdata <- copy(pca)
	probs <- paste0("p",0:9)
	regdata[,(probs):=lapply(regs,predict,regdata)]
	# convert probabilities into softmaxes
	regdata[,(probs):=lapply(.SD,exp),.SDcols=probs]
	regdata[,total:=p0+p1+p2+p3+p4+p5+p6+p7+p8+p9]
	regdata[,(probs):=lapply(.SD,`*`,1/total),.SDcols=probs]
	regdata[,phat.softmax:=pmax(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9)]
	regdata[,phat.linear:=log(total*phat.softmax)]
	vectable <- regdata[,mget(probs)]
	regdata[,yhat:=sapply(1:nrow(regdata),argpmax,vectable)]
	return(regdata[,list(y,yhat,phat.linear,phat.softmax)])
}

forecast.logit <- function(pca,regs){
	regdata <- copy(pca)
	probs <- paste0("p",0:9)
	regdata[,(probs):=lapply(regs,predict,regdata,type="response")]
	regdata[,phat:=pmax(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9)]
	vectable <- regdata[,mget(probs)]
	regdata[,yhat:=sapply(1:nrow(regdata),argpmax,vectable)]
	return(regdata[,list(y,yhat,phat)])
}

get.accuracy <- function(pca){
	return(nrow(pca[round(y)==yhat])/nrow(pca))
}

run.regs <- function(pca,factors){
	regdata <- copy(pca)
	for(j in 0:9){
		outcome <- paste0("y",j)
		regdata[,(outcome):= as.numeric(round(y)==j)]
	}
	ten.eqns <- formula.j(factors)
	ten.regs <- lapply(ten.eqns,lm,data=regdata[source=="train"])
	return(ten.regs)
}

run.logits <- function(pca,factors){
	regdata <- copy(pca)
	for(j in 0:9){
		outcome <- paste0("y",j)
		regdata[,(outcome):= as.numeric(round(y)==j)]
	}
	ten.eqns <- formula.j(factors)
	ten.regs <- lapply(ten.eqns,glm,data=regdata[source=="train"],family="binomial")
	return(ten.regs)
}

factor.test <- function(factors,dataset){
	reg.set <- run.regs(dataset,factors)
	outcomes <- forecast.outcome(dataset,reg.set)
	train.acc <- get.accuracy(outcomes[source=="train"])
	test.acc <- get.accuracy(outcomes[source=="test"])
	return(list(train=train.acc,test=test.acc))
}

# pcalist <- readRDS("pcalist.RDS")
name.list <- c("mnist","kmnist","fashionmnist","svhn","cifar")
# ds.pcas <- readRDS('ds.pcas.RDS')

for(i in 4:length(name.list)){
	if(grepl("mnist",name.list[i])){
		ds.rates <- c(1,2,4,7,14)
	} else {
		ds.rates <- c(1,2,4,8,16)
	}	
	for(j in c(300,500,750,1000)){
#	for(j in c(2,10,25,50,100,200)){
		ds.pcas <- readRDS(paste0(name.list[i],".ds.pcas.RDS"))
		regs <- run.regs(ds.pcas[[1]],j)
		saveRDS(regs,paste0("/Volumes/T7/probs/",name.list[i],"_regs",j,".RDS"))
		for(k in 1:length(ds.rates)){
			outfile.train <- paste0("/Volumes/T7/probs/",name.list[i],ds.rates[k],"_train_probs",j,".RDS")
			outcomes.train <- forecast.outcome(ds.pcas[[k]][source=="train"],regs)
			saveRDS(outcomes.train,outfile.train)
			outfile.test <- paste0("/Volumes/T7/probs/",name.list[i],ds.rates[k],"_test_probs",j,".RDS")
			outcomes.test <- forecast.outcome(ds.pcas[[k]][source=="test"],regs)
			saveRDS(outcomes.test,outfile.test)
		}
	}
}

for(i in 5:length(name.list)){
	if(grepl("mnist",name.list[i])){
		ds.rates <- c(1,2,4,7,14)
	} else {
		ds.rates <- c(1,2,4,8,16)
	}	
	for(j in c(300)){
#	for(j in c(2,10,25,50,100,200)){
		ds.pcas <- readRDS(paste0(name.list[i],".ds.pcas.RDS"))
		regs <- run.logits(ds.pcas[[1]],j)
		saveRDS(regs,paste0("/Volumes/T7/probs/",name.list[i],"_logits",j,".RDS"))
		for(k in 1:length(ds.rates)){
			outfile.train <- paste0("/Volumes/T7/probs/",name.list[i],ds.rates[k],"_train_lprobs",j,".RDS")
			outcomes.train <- forecast.logit(ds.pcas[[k]][source=="train"],regs)
			saveRDS(outcomes.train,outfile.train)
			outfile.test <- paste0("/Volumes/T7/probs/",name.list[i],ds.rates[k],"_test_lprobs",j,".RDS")
			outcomes.test <- forecast.logit(ds.pcas[[k]][source=="test"],regs)
			saveRDS(outcomes.test,outfile.test)			
		}
	}
}


