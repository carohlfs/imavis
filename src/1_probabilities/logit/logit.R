
library(data.table)
library(tidyr)
library(lmtest)
library(aod)

probs.direc <- '../tmp/probs/'

formula.j <- function(j){
	xset <- paste0("V",1:j,collapse=' + ')
	yset <- paste0("y",0:9)
	return(paste(yset,xset,sep=' ~ '))
}

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
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

datanames <- c("mnist","kmnist","fashionmnist","cifar","svhn")

for(i in 1:5){
	if(grepl("mnist",datanames[i])){
		ds.rates <- c(28,14,7,4,2)
		factors <- 100
	} else {
		ds.rates <- c(32,16,8,4,2)
		factors <- 300
	}
	train_pca <- readRDS(paste0(probs.direc,datanames[i],"_train_pca.RDS"))
	regs <- run.logits(train_pca,factors)
	saveRDS(regs,paste0(probs.direc,datanames[i],"_logits.RDS"))
	cat(paste0("saved ",probs.direc,datanames[i],"_logits.RDS.\n"))
	train_probs <- forecast.logit(train_pca,regs)
	saveRDS(train_probs,paste0(probs.direc,datanames[i],"_train_lprobs.RDS"))
	cat(paste0("saved ",probs.direc,datanames[i],"_train_lprobs.RDS.\n"))
	for(r in ds.rates){
		ds_pca <- readRDS(paste0(probs.direc,datanames[i],"_test",r,"_pca.RDS"))
		ds_probs <- forecast.logit(ds_pca,regs)
		saveRDS(ds_probs,paste0(probs.direc,datanames[i],"_test",r,"_lprobs.RDS"))
		cat(paste0("saved ",probs.direc,datanames[i],"_test",r,"_lprobs.RDS.\n"))
	}
}


