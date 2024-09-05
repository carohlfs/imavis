
library(data.table)
library(tidyr)
library(lmtest)
library(aod)

data.direc <- '../data/'
probs.direc <- '../tmp/probs/'

# The outcome data live in the data directory, having
# been saved by the savecsv.py file.
load.y <- function(sample,dataname){
	y.file <- paste0(data.direc,dataname,"/",dataname,"_",sample,".csv")
	y <- fread(y.file)[,list(V1)]
	setnames(y,"y")
	y[,id:=.I]
	y[,source:=sample]
	setcolorder(y,c("source","id","y"))
	return(y)
}

assemble.y <- function(dataname){
	y.data <- lapply(c("train","test"),load.y,dataname)
	y.data <- rbindlist(y.data)
	setkey(y.data,source,id)
	return(y.data)
}

# The normalized pixel intensity data live in the tmp
# directory, having been saved by lenet.py.
load.x <- function(dsrate,sample,dataname){
	x.file <- paste0(probs.direc,dataname,dsrate,"_",sample,"_X.csv")
	x <- fread(x.file)
	xvars <- copy(names(x))
	x[,id:=.I]
	x[,source:=sample]
	x[,dsrate:=dsrate]
	setcolorder(x,c("source","id","dsrate",xvars))
	return(x)
}

assemble.x <- function(dataname){
	ds.rates <- if(grepl("mnist",dataname)){
		c(28,14,7,4,2)
	} else {
		c(32,16,8,4,2)
	}
	x.test <- lapply(ds.rates,load.x,"test",dataname)
	x <- rbindlist(x.test)
	x.train <- load.x(ds.rates[1],"train",dataname)
	x <- rbind(x,x.train)
	return(x)
}

assemble.yx <- function(dataname){
	y <- assemble.y(dataname)
	setkey(y,source,id)
	x <- assemble.x(dataname)
	xvars <- copy(names(x))[!(c("source","id","dsrate") %in% names(x))]
	setkey(x,source,id)
	dataset <- y[x]
	dataset[,dataname:=dataname]
	setcolorder(dataset,c("dataname","source","id","dsrate","y",xvars))
	saveRDS(dataset,paste0(probs.direc,dataname,".RDS"))
	cat(paste0("Saved ",probs.direc,dataname,".RDS\n"))
	return(dataset)
}

assemble.data <- function(){
	datanames <- c('mnist','kmnist','fashionmnist','cifar','svhn')
	data.list <- lapply(datanames,assemble.yx)
	return(data.list)
}

gen.eigs <- function(dataset){
	Vs <- names(dataset)[grepl("V",names(dataset))]
	xvars <- as.matrix(dataset[source=="train",mget(Vs)])
	covs <- cov(xvars)
	eigs <- eigen(covs)
	eigs <- eigs$vectors
	return(eigs)
}

gen.pca <- function(dataset,eigs){
	Vs <- names(dataset)[grepl("V",names(dataset))]
	xvars <- as.matrix(dataset[,mget(Vs)])
	pca <- xvars %*% eigs
	pca <- data.table(pca)
	pca <- cbind(dataset[,list(source,id,dsrate,y)],pca)
	return(pca)
}

datanames <- c('mnist','kmnist','fashionmnist','cifar','svhn')
# To skip regenerating on later calls, use this instead
# data.files <- paste0(probs.direc,datanames,".RDS")
# datasets <- lapply(data.files,readRDS)
datasets <- assemble.data()

for(i in 1:5){
	eigs <- gen.eigs(datasets[[i]])
	saveRDS(eigs,paste0(probs.direc,datanames[i],"_eigs.RDS"))
	cat(paste0("saved ",probs.direc,datanames[i],"_eigs.RDS.\n"))
	train.pca <- gen.pca(datasets[[i]][source=="train"],eigs)
	saveRDS(train.pca,paste0(probs.direc,datanames[i],"_train_pca.RDS"))
	cat(paste0("saved ",probs.direc,datanames[i],"_train_pca.RDS.\n"))
	ds.rates <- if(grepl("mnist",datanames[i])){
		c(28,14,7,4,2)
	} else {
		c(32,16,8,4,2)
	}
	for(j in ds.rates){
		ds.pca <- gen.pca(datasets[[i]][source=="test" & dsrate==j],eigs)
		saveRDS(ds.pca,paste0(probs.direc,datanames[i],"_test",j,"_pca.RDS"))
		cat(paste0("saved ",probs.direc,datanames[i],"_test",j,"_pca.RDS.\n"))
	}
}


