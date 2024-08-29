
library(data.table)
library(tidyr)
library(lmtest)
library(aod)


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
	pca <- cbind(dataset[,list(id,source,y)],pca)
	return(pca)
}

gen.ds.pcas <- function(dataname){
	if(grepl("mnist",dataname)){
		ds.rates <- c(1,2,4,7,14)
	} else {
		ds.rates <- c(1,2,4,8,16)
	}
	main.file <- paste0(dataname,".RDS")
	main.data <- readRDS(main.file)
	main.eigs <- gen.eigs(main.data)

	ds.files <- spec.list[data.name==dataname,filename]
	ds.data <- lapply(ds.files,readRDS)
	ds.pcas <- lapply(ds.data,gen.pca,main.eigs)
	saveRDS(ds.pcas,paste0(dataname,'.ds.pcas.RDS'))
}

# name.list <- c("mnist","kmnist","fashionmnist","svhn","cifar")
name.list <- "cifar"
ds.rates <- c(1,2,4,7,14)
spec.list <- data.table(expand.grid(data.name=name.list,ds.rate=ds.rates))
spec.list[!grepl("mnist",data.name) & ds.rate %% 7==0,ds.rate:=ds.rate*8/7]
spec.list[,filename:=paste0(data.name,ds.rate,".RDS")]
setkey(spec.list,data.name,ds.rate)

scrap <- lapply(name.list,gen.ds.pcas)
# for(idx in 1:5){
#	train.file <- paste0('../',name.list[idx],"_train_actuals.csv")
#	test.file <- paste0('../',name.list[idx],"_test_actuals.csv")
#	train.data <- fread(train.file)
#	setnames(train.data,"y")
#	test.data <- fread(test.file)
#	setnames(test.data,"y")
#	for(j in 1:5){
#		ds.pcas[[idx]][[j]] <- cbind(train.data,ds.pcas[[idx]][[j]])
#		ds.pcas[[idx]][[j+5]] <- cbind(test.data,ds.pcas[[idx]][[j+5]])
#	}	
# }
# saveRDS(ds.pcas,"ds.pcas.RDS")

# pcalist <- lapply(ds.pcas,`[[`,1)
# saveRDS(pcalist,"pcalist.RDS")




