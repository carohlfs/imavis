
library(data.table)

load.mnist <- function(file){
	train.file <- paste0(file,"_train.csv")
	test.file <- paste0(file,"_test.csv")
	mnist.train <- fread(train.file)
	mnist.train[,source:="train"]
	mnist.test <- fread(test.file)
	mnist.test[,source:="test"]
	mnist <- rbind(mnist.train,mnist.test)
	mnist.names <- c("y",paste0("V",1:784),"source")
	setnames(mnist,mnist.names)
	mnist[,(paste0("V",1:784)):=lapply(.SD,as.double),.SDcols=paste0("V",1:784)]
	mnist[,id:=seq(1,nrow(mnist))]
	setcolorder(mnist,c("id","source","y",paste0("V",1:784)))
	return(mnist)
}

downsample.mnist <- function(rate,dataset){
	xvars <- names(dataset)[grepl("V",names(dataset))]
	idxes <- data.table(variable=xvars)
	xvars <- dataset[,mget(xvars)]
	xvars <- cbind(dataset[,list(id)],xvars)
	xvars <- melt(xvars,id="id")
	xvars[,value:=as.double(value)]
	xvars[,idx:=as.numeric(gsub("V","",variable))]
	xvars[,i:= idx %% 28]
	xvars[i==0,i:=28]
	xvars[,j:= ceiling(idx/28)]
	xvars[,dsi:= rate*ceiling(i/rate)]
	xvars[,dsj:= rate*ceiling(j/rate)]
	xvars[,vargroup:=paste0("V",28*(dsj-1)+dsi)]
	xvars[,value:=mean(value),by=list(id,vargroup)]
	xvars[,c("i","j","dsi","dsj","vargroup","idx"):=NULL]
	xvars <- dcast(xvars,id~variable)
	xvars[,id:=NULL]
	xvars <- cbind(dataset[,list(id,source,y)],xvars)
	return(xvars)
}

downsample.3channel <- function(rate,dataset){
	xvars <- names(dataset)[grepl("V",names(dataset))]
	idxes <- data.table(variable=xvars)
	xvars <- dataset[,mget(xvars)]
	xvars <- cbind(dataset[,list(id)],xvars)
	xvars <- melt(xvars,id="id")
	xvars[,value:=as.double(value)]
	xvars[,idx:=as.numeric(gsub("V","",variable))]
	# we don't input data name, so let's
	# just use a fact about SVHN for now.
	if(nrow(dataset)>80000){
		xvars[,k:= idx %% 3]
		xvars[k==0,k:= 3]
		xvars[,i:= ceiling(idx/3) %% 32]
		xvars[i==0,i:=32]
		xvars[,j:= ceiling(idx/96)]
	} else {
		xvars[,j:= idx %% 32]
		xvars[j==0,j:=32]
		xvars[,i:= ceiling(idx/32) %% 32]
		xvars[i==0,i:=32]
		xvars[,k:=ceiling(idx/1024)]
	}
	xvars[,dsi:= rate*ceiling(i/rate)]
	xvars[,dsj:= rate*ceiling(j/rate)]
	xvars[,vargroup:=max(idx),by=list(dsi,dsj,k)]
	xvars[,value:=mean(value),by=list(id,vargroup)]
	xvars[,c("i","j","k","dsi","dsj","vargroup","idx"):=NULL]
	xvars <- dcast(xvars,id~variable)
	xvars[,id:=NULL]
	xvars <- cbind(dataset[,list(id,source,y)],xvars)
	return(xvars)
}

# ds.rates of 1 are included as a test to ensure
# that reshaping process isn't altering data on its own
downsample <- function(dataname){
	if(grepl("mnist",dataname)){
		ds.rates <- c(1,2,4,7,14)
		outfiles <- paste0(dataname,ds.rates,".RDS")
		downsampled <- lapply(ds.rates,downsample.mnist,get(dataname))
	} else {
		ds.rates <- c(1,2,4,8,16)
		outfiles <- paste0(dataname,ds.rates,".RDS")
		downsampled <- lapply(ds.rates,downsample.3channel,get(dataname))
	}
	mapply(saveRDS,downsampled,outfiles)
}

# Clean MNIST Data
mnist <- load.mnist("mnist")
saveRDS(mnist,"mnist.RDS")

# Clean KMNIST Data
kmnist <- load.mnist("kmnist")
saveRDS(kmnist,"kmnist.RDS")

# Clean FashionMNIST Data
fashionmnist <- load.mnist("fashionmnist")
saveRDS(fashionmnist,"fashionmnist.RDS")

# Clean SVHN Data
svhn.train <- fread("svhn_train.csv")
svhn.train[,source:="train"]
svhn.test <- fread("svhn_test.csv")
svhn.test[,source:="test"]
svhn <- rbind(svhn.train,svhn.test)
svhn[,V1:=NULL]
svhn.names <- c("y",paste0("V",1:3072),"source")
setnames(svhn,svhn.names)
svhn[,(paste0("V",1:3072)):=lapply(.SD,as.double),.SDcols=paste0("V",1:3072)]
svhn[,id:=seq(1,nrow(svhn))]
svhn[y==10,y:=0]
setcolorder(svhn,c("id","source","y",paste0("V",1:3072)))
saveRDS(svhn,"svhn.RDS")

# Clean CIFAR Data
cifar.direc <- "/Volumes/T7/cifar-10-batches-py/"
cifar.train.files <- paste0(cifar.direc,"cifar_data",1:5,".csv")
cifar.train.data <- lapply(cifar.train.files,fread)
cifar.train.data <- rbindlist(cifar.train.data)
cifar.train.labelfiles <- paste0(cifar.direc,"cifar_labels",1:5,".csv")
cifar.train.labels <- lapply(cifar.train.labelfiles,fread)
cifar.train.labels <- rbindlist(cifar.train.labels)
cifar.test.file <- paste0(cifar.direc,"cifar_testdata.csv")
cifar.test.data <- fread(cifar.test.file)
cifar.test.labelfile <- paste0(cifar.direc,"cifar_testlabels.csv")
cifar.test.labels <- fread(cifar.test.labelfile)
setnames(cifar.train.labels,"y")
setnames(cifar.test.labels,"y")
cifar.train.labels[,source:="train"]
cifar.test.labels[,source:="test"]
cifar.labels <- rbind(cifar.train.labels,cifar.test.labels)
cifar.labels[,id:=seq(1,nrow(cifar.labels))]
setcolorder(cifar.labels,c("id","source","y"))
cifar <- rbind(cifar.train.data,cifar.test.data)
cifar <- cbind(cifar.labels,cifar)
cifar[,(paste0("V",1:3072)):=lapply(.SD,as.double),.SDcols=paste0("V",1:3072)]
saveRDS(cifar,"cifar.RDS")

datanames <- c("mnist","kmnist","fashionmnist","svhn","cifar")
lapply(datanames,downsample)






