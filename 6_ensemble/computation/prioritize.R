library(data.table)
library(data.table)
library(tidyr)
library(scales)
library(lmtest)
library(aod)
library(parallel)
library(zoo)

prioritize.thresh <- function(threshold,datasets,model.set,adj=FALSE){
	last.model <- model.set[length(model.set)]
	priority.data <- small.results[dataset %in% datasets & model %in% model.set & sample=="test"]
	if(adj){
		priority.data[,phat:=NULL]
		setnames(priority.data,"phat.new","phat")
	}
	setkey(priority.data,dataset,id,gmacs)
	priority.data[,cum.flops:=cumsum(gmacs),by=list(dataset,id)]
	priority.data[,cum.msec:=cumsum(msec),by=list(dataset,id)]
	priority.data[,cummax.phat:=cummax(phat),by=list(dataset,id)]

	priority.data[,cummax.yhat:=yhat]
	priority.data[phat!=cummax.phat,cummax.yhat:=NA]

	priority.data[,cummax.yhat:=na.locf(cummax.yhat,na.rm=FALSE)]
	priority.data[,cummax.correct:=as.numeric(round(cummax.yhat)==round(y))]
	priority.data <- priority.data[phat>=threshold | model==last.model]
	priority.data[,minflops:=min(cum.flops),by=list(dataset,id)]
	priority.data <- priority.data[minflops==cum.flops,list(dataset,id,model,cummax.phat,cummax.correct,cum.msec,cum.flops)]
	priority.means <- priority.data[,list(accuracy=mean(cummax.correct),flops=mean(cum.flops),msec=mean(cum.msec)),by=dataset]
	return(priority.means)
}

prioritize <- function(datasets,model.set,adj=FALSE){
	thresholds <- c(0.7,0.8,0.9,0.95,0.975,0.99)
	priority.sets <- lapply(thresholds,prioritize.thresh,datasets,model.set,adj)
	thresh.nrows <- sapply(priority.sets,nrow)
	thresh.vec <- rep(thresholds,thresh.nrows)
	priority.sets <- rbindlist(priority.sets)
	priority.sets[,threshold:=thresh.vec]
	priority.sets <- melt(priority.sets,id=c("dataset","threshold"))
	priority.sets[,phat:=ifelse(adj,"adj","unadj")]
	priority.sets <- spread(priority.sets,threshold,value)
	return(priority.sets)
}

prioritize.full <- function(adj=FALSE){

	mnist.datasets <- c("mnist","kmnist","fashionmnist")
	mnist.models <- c("logit","lenet")

	rgb.datasets <- c("svhn","cifar")
	rgb.full <- c("logit","lenet","resnet","dla")
	rgb.neural <- c("resnet","dla")

	p.mnist <- prioritize(mnist.datasets,mnist.models)
	p.mnist.adj <- prioritize(mnist.datasets,mnist.models,adj=TRUE)
	p.rgb.full <- prioritize(rgb.datasets,rgb.full)
	p.rgb.full.adj <- prioritize(rgb.datasets,rgb.full,adj=TRUE)
	p.rgb.neural <- prioritize(rgb.datasets,rgb.neural)
	p.rgb.neural.adj <- prioritize(rgb.datasets,rgb.neural,adj=TRUE)

	priority <- list(p.mnist,p.mnist.adj,p.rgb.full,p.rgb.full.adj,
		p.rgb.neural,p.rgb.neural.adj)

	model.specs <- c("full","full","full","full","neural","neural")
	nrow.vec <- sapply(priority,nrow)
	model.vec <- rep(model.specs,nrow.vec)

	priority <- rbindlist(priority)
	priority[,candidates:=model.vec]

	priority[,phat:=factor(phat,levels=c("unadj","adj"))]

	setcolorder(priority,c("dataset","candidates","phat","variable",
		"0.7","0.8","0.9","0.95","0.975","0.99"))

	setkey(priority,dataset,candidates,phat,variable)

	return(priority)
}

max.model <- function(datasets,model.set){
	max.data <- small.results[dataset %in% datasets & model %in% model.set]
	max.data[,flops:=sum(gmacs),by=list(dataset,id)]
	max.data[,msec:=sum(msec),by=list(dataset,id)]
	max.data[,pmax:=max(phat),by=list(dataset,id)]
	max.data[,pmax.adj:=max(phat.new),by=list(dataset,id)]
	max.vals <- max.data[phat==pmax,list(dataset,id,model,flops,msec,correct)]
	max.vals[,phat:="unadj"]
	max.vals.adj <- max.data[phat.new==pmax.adj,list(dataset,id,model,flops,msec,correct)]
	max.vals.adj[,phat:="adj"]
	max.vals <- rbind(max.vals,max.vals.adj)
	max.means <- max.vals[,list(accuracy=mean(correct),flops=mean(flops),msec=mean(msec)),by=list(dataset,phat)]
	return(max.means)
}

max.full <- function(){

	mnist.datasets <- c("mnist","kmnist","fashionmnist")
	mnist.models <- c("logit","lenet")

	rgb.datasets <- c("svhn","cifar")
	rgb.full <- c("logit","lenet","resnet","dla")
	rgb.neural <- c("resnet","dla")

	m.mnist <- max.model(mnist.datasets,mnist.models)
	m.rgb.full <- max.model(rgb.datasets,rgb.full)
	m.rgb.neural <- max.model(rgb.datasets,rgb.neural)

	maxes <- list(m.mnist,m.rgb.full,m.rgb.neural)

	model.specs <- c("full","full","neural")
	nrow.vec <- sapply(maxes,nrow)
	model.vec <- rep(model.specs,nrow.vec)

	maxes <- rbindlist(maxes)
	maxes[,candidates:=model.vec]

	maxes <- melt(maxes,id=c("dataset","candidates","phat"))
	setnames(maxes,"value","pmax")
	maxes[,phat:=factor(phat,levels=c("unadj","adj"))]
	setkey(maxes,dataset,candidates,phat,variable)

	return(maxes)
}

make.benchmark <- function(datasets,model.set){
	benchmark <- small.results[dataset %in% datasets & model %in% model.set,list(accuracy=mean(correct),flops=mean(gmacs),msec=mean(msec)),by=list(dataset,model)]
	benchmark[,max.accuracy:=max(accuracy),by=dataset]
	benchmark <- benchmark[accuracy==max.accuracy,list(dataset,accuracy,flops,msec)]
	return(benchmark)
}

make.benchmarks <- function(){

	mnist.datasets <- c("mnist","kmnist","fashionmnist")
	mnist.models <- c("logit","lenet")

	rgb.datasets <- c("svhn","cifar")
	rgb.full <- c("logit","lenet","resnet","dla")
	rgb.neural <- c("resnet","dla")

	b.mnist <- make.benchmark(mnist.datasets,mnist.models)
	b.rgb.full <- make.benchmark(rgb.datasets,rgb.full)
	b.rgb.neural <- make.benchmark(rgb.datasets,rgb.neural)

	benchmarks <- list(b.mnist,b.rgb.full,b.rgb.neural)

	model.specs <- c("full","full","neural")
	nrow.vec <- sapply(benchmarks,nrow)
	model.vec <- rep(model.specs,nrow.vec)

	benchmarks <- rbindlist(benchmarks)
	benchmarks[,candidates:=model.vec]
	benchmarks[,phat:="unadj"]
	bcopy <- copy(benchmarks)
	bcopy[,phat:="adj"]
	benchmarks <- rbind(benchmarks,bcopy)
	benchmarks[,phat:=factor(phat,levels=c("unadj","adj"))]

	benchmarks <- melt(benchmarks,id=c("dataset","candidates","phat"))
	setnames(benchmarks,"value","benchmark")
	setkey(benchmarks,dataset,candidates,phat,variable)

	return(benchmarks)
}

small.results <- readRDS('small.results.RDS')
small.results <- small.results[sample=="test"]

small.results[model=="logit",msec:=as.numeric(0)]
small.results[model=="lenet",msec:=0.1]
small.results[model=="dla",msec:=5.8]
small.results[model=="resnet",msec:=4.8]

priorities <- prioritize.full()
maxes <- max.full()
benchmarks <- make.benchmarks()

priorities <- cbind(benchmarks,maxes[,list(pmax)],priorities[,list(`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
write.table(priorities,"priorities.small.txt",sep='\t',row.names=FALSE)
