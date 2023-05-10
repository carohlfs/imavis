library(data.table)
library(data.table)
library(tidyr)
library(scales)
library(lmtest)
library(aod)
library(parallel)
library(zoo)

cleaned.data <- readRDS('cleaned.data.RDS')
cleaned.data <- cleaned.data[size==256]
cleaned.data[,size:=NULL]
cleaned.data[,sample:=factor(sample,levels=c("valid","test"))]

cleaned.data[model=="resnext101_32x8d",gmacs:=16.51]
cleaned.data[model=="wide_resnet101_2",gmacs:=22.82]
cleaned.data[model=="efficientnet_b0",gmacs:=0.40167]
cleaned.data[model=="efficientnet_b7",gmacs:=5.27]
cleaned.data[model=="resnet101",gmacs:=7.85]
cleaned.data[model=="densenet201",gmacs:=4.37]
cleaned.data[model=="vgg19_bn",gmacs:=19.7]
cleaned.data[model=="mobilenet_v3_large",gmacs:=0.22836]
cleaned.data[model=="mobilenet_v3_small",gmacs:=0.06013]
cleaned.data[model=="googlenet",gmacs:=1.51]
cleaned.data[model=="inception_v3",gmacs:=2.85]
cleaned.data[model=="alexnet",gmacs:=0.71556]

cleaned.data[model=="resnext101_32x8d",msecs:=67.4]
cleaned.data[model=="wide_resnet101_2",msecs:=71.2]
cleaned.data[model=="efficientnet_b0",msecs:=25.5]
cleaned.data[model=="efficientnet_b7",msecs:=126.0]
cleaned.data[model=="resnet101",msecs:=29.6]
cleaned.data[model=="densenet201",msecs:=35.6]
cleaned.data[model=="vgg19_bn",msecs:=156.9]
cleaned.data[model=="mobilenet_v3_large",msecs:=15.0]
cleaned.data[model=="mobilenet_v3_small",msecs:=10.7]
cleaned.data[model=="googlenet",msecs:=14.5]
cleaned.data[model=="inception_v3",msecs:=20.8]
cleaned.data[model=="alexnet",msecs:=6.8]

nothresh <- cleaned.data[,list(correct=mean(correct),flops=mean(gmacs),msecs=mean(msecs)),by=list(sample,model)]
saveRDS(nothresh,"nothresh.RDS")

prioritize.thresh <- function(threshold,model.set,cost.var="gmacs"){
	last.model <- model.set[length(model.set)]
	priority.data <- cleaned.data[model %in% model.set]
	priority.data[,cost:=get(cost.var)]
	setkey(priority.data,sample,obs,cost)
	priority.data[,cum.cost:=cumsum(cost),by=list(sample,obs)]
	priority.data[,cum.flops:=cumsum(gmacs),by=list(sample,obs)]
	priority.data[,cum.msecs:=cumsum(msecs),by=list(sample,obs)]
	priority.data[,cummax.prob:=cummax(prob),by=list(sample,obs)]
	priority.data[,cummax.pred:=pred]
	priority.data[prob!=cummax.prob,cummax.pred:=NA]
	priority.data[,cummax.pred:=na.locf(cummax.pred,na.rm=FALSE)]
	priority.data[,cummax.correct:=as.numeric(round(cummax.pred)==round(actual))]
	priority.data <- priority.data[prob>=threshold | model==last.model]
	priority.data[,mincost:=min(cum.cost),by=list(sample,obs)]
	priority.data <- priority.data[mincost==cum.cost,list(sample,obs,model,cummax.prob,cum.flops,cum.msecs,cummax.correct)]
	priority.means <- priority.data[,list(accuracy=mean(cummax.correct),flops=mean(cum.flops),msecs=mean(cum.msecs)),by=sample]
	return(priority.means)
}

prioritize <- function(model.set,cost.var){
	thresholds <- c(0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99)
	priority.sets <- rbindlist(lapply(thresholds,prioritize.thresh,model.set,cost.var))
	thresh.vec <- rep(thresholds,each=2)
	priority.sets[,threshold:=thresh.vec]
	return(priority.sets)
}

prioritize.full <- function(){

	all.models <- c("mobilenet_v3_small","mobilenet_v3_large","efficientnet_b0",
		"alexnet","googlenet","inception_v3","densenet201","efficientnet_b7",
		"resnet101","resnext101_32x8d","vgg19_bn","wide_resnet101_2")
	select1 <- c("mobilenet_v3_small","efficientnet_b0","resnext101_32x8d")
	select2 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnet101",
		"resnext101_32x8d","wide_resnet101_2")
	select3 <- c("mobilenet_v3_small","efficientnet_b0","resnet101",
		"resnext101_32x8d")
	select4 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnext101_32x8d")

	candidates <- c("all1","all2","select1","select2","select3","select4")
	priorities.all1 <- prioritize(all.models,cost.var="gmacs")
	priorities.all2 <- prioritize(all.models,cost.var="msecs")
	priorities.select1 <- prioritize(select1,cost.var="msecs")
	priorities.select2 <- prioritize(select2,cost.var="gmacs")
	priorities.select3 <- prioritize(select3,cost.var="msecs")
	priorities.select4 <- prioritize(select4,cost.var="gmacs")

	priorities <- list(priorities.all1,priorities.all2,priorities.select1,
		priorities.select2,priorities.select3,priorities.select4)

	priorities.nrow <- lapply(priorities,nrow)
	candidate.vec <- rep(candidates,priorities.nrow)

	priorities <- rbindlist(priorities)
	priorities[,candidates:=candidate.vec]

	priorities <- melt(priorities,id=c("candidates","sample","threshold"))
	priorities <- spread(priorities,threshold,value)
	setkey(priorities,sample,candidates,variable)

	return(priorities)
}

max.model <- function(model.set){
	max.data <- cleaned.data[model %in% model.set]
	max.data[,flops:=sum(gmacs),by=list(sample,obs)]
	max.data[,msecs:=sum(msecs),by=list(sample,obs)]
	max.data[,pmax:=max(prob),by=list(sample,obs)]
	max.data <- max.data[prob==pmax,list(sample,obs,model,prob,flops,msecs,correct)]
	max.means <- max.data[,list(accuracy=mean(correct),flops=mean(flops),msecs=mean(msecs)),by=sample]
	return(max.means)
}

max.full <- function(){

	all.models <- c("mobilenet_v3_small","mobilenet_v3_large","efficientnet_b0",
		"alexnet","googlenet","inception_v3","densenet201","efficientnet_b7",
		"resnet101","resnext101_32x8d","vgg19_bn","wide_resnet101_2")
	select1 <- c("mobilenet_v3_small","efficientnet_b0","resnext101_32x8d")
	select2 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnet101",
		"resnext101_32x8d","wide_resnet101_2")
	select3 <- c("mobilenet_v3_small","efficientnet_b0","resnet101",
		"resnext101_32x8d")
	select4 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnext101_32x8d")

	candidates <- c("all1","all2","select1","select2","select3","select4")

	max.all <- max.model(all.models)
	max.select1 <- max.model(select1)
	max.select2 <- max.model(select2)
	max.select3 <- max.model(select3)
	max.select4 <- max.model(select4)

	maxes <- list(max.all,max.all,max.select1,max.select2,max.select3,max.select4)
	maxes.nrow <- lapply(maxes,nrow)
	candidate.vec <- rep(candidates,maxes.nrow)

	maxes <- rbindlist(maxes)
	maxes[,candidates:=candidate.vec]
	setkey(maxes,sample,candidates)
	setcolorder(maxes,c("sample","candidates","accuracy","flops","msecs"))

	return(maxes)

}

make.benchmark <- function(model.set){
	benchmark <- cleaned.data[model %in% model.set,list(accuracy=mean(correct),flops=mean(gmacs),msecs=mean(msecs)),by=list(sample,model)]
	benchmark[,max.accuracy:=max(accuracy),by=sample]
	benchmark <- benchmark[accuracy==max.accuracy,list(sample,accuracy,flops,msecs)]
	return(benchmark)
}

make.benchmarks <- function(){

	all.models <- c("mobilenet_v3_small","mobilenet_v3_large","efficientnet_b0",
		"alexnet","googlenet","inception_v3","densenet201","efficientnet_b7",
		"resnet101","resnext101_32x8d","vgg19_bn","wide_resnet101_2")
	select1 <- c("mobilenet_v3_small","efficientnet_b0","resnext101_32x8d")
	select2 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnet101",
		"resnext101_32x8d","wide_resnet101_2")
	select3 <- c("mobilenet_v3_small","efficientnet_b0","resnet101",
		"resnext101_32x8d")
	select4 <- c("mobilenet_v3_large","efficientnet_b0","densenet201","resnext101_32x8d")

	candidates <- c("all1","all2","select1","select2","select3","select4")

	benchmark.all <- make.benchmark(all.models)
	benchmark.select1 <- make.benchmark(select1)
	benchmark.select2 <- make.benchmark(select2)
	benchmark.select3 <- make.benchmark(select3)
	benchmark.select4 <- make.benchmark(select4)

	benchmarks <- list(benchmark.all,benchmark.all,benchmark.select1,benchmark.select2,benchmark.select3,benchmark.select4)

	benchmark.nrows <- lapply(benchmarks,nrow)
	candidate.vec <- rep(candidates,benchmark.nrows)

	benchmarks <- rbindlist(benchmarks)
	benchmarks[,candidates:=candidate.vec]
	benchmarks <- melt(benchmarks,id=c("sample","candidates"))
	setkey(benchmarks,sample,candidates,variable)
	setnames(benchmarks,"value","benchmark")
	return(benchmarks)

}

benchmarks <- make.benchmarks()

priorities <- prioritize.full()

maxes <- max.full()
maxes <- melt(maxes,id=c("sample","candidates"))
setkey(maxes,sample,candidates,variable)
setnames(maxes,"value","pmax")

priorities <- cbind(benchmarks,maxes[,list(pmax)],priorities[,list(`0.5`,`0.6`,`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
write.table(priorities,"priorities.inet.txt",sep='\t',row.names=FALSE)

# ptable <- priorities[candidates %in% c("all1","select2","select4")]
# write.table(ptable,"priorities.inet.txt",sep='\t',row.names=FALSE)