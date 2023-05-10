library(data.table)
library(tidyr)
library(scales)
library(lmtest)
library(aod)
library(parallel)
library(zoo)
library(svglite)

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
}

get.linear <- function(i){
	dataname <- linear.specs[i,data]
	pullsource <- linear.specs[i,source]
	modeltype <- linear.specs[i,model]
	factors <- linear.specs[i,factors]
	ds.rate <- linear.specs[i,ds.rate]
	probtype <- ifelse(modeltype=="linear","_probs","_lprobs")
	filename <- paste0("/Volumes/T7/effort/probs/",dataname,ds.rate,"_",pullsource,probtype,factors,".RDS")
	data <- readRDS(filename)

	data[,pcat:=floor(10*phat)]
	data[pcat<1,pcat:=1]
	data[pcat>9,pcat:=9]
	n <- nrow(data)
	data[,correct:= as.numeric(y==yhat)]
	data[,count:=as.numeric(1/n)]

	cats <- data[,list(proj = mean(phat), actual = mean(correct), count = sum(count)),by=pcat]

	cats[,data:=dataname]
	cats[,model:=modeltype]
	cats[,factors:=factors]
	cats[,ds.rate:=ds.rate]
	cats[,source:=pullsource]
	setcolorder(cats,c("data","source","model","factors","ds.rate","pcat","proj","actual","count"))
	return(cats)
}

get.lenet <- function(i){
	dataname = lenet.specs[i,data]
	pullsource = lenet.specs[i,source]
	ds.rate <- lenet.specs[i,ds.rate]
	ds.show <- ifelse(ds.rate==1,"",as.character(ds.rate))
	actuals.file <- paste0('../',dataname,"_",pullsource,"_actuals.csv")
	probs.file <- paste0("../",dataname,ds.show,"_",pullsource,"_probs.csv")
	data <- fread(actuals.file)
	probs <- fread(probs.file)
	preds <- sapply(1:nrow(probs),argpmax,probs)
	probs[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
	data[,proj:=preds]
	data[,phat:=probs[,phat]]

	data[,pcat:=floor(10*phat)]
	data[pcat<1,pcat:=1]
	data[pcat>9,pcat:=9]
	n <- nrow(data)
	data[,correct:= as.numeric(round(V1)==round(proj))]
	data[,count:=as.numeric(1/n)]

	cats <- data[,list(proj = mean(phat), actual = mean(correct), count = sum(count)),by=pcat]

	cats[,data:=dataname]
	cats[,model:="lenet"]
	cats[,factors:=as.numeric(NA)]
	cats[,ds.rate:=ds.rate]
	cats[,source:=pullsource]
	setcolorder(cats,c("data","source","model","factors","ds.rate","pcat","proj","actual","count"))
	return(cats)	
}

get.dla <- function(i){
	dataname = dla.specs[i,data]
	pullsource = dla.specs[i,source]
	ds.rate <- dla.specs[i,ds.rate]
	ds.show <- ifelse(ds.rate==1,"",as.character(ds.rate))
	actuals.file <- paste0('../pytorch-',dataname,"-master/",dataname,"_",pullsource,"_dlaactuals.csv")
	probs.file <- paste0('../pytorch-',dataname,"-master/",dataname,ds.show,"_",pullsource,"_dlaprobs.csv")
	data <- fread(actuals.file)
	probs <- fread(probs.file)
	preds <- sapply(1:nrow(probs),argpmax,probs)
	Vs <- paste0("V",1:10)
	probs[,(Vs):=lapply(.SD,exp),.SDcols=Vs]
	probs[,total:=V1+V2+V3+V4+V5+V6+V7+V8+V9+V10]
	probs[,(Vs):=lapply(.SD,`*`,1/total),.SDcols=Vs]
	probs[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
	data[,proj:=preds]
	data[,phat:=probs[,phat]]
	data[,pcat:=floor(10*phat)]
	data[pcat<1,pcat:=1]
	data[pcat>9,pcat:=9]
	n <- nrow(data)
	data[,correct:= as.numeric(round(V1)==round(proj))]
	data[,count:=as.numeric(1/n)]

	cats <- data[,list(proj = mean(phat), actual = mean(correct), count = sum(count)),by=pcat]

	cats[,data:=dataname]
	cats[,model:="dla"]
	cats[,factors:=as.numeric(NA)]
	cats[,ds.rate:=ds.rate]
	cats[,source:=pullsource]
	setcolorder(cats,c("data","source","model","factors","ds.rate","pcat","proj","actual","count"))
	return(cats)	
}

get.res <- function(i){
	dataname = res.specs[i,data]
	pullsource = res.specs[i,source]
	ds.rate <- res.specs[i,ds.rate]
	ds.show <- ifelse(ds.rate==1,"",as.character(ds.rate))
	actuals.file <- paste0('../pytorch-',dataname,"-resnet/",dataname,"_",pullsource,"_resactuals.csv")
	probs.file <- paste0('../pytorch-',dataname,"-resnet/",dataname,ds.show,"_",pullsource,"_resprobs.csv")
	data <- fread(actuals.file)
	probs <- fread(probs.file)
	preds <- sapply(1:nrow(probs),argpmax,probs)
	Vs <- paste0("V",1:10)
	probs[,(Vs):=lapply(.SD,exp),.SDcols=Vs]
	probs[,total:=V1+V2+V3+V4+V5+V6+V7+V8+V9+V10]
	probs[,(Vs):=lapply(.SD,`*`,1/total),.SDcols=Vs]
	probs[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
	data[,proj:=preds]
	data[,phat:=probs[,phat]]
	data[,pcat:=floor(10*phat)]
	data[pcat<1,pcat:=1]
	data[pcat>9,pcat:=9]
	n <- nrow(data)
	data[,correct:= as.numeric(round(V1)==round(proj))]
	data[,count:=as.numeric(1/n)]

	cats <- data[,list(proj = mean(phat), actual = mean(correct), count = sum(count)),by=pcat]

	cats[,data:=dataname]
	cats[,model:="res"]
	cats[,factors:=as.numeric(NA)]
	cats[,ds.rate:=ds.rate]
	cats[,source:=pullsource]
	setcolorder(cats,c("data","source","model","factors","ds.rate","pcat","proj","actual","count"))
	return(cats)	
}

gen.graph.row <- function(results.table,model,variable,legend.location="bottomright"){
	get.vec <- function(data.name,variable,drate){
		vals <- rep(as.numeric(NA),9)
		for(v in 1:9){
			row <- results.table[data==data.name & ds.rate==drate & pcat==v]
			if(nrow(row)==1){
				vals[v] <- row[,get(variable)]
			}
		}
		return(vals)
	}

	if(length(legend.location)==1){
		legend.location <- rep(legend.location,5)
	}

	if(variable=="actual"){
		graph.ylabel = "% Test Cases Correct"
	} else if(variable=="count"){
		graph.ylabel = "% of Test Sample"
	}

	graph.name <- paste0(model,"_",variable,"_pcats.svg")

	yticks_val <- pretty_breaks(n=5)(c(0,1))

	x.labels <- c("\n\n\u2264 20%",
		"\n\n20% -\n30%",
		"\n\n30% -\n40%",
		"\n\n40% -\n50%",
		"\n\n50% -\n60%",
		"\n\n60% -\n70%",
		"\n\n70% -\n80%",
		"\n\n80% -\n90%",">90%")

	pcats <- factor(x.labels,levels=x.labels)	

	image.sizes <- c("28x28 / 32x32","14x14 / 16x16","7x7 / 8x8","4x4","2x2")
	ds.mnist <- c(1,2,4,7,14)
	ds.rgb <- c(1,2,4,8,16)

	svg(graph.name,width=83,height=11)
	par(mfrow=c(1,5))
	for(i in 1:5){

		mnist <- get.vec("mnist",variable,ds.mnist[i])
		kmnist <- get.vec("kmnist",variable,ds.mnist[i])
		fashionmnist <- get.vec("fashionmnist",variable,ds.mnist[i])
		svhn <- get.vec("svhn",variable,ds.rgb[i])
		cifar <- get.vec("cifar",variable,ds.rgb[i])

		par(mar=c(6,6,5,2), mgp=c(5,2,0), cex=2.1, cex.main=4.0, cex.lab=2.0)
		ylims=c(0,1)
		matplot(pcats,mnist,type="n",col="red3",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
		# rect(par("usr")[1], par("usr")[3],
		#     par("usr")[2], par("usr")[4],
		#     col = "white") # Color
		abline(h = seq(0.2,1.0,0.2), lty = "dashed", col = "gray30")
		lines(pcats,mnist,col="red3",lwd=10,lty=1)
		lines(pcats,kmnist,col="tan1",lwd=10,lty=1)
		lines(pcats,fashionmnist,col="green",lwd=10,lty=1)
		lines(pcats,svhn,col="dodgerblue",lwd=10,lty=1)
		lines(pcats,cifar,col="violet",lwd=10,lty=1)
		axis(2, at=yticks_val, lab=percent(yticks_val), pos=1, lwd=0, las=1)
		axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
		title(main=image.sizes[i])
		title(ylab=graph.ylabel, mgp = c(3.5, 1, 0))
		title(xlab="Model Forecasted Probability of Chosen Class", mgp = c(4, 1, 0))
		if(legend.location[i]!="none"){
			legend(x = legend.location[i],cex=2,
		      ncol = 1, bty="n",
		      bg = "transparent",
		      legend = c("MNIST","KMNIST","Fashion MNIST","SVHN","CIFAR-10"),  # Legend texts
		      lty = c(1, 1, 1, 1, 1),           # Line types
		      col = c('red3', 'tan1', 'green', 'dodgerblue', 'violet'),           # Line colors
		      lwd = c(10,10,10,10,10))                 # Line width		
		}
	}
	dev.off()
}

gen.dlares <- function(results.table,variable,legend.location="bottomright"){

	get.vec <- function(data.name,variable,modelname,drate){
		vals <- rep(as.numeric(NA),9)
		for(v in 1:9){
			row <- results.table[data==data.name & model==modelname & ds.rate==drate & pcat==v]
			if(nrow(row)==1){
				vals[v] <- row[,get(variable)]
			}
		}
		return(vals)
	}

	if(length(legend.location)==1){
		legend.location <- rep(legend.location,5)
	}

	if(variable=="actual"){
		graph.ylabel = "% Test Cases Correct"
	} else if(variable=="count"){
		graph.ylabel = "% of Test Sample"
	}

	graph.name <- paste0("dlares_",variable,"_pcats.svg")

	yticks_val <- pretty_breaks(n=5)(c(0,1))

	x.labels <- c("\n\n\u2264 20%",
		"\n\n20% -\n30%",
		"\n\n30% -\n40%",
		"\n\n40% -\n50%",
		"\n\n50% -\n60%",
		"\n\n60% -\n70%",
		"\n\n70% -\n80%",
		"\n\n80% -\n90%",">90%")

	pcats <- factor(x.labels,levels=x.labels)	

	image.sizes <- c("32x32","16x16","8x8","4x4","2x2")
	ds.rgb <- c(1,2,4,8,16)

	svg(graph.name,width=83,height=11)
	par(mfrow=c(1,5))
	for(i in 1:5){

		svhn.dla <- get.vec("svhn",variable,"dla",ds.rgb[i])
		svhn.res <- get.vec("svhn",variable,"res",ds.rgb[i])
		cifar.dla <- get.vec("cifar",variable,"dla",ds.rgb[i])
		cifar.res <- get.vec("cifar",variable,"res",ds.rgb[i])

		par(mar=c(6,6,5,2), mgp=c(5,2,0), cex=2.1, cex.main=4.0, cex.lab=2.0)
		ylims=c(0,1)
		matplot(pcats,svhn.dla,type="n",col="red3",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
		# rect(par("usr")[1], par("usr")[3],
		#     par("usr")[2], par("usr")[4],
		#     col = "white") # Color
		abline(h = seq(0.2,1.0,0.2), lty = "dashed", col = "gray30")
		lines(pcats,svhn.dla,col="dodgerblue",lwd=10,lty=1)
		lines(pcats,svhn.res,col="dodgerblue",lwd=10,lty=3)
		lines(pcats,cifar.dla,col="violet",lwd=10,lty=1)
		lines(pcats,cifar.res,col="violet",lwd=10,lty=3)
		axis(2, at=yticks_val, lab=percent(yticks_val), pos=1, lwd=0, las=1)
		axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
		title(main=image.sizes[i])
		title(ylab=graph.ylabel, mgp = c(3.5, 1, 0))
		title(xlab="Model Forecasted Probability of Chosen Class", mgp = c(4, 1, 0))
		if(legend.location[i]!="none"){
			legend(x = legend.location[i],cex=2,
		      ncol = 1, bty="n",
		      bg = "transparent",
		      legend = c("SVHN - DLA-Simple","SVHN - ResNet-18","CIFAR-10 - DLA-Simple","CIFAR-10 - ResNet-18"),  # Legend texts
		      lty = c(1, 3, 1, 3),           # Line types
		      col = c('dodgerblue', 'dodgerblue', 'violet','violet'),           # Line colors
		      lwd = c(10,10,10,10))                 # Line width		
		}
	}
	dev.off()
}




name.list <- c("mnist","kmnist","fashionmnist","svhn","cifar")
sample.list <- c("train","test")
model.list <- c("logit")
factor.list <- c(100,300)
ds.rates <- c(1,2,4,7,8,14,16)

linear.specs <- data.table(expand.grid(
	data = name.list,
	source = sample.list,
	model = model.list,
	factors = factor.list,
	ds.rate = ds.rates
	))

linear.specs <- linear.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
linear.specs <- linear.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
linear.specs <- linear.specs[!(grepl("mnist",data) & factors==300)]
linear.specs <- linear.specs[!(!grepl("mnist",data) & factors==100)]
linear.specs <- linear.specs[source=="test" | ds.rate==1]
linear.results <- lapply(1:nrow(linear.specs),get.linear)
linear.results <- rbindlist(linear.results)

logit.results <- linear.results[source=="test"]
scrap <- gen.graph.row(logit.results,"logistic","count",c("none","none","none","none","topright"))
scrap <- gen.graph.row(logit.results,"logistic","actual",c("none","none","none","none","topleft"))

lenet.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

lenet.specs <- lenet.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
lenet.specs <- lenet.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
lenet.specs <- lenet.specs[source=="test" | ds.rate==1]
lenet.results <- lapply(1:nrow(lenet.specs),get.lenet)
lenet.results <- rbindlist(lenet.results)

scrap <- gen.graph.row(lenet.results[source=="test"],"lenet","count",c("none","none","none","none","topleft"))
scrap <- gen.graph.row(lenet.results[source=="test"],"lenet","actual",c("none","none","none","none","topleft"))


name.list <- c("svhn","cifar")

dla.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

dla.specs <- dla.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
dla.specs <- dla.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
dla.specs <- dla.specs[source=="test" | ds.rate==1]
dla.results <- lapply(1:nrow(dla.specs),get.dla)
dla.results <- rbindlist(dla.results)


res.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

res.specs <- res.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
res.specs <- res.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
res.specs <- res.specs[source=="test" | ds.rate==1]
res.results <- lapply(1:nrow(res.specs),get.res)
res.results <- rbindlist(res.results)

dlares.results <- rbind(dla.results,res.results)
scrap <- gen.dlares(dlares.results[source=="test"],"count",c("none","none","none","none","topleft"))
scrap <- gen.dlares(dlares.results[source=="test"],"actual",c("none","none","none","none","topleft"))

results <- rbind(linear.results,lenet.results)
results[,factors:=as.character(factors)]
results[is.na(factors),factors:=""]
results <- results[factors!="500"]
results[grepl("mnist",data) & ds.rate==7,ds.rate:=8]
results[grepl("mnist",data) & ds.rate==14,ds.rate:=16]

results[,time.series:=paste0(model,factors," ",source," ",ds.rate)]
setkey(results,data,pcat)

proj <- spread(results[,list(time.series,data,pcat,proj)],time.series,proj)
actual <- spread(results[,list(time.series,data,pcat,actual)],time.series,actual)
count <- spread(results[,list(time.series,data,pcat,count)],time.series,count)

types <- c("train 1",paste0("test ",c(1,2,4,8,16)))
# mods <- c("logit2")
# cols <- c("data","pcat",paste("logit2",types))
# setcolorder(proj,cols)
# setcolorder(actual,cols)
# setcolorder(count,cols)

write.table(proj,"phat.txt",sep='\t',row.names=FALSE)
write.table(actual,"pactual.txt",sep='\t',row.names=FALSE)
write.table(count,"pcount.txt",sep='\t',row.names=FALSE)


