library(data.table)
library(tidyr)
library(scales)
library(svglite)

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
}

linear.correl <- function(i){
	dataname <- linear.specs[i,data]
	pullsource <- linear.specs[i,source]
	modeltype <- linear.specs[i,model]
	factors <- linear.specs[i,factors]
	ds.rate <- linear.specs[i,ds.rate]
	probtype <- ifelse(modeltype=="linear","_probs","_lprobs")
	filename <- paste0("/Volumes/T7/effort/probs/",dataname,ds.rate,"_",pullsource,probtype,factors,".RDS")
	data <- readRDS(filename)
	data[,correct:= as.numeric(round(y)==round(yhat))]

	correl <- cor(data[,correct],data[,phat])
	linear.specs[i,correlation:=correl]
}

lenet.correl <- function(i){
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

	data[,correct:= as.numeric(round(V1)==round(proj))]
	
	correl <- cor(data[,correct],data[,phat])
	lenet.specs[i,correlation:=correl]
}

dla.correl <- function(i){
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
	data[,correct:= as.numeric(round(V1)==round(proj))]
	
	correl <- cor(data[,correct],data[,phat])
	dla.specs[i,correlation:=correl]
}

res.correl <- function(i){
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
	data[,correct:= as.numeric(round(V1)==round(proj))]
	
	correl <- cor(data[,correct],data[,phat])
	res.specs[i,correlation:=correl]
}


gen.correl.graph <- function(results.table,model,legend.location="topleft"){

	mnist <- results.table[data=="mnist",correlation]
	kmnist <- results.table[data=="kmnist",correlation]
	fashionmnist <- results.table[data=="fashionmnist",correlation]
	svhn <- results.table[data=="svhn",correlation]
	cifar <- results.table[data=="cifar",correlation]

	x.labels <- c("28x28/\n32x32","14x14/\n16x16","7x7/\n8x8","4x4","2x2")
	sizes <- factor(x.labels,levels=x.labels)

	graph.name <- paste0(model,"_correl.svg")

	yticks_val <- pretty_breaks(n=5)(c(-0.2,0.8))

	svg(graph.name,width=17,height=11)

	par(mar=c(7,10,1,1), mgp=c(5,2.5,0), cex=2.1, cex.lab=2.0, cex.axis=1.5)
	ylims=c(-0.2,0.8)
	matplot(sizes,mnist,type="n",col="red3",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
	abline(h = seq(0.0,0.8,0.2), lty = "dashed", col = "gray30")
	lines(sizes,mnist,col="red3",lwd=10,lty=1)
	lines(sizes,kmnist,col="tan1",lwd=10,lty=1)
	lines(sizes,fashionmnist,col="green",lwd=10,lty=1)
	lines(sizes,svhn,col="dodgerblue",lwd=10,lty=1)
	lines(sizes,cifar,col="violet",lwd=10,lty=1)
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="Correlation between\nPropensity and Accuracy", mgp = c(4.6, 1, 0))
	title(xlab="Size of Test Image", mgp = c(4.9, 1, 0))
	if(legend.location!="none"){
		legend(x = legend.location,
		      ncol = 1, bty="n",cex=1.5,
		      bg = "transparent",
		      legend = c("MNIST","KMNIST","Fashion MNIST","SVHN","CIFAR-10"),  # Legend texts
		      lty = c(1, 1, 1, 1, 1),           # Line types
		      col = c('red3', 'tan1', 'green', 'dodgerblue', 'violet'),           # Line colors
		      lwd = c(10,10,10,10,10))               # Line width		
	}
	dev.off()
}

gen.dlares.correl.graph <- function(results.table,legend.location="topleft"){

	svhn.dla <- results.table[data=="svhn" & model=="dla",correlation]
	svhn.res <- results.table[data=="svhn" & model=="resnet",correlation]
	cifar.dla <- results.table[data=="cifar" & model=="dla",correlation]
	cifar.res <- results.table[data=="cifar" & model=="resnet",correlation]

	x.labels <- c("32x32","16x16","8x8","4x4","2x2")
	sizes <- factor(x.labels,levels=x.labels)

	graph.name <- paste0("dlares_correl.svg")

	yticks_val <- pretty_breaks(n=5)(c(-0.2,0.8))

	svg(graph.name,width=17,height=11)

	par(mar=c(7,10,1,1), mgp=c(5,2.5,0), cex=2.1, cex.lab=2.0, cex.axis=1.5)
	ylims=c(-0.2,0.8)
	matplot(sizes,svhn.dla,type="n",col="dodgerblue",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
	abline(h = seq(0.0,0.8,0.2), lty = "dashed", col = "gray30")
	lines(sizes,svhn.dla,col="dodgerblue",lwd=10,lty=1)
	lines(sizes,svhn.res,col="dodgerblue",lwd=10,lty=3)
	lines(sizes,cifar.dla,col="violet",lwd=10,lty=1)
	lines(sizes,cifar.res,col="violet",lwd=10,lty=3)
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="Correlation between \nPropensity and Accuracy", mgp = c(4.6, 1, 0))
	title(xlab="Size of Test Image", mgp = c(4.9, 1, 0))
	if(legend.location!="none"){
		legend(x = legend.location,
		      ncol = 1, bty="n",cex=1.5,
		      bg = "transparent",
		      legend = c("SVHN - DLA-Simple","SVHN - ResNet-18","CIFAR-10 - DLA-Simple","CIFAR-10 - ResNet-18"),  # Legend texts
		      lty = c(1, 3, 1, 3),           # Line types
		      col = c('dodgerblue', 'dodgerblue', 'violet','violet'),           # Line colors
		      lwd = c(10,10,10,10))                 # Line width		
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
scrap <- lapply(1:nrow(linear.specs),linear.correl)

logit.correls <- linear.specs[source=="test"]
gen.correl.graph(logit.correls,"logistic","none")

lenet.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

lenet.specs <- lenet.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
lenet.specs <- lenet.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
lenet.specs <- lenet.specs[source=="test" | ds.rate==1]
scrap <- lapply(1:nrow(lenet.specs),lenet.correl)

lenet.correls <- lenet.specs[source=="test"]
gen.correl.graph(lenet.correls,"lenet","bottomleft")

name.list <- c("svhn","cifar")

dla.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

dla.specs <- dla.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
dla.specs <- dla.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
dla.specs <- dla.specs[source=="test" | ds.rate==1]
scrap <- lapply(1:nrow(dla.specs),dla.correl)


res.specs <- data.table(expand.grid(
	data=name.list,
	source=c("train","test"),
	ds.rate=c(1,2,4,7,8,14,16)
	))

res.specs <- res.specs[!(grepl("mnist",data) & ds.rate %in% c(8,16))]
res.specs <- res.specs[!(data %in% c("svhn","cifar") & ds.rate %in% c(7,14))]
res.specs <- res.specs[source=="test" | ds.rate==1]
scrap <- lapply(1:nrow(res.specs),res.correl)

dla.specs[,model:="dla"]
res.specs[,model:="resnet"]
dlares.specs <- rbind(dla.specs,res.specs)

dlares.correls <- dlares.specs[source=="test"]
gen.dlares.correl.graph(dlares.correls,"topright")
