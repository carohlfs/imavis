library(data.table)
library(tidyr)
library(scales)
library(svglite)

argpmax <- function(idx,dataset){
	return(which.max(as.numeric(dataset[idx]))-1)
}

switching.linear <- function(i){
	dataname <- spec.list[i,data.name]
	modeltype <- spec.list[i,model]
	orderidx <- spec.list[i,ordering]
	if(grepl("mnist",dataname)){
		order <- order.mnist[[orderidx]]
		orderall <- sort(unique(unlist(order.mnist)))
		factors <- 100
	} else {
		order <- order.rgb[[orderidx]]
		orderall <- sort(unique(unlist(order.rgb)))
		factors <- 300
	}
	threshold <- spec.list[i,threshold]
	probtype <- ifelse(grepl("linear",modeltype),"_test_probs","_test_lprobs")
	filenames <- paste0("/Volumes/T7/effort/probs/",dataname,order,probtype,factors,".RDS")
	data <- lapply(filenames,readRDS)
	rows <- sapply(data,nrow)
	order.vec <- rep(order,rows)
	data <- rbindlist(data)
	data[,ds.rate:=order.vec]
	data[,id:=seq(1:nrow(data))]
	data[,minid:=min(id),by=ds.rate]
	data[,id:=id-minid+1]
	data[,c("minid"):=NULL]
	phats <- data[,list(id,phat,ds.rate)]
	phats <- dcast(phats,id ~ ds.rate,value.var="phat")
	setnames(phats,c("id",paste0("phat",rev(order))))
	yhats <- data[,list(id,yhat,ds.rate)]
	yhats <- dcast(yhats,id ~ ds.rate,value.var="yhat")
	setnames(yhats,c("id",paste0("yhat",rev(order))))
	yhats[,id:=NULL]
	phats <- cbind(phats,yhats)
	phatvars <- paste0("phat",rev(order))
	usedvars <- paste0("used",rev(order))
	allorders <- paste0("used",orderall)
	phats[,(allorders):=0]
	phats[,(usedvars):=1]
	phats[,choice:=phatvars[1]]
	j <- 2
	while(j<=length(phatvars)){
		phats[get(phatvars[j])>threshold & choice==phatvars[j-1],choice:=phatvars[j]]
		phats[choice==phatvars[j],(usedvars[j-1]):=0]
		j <- j+1
	}
	for(p in phatvars){
		phats[choice==p,prob:=get(p)]
		phats[choice==p,class:=get(gsub("p","y",p))]
	}
	yactual <- data[ds.rate==1,list(y)]
	phats <- cbind(phats,yactual)
	phats[,correct:= as.numeric(round(class)==round(y))]
	phats[,c(phatvars,gsub("phat","yhat",phatvars),"class","choice","y"):=NULL]
	phats <- melt(phats,id="id")
	phats <- phats[,mean(value),by=variable]
	phats[,id:=i]
	phats <- spread(phats,variable,V1)
	if(grepl("mnist",dataname)){
		setnames(phats,"used7","used8")
		setnames(phats,"used14","used16")
	}
	setcolorder(phats,rev(names(phats)))
	phats[,id:=NULL]
	phats <- cbind(spec.list[i],phats)
	return(phats)
}

switching.neural <- function(i){
	dataname <- neural.specs[i,data.name]
	modeltype <- neural.specs[i,model]
	orderidx <- neural.specs[i,ordering]
	if(grepl("mnist",dataname)){
		order <- order.mnist[[orderidx]]
		orderall <- sort(unique(unlist(order.mnist)))
	} else {
		order <- order.rgb[[orderidx]]
		orderall <- sort(unique(unlist(order.rgb)))
	}
	threshold <- neural.specs[i,threshold]
	if(modeltype=="lenet"){
		show.order <- as.character(order)
		show.order[show.order=="1"] <- ""
		filenames <- paste0("../",dataname,show.order,"_test_probs.csv")
	} else if(modeltype=="dla"){
		filenames<- paste0('../pytorch-',dataname,"-master/",dataname,order,"_test_dlaprobs.csv")
	} else if(modeltype=="resnet"){
		filenames<- paste0('../pytorch-',dataname,"-resnet/",dataname,order,"_test_resprobs.csv")
	}
	data <- lapply(filenames,fread)
	rows <- sapply(data,nrow)
	order.vec <- rep(order,rows)
	data <- rbindlist(data)
	data[,ds.rate:=order.vec]
	data[,id:=seq(1:nrow(data))]
	data[,minid:=min(id),by=ds.rate]
	data[,id:=id-minid+1]
	data[,minid:=NULL]
	Vs <- paste0("V",1:10)
	preds <- sapply(1:nrow(data),argpmax,data[,mget(Vs)])
	if(modeltype!="lenet"){
		data[,(Vs):=lapply(.SD,exp),.SDcols=Vs]
		data[,total:=V1+V2+V3+V4+V5+V6+V7+V8+V9+V10]
		data[,(Vs):=lapply(.SD,`*`,1/total),.SDcols=Vs]
	}
	data[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
	data[,yhat:=preds]
	phats <- data[,list(id,phat,ds.rate)]
	phats <- dcast(phats,id ~ ds.rate,value.var="phat")
	setnames(phats,c("id",paste0("phat",rev(order))))
	yhats <- data[,list(id,yhat,ds.rate)]
	yhats <- dcast(yhats,id ~ ds.rate,value.var="yhat")
	setnames(yhats,c("id",paste0("yhat",rev(order))))
	yhats[,id:=NULL]
	phats <- cbind(phats,yhats)
	phatvars <- paste0("phat",rev(order))
	usedvars <- paste0("used",rev(order))
	allorders <- paste0("used",orderall)
	phats[,(allorders):=0]
	phats[,(usedvars):=1]
	phats[,choice:=phatvars[1]]
	j <- 2
	while(j<=length(phatvars)){
		phats[get(phatvars[j])>threshold & choice==phatvars[j-1],choice:=phatvars[j]]
		phats[choice==phatvars[j],(usedvars[j-1]):=0]
		j <- j+1
	}
	for(p in phatvars){
		phats[choice==p,prob:=get(p)]
		phats[choice==p,class:=get(gsub("p","y",p))]
	}
	if(modeltype=="lenet"){
		actuals.file <- paste0('../',dataname,"_test_actuals.csv")
	} else if(modeltype=="dla"){
		actuals.file <- paste0('../pytorch-',dataname,"-master/",dataname,"_test_dlaactuals.csv")
	} else if(modeltype=="resnet"){
		actuals.file <- paste0('../pytorch-',dataname,"-resnet/",dataname,"_test_resactuals.csv")
	}
	yactual <- fread(actuals.file)
	setnames(yactual,"y")
	phats <- cbind(phats,yactual)
	phats[,correct:= as.numeric(round(class)==round(y))]
	phats[,c(phatvars,gsub("phat","yhat",phatvars),"class","choice","y"):=NULL]
	phats <- melt(phats,id="id")
	phats <- phats[,mean(value),by=variable]
	phats[,id:=i]
	phats <- spread(phats,variable,V1)
	if(grepl("mnist",dataname)){
		setnames(phats,"used7","used8")
		setnames(phats,"used14","used16")
	}
	setcolorder(phats,rev(names(phats)))
	phats[,id:=NULL]
	phats <- cbind(neural.specs[i],phats)
	return(phats)
}

gen.ppf <- function(results.table,benchmark.table,model){

	mnist.bytes <- results.table[data.name=="mnist",bytes]
	mnist.correct <- results.table[data.name=="mnist",correct]
	kmnist.bytes <- results.table[data.name=="kmnist",bytes]
	kmnist.correct <- results.table[data.name=="kmnist",correct]
	fashionmnist.bytes <- results.table[data.name=="fashionmnist",bytes]
	fashionmnist.correct <- results.table[data.name=="fashionmnist",correct]
	svhn.bytes <- results.table[data.name=="svhn",bytes]
	svhn.correct <- results.table[data.name=="svhn",correct]
	cifar.bytes <- results.table[data.name=="cifar",bytes]
	cifar.correct <- results.table[data.name=="cifar",correct]

	mnist.benchmark.bytes <- benchmark.table[data.name=="mnist",bytes]
	mnist.benchmark.correct <- benchmark.table[data.name=="mnist",correct]
	kmnist.benchmark.bytes <- benchmark.table[data.name=="kmnist",bytes]
	kmnist.benchmark.correct <- benchmark.table[data.name=="kmnist",correct]
	fashionmnist.benchmark.bytes <- benchmark.table[data.name=="fashionmnist",bytes]
	fashionmnist.benchmark.correct <- benchmark.table[data.name=="fashionmnist",correct]
	svhn.benchmark.bytes <- benchmark.table[data.name=="svhn",bytes]
	svhn.benchmark.correct <- benchmark.table[data.name=="svhn",correct]
	cifar.benchmark.bytes <- benchmark.table[data.name=="cifar",bytes]
	cifar.benchmark.correct <- benchmark.table[data.name=="cifar",correct]

	graph.name <- paste0(model,"_ppf.svg")

	yticks_val <- pretty_breaks(n=5)(c(0,1))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0,1)
	if(model=="lenet"){
		xlims <- c(0,1600)
	} else {
		xlims <- c(0,4200)
	}
	matplot(mnist.bytes,mnist.correct,type="n",col="red3",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.2,0.2),lty="dashed",col="gray30")
	lines(xlims,c(0.4,0.4),lty="dashed",col="gray30")
	lines(xlims,c(0.6,0.6),lty="dashed",col="gray30")
	lines(xlims,c(0.8,0.8),lty="dashed",col="gray30")
	lines(mnist.bytes,mnist.correct,col="red3",lwd=10,lty=1)
	lines(kmnist.bytes,kmnist.correct,col="tan1",lwd=10,lty=1)
	lines(fashionmnist.bytes,fashionmnist.correct,col="green",lwd=10,lty=1)
	lines(svhn.bytes,svhn.correct,col="dodgerblue",lwd=10,lty=1)
	lines(cifar.bytes,cifar.correct,col="violet",lwd=10,lty=1)
	points(mnist.benchmark.bytes,mnist.benchmark.correct,col="red3",pch=8,cex=2.0)
	points(kmnist.benchmark.bytes,kmnist.benchmark.correct,col="tan1",pch=8,cex=2.0)
	points(fashionmnist.benchmark.bytes,fashionmnist.benchmark.correct,col="green",pch=8,cex=2.0)
	points(svhn.benchmark.bytes,svhn.benchmark.correct,col="dodgerblue",pch=8,cex=2.0)
	points(cifar.benchmark.bytes,cifar.benchmark.correct,col="violet",pch=8,cex=2.0)
	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="d", big.mark=','))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="% Test Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="Bytes Read per Test Case", mgp = c(4.0, 1, 0))
	legend(x = "topright",
      ncol = 2,cex=1.6,inset=c(-0.82,-0.10),y.intersp=2,
      bg = "transparent",bty="n",pt.cex=2,
      legend = c("MNIST \n- Benchmark","KMNIST \n- Benchmark","Fashion MNIST \n- Benchmark","SVHN \n- Benchmark","CIFAR-10 \n- Benchmark",
      	"MNIST -\nThreshold-based","KMNIST -\nThreshold-based","Fashion MNIST -\nThreshold-based","SVHN -\nThreshold-based","CIFAR-10 -\nThreshold-based")
  ,  # Legend texts
      lty = c(NA, NA, NA, NA, NA,1,1,1,1,1),           # Line types
      pch = c(8,8,8,8,8,NA,NA,NA,NA,NA),
      col = c('red3','tan1','green','dodgerblue','violet','red3','tan1','green','dodgerblue','violet'),           # Line colors
	  lwd = c(NA,NA,NA,NA,NA,10,10,10,10,10))
	dev.off()
}

gen.dlares.ppf <- function(results.table,benchmark.table){

	svhn.dla.bytes <- results.table[data.name=="svhn" & model=="dla",bytes]
	svhn.dla.correct <- results.table[data.name=="svhn" & model=="dla",correct]
	svhn.res.bytes <- results.table[data.name=="svhn" & model=="resnet",bytes]
	svhn.res.correct <- results.table[data.name=="svhn" & model=="resnet",correct]
	cifar.dla.bytes <- results.table[data.name=="cifar" & model=="dla",bytes]
	cifar.dla.correct <- results.table[data.name=="cifar" & model=="dla",correct]
	cifar.res.bytes <- results.table[data.name=="cifar" & model=="resnet",bytes]
	cifar.res.correct <- results.table[data.name=="cifar" & model=="resnet",correct]

	svhn.dla.benchmark.bytes <- benchmark.table[data.name=="svhn" & model=="dla",bytes]
	svhn.dla.benchmark.correct <- benchmark.table[data.name=="svhn" & model=="dla",correct]
	svhn.res.benchmark.bytes <- benchmark.table[data.name=="svhn" & model=="resnet",bytes]
	svhn.res.benchmark.correct <- benchmark.table[data.name=="svhn" & model=="resnet",correct]
	cifar.dla.benchmark.bytes <- benchmark.table[data.name=="cifar" & model=="dla",bytes]
	cifar.dla.benchmark.correct <- benchmark.table[data.name=="cifar" & model=="dla",correct]
	cifar.res.benchmark.bytes <- benchmark.table[data.name=="cifar" & model=="resnet",bytes]
	cifar.res.benchmark.correct <- benchmark.table[data.name=="cifar" & model=="resnet",correct]

	graph.name <- paste0("dlares_ppf.svg")

	yticks_val <- pretty_breaks(n=5)(c(0,1))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0,1)
	xlims <- c(0,4200)
	matplot(svhn.dla.bytes,svhn.dla.correct,type="n",col="dodgerblue",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.2,0.2),lty="dashed",col="gray30")
	lines(xlims,c(0.4,0.4),lty="dashed",col="gray30")
	lines(xlims,c(0.6,0.6),lty="dashed",col="gray30")
	lines(xlims,c(0.8,0.8),lty="dashed",col="gray30")
	lines(svhn.dla.bytes,svhn.dla.correct,col="dodgerblue",lwd=10,lty=1)
	lines(svhn.res.bytes,svhn.res.correct,col="dodgerblue",lwd=10,lty=3)
	lines(cifar.dla.bytes,cifar.dla.correct,col="violet",lwd=10,lty=1)
	lines(cifar.res.bytes,cifar.res.correct,col="violet",lwd=10,lty=3)
	points(svhn.dla.benchmark.bytes,svhn.dla.benchmark.correct,col="dodgerblue",pch=8,cex=2.0)
	points(svhn.res.benchmark.bytes,svhn.res.benchmark.correct,col="dodgerblue",pch=9,cex=2.0)
	points(cifar.dla.benchmark.bytes,cifar.dla.benchmark.correct,col="violet",pch=8,cex=2.0)
	points(cifar.res.benchmark.bytes,cifar.res.benchmark.correct,col="violet",pch=9,cex=2.0)
	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="d", big.mark=','))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="% Test Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="Bytes Read per Test Case", mgp = c(4.0, 1, 0))
	legend(x = "topright",
       ncol = 2,cex=1.6,inset=c(-0.82,-0.15),y.intersp=2,
      bg = "transparent",bty="n",pt.cex=2.0,
      legend = c("SVHN -\nDLA-Simple -\nBenchmark","SVHN -\nResNet-18 -\nBenchmark",
		"CIFAR-10 -\nDLA-Simple -\nBenchmark","CIFAR-10 -\nResNet-18 -\nBenchmark",
		"SVHN -\nDLA-Simple -\nThreshold-based","SVHN -\nResNet-18 -\nThreshold-based",
		"CIFAR-10 -\nDLA-Simple -\nThreshold-based","CIFAR-10 -\nResNet-18 -\nThreshold-based")
  ,  # Legend texts
      lty = c(NA,NA,NA,NA,1,3,1,3),           # Line types
      pch = c(8,9,8,9,NA,NA,NA,NA),
      col = c('dodgerblue','dodgerblue','violet','violet','dodgerblue','dodgerblue','violet','violet'),           # Line colors
	  lwd = c(NA,NA,NA,NA,10,10,10,10))
	dev.off()
}

name.list <- c("mnist","kmnist","fashionmnist","svhn","cifar")

models <- c("logit")

benchmark <- c(1)
order1 <- c(16,8,4,2,1)
order2 <- c(8,4,2,1)
order3 <- c(4,2,1)
order.rgb <- list(benchmark,order1,order2,order3)
order4 <- c(14,7,4,2,1)
order5 <- c(7,4,2,1)
order.mnist <- list(benchmark,order4,order5,order3)

thresholds <- c(0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,0.999)

spec.list <- data.table(expand.grid(
	data.name = name.list,
	model = models,
	ordering = 1:4,
	threshold = thresholds
	))

results <- NULL
for(i in 1:nrow(spec.list)){
	results[[i]] <- switching.linear(i)
}
results <- rbindlist(results)

results[grepl("mnist",data.name),bytes:=784*used1 + (784/4)*used2 + (784/16)*used4 + (784/49)*used8 + (784/196)*used16]
results[!grepl("mnist",data.name),bytes:=3072*used1 + (3072/4)*used2 + (3072/16)*used4 + (3072/64)*used8 + (3072/256)*used16]
results[,c("used1","used2","used4","used8","used16"):=NULL]

ppf.table <- results[ordering==4]
ppf.benchmark <- results[ordering==1 & threshold==0.8]
scrap <- gen.ppf(ppf.table,ppf.benchmark,"logistic")

name.list <- c("mnist","kmnist","fashionmnist","svhn","cifar")

models <- c("lenet","dla","resnet")

neural.specs <- data.table(expand.grid(
	data.name = name.list,
	model = models,
	ordering = 1:4,
	threshold = thresholds
	))

neural.specs <- neural.specs[model=="lenet" | data.name %in% c("svhn","cifar")]

neural.results <- NULL
for(i in 1:nrow(neural.specs)){
	neural.results[[i]] <- switching.neural(i)
}
neural.results <- rbindlist(neural.results)
neural.results[grepl("mnist",data.name),bytes:=784*used1 + (784/4)*used2 + (784/16)*used4 + (784/49)*used8 + (784/196)*used16]
neural.results[!grepl("mnist",data.name),bytes:=3072*used1 + (3072/4)*used2 + (3072/16)*used4 + (3072/64)*used8 + (3072/256)*used16]
neural.results[!grepl("mnist",data.name) & model=="lenet",bytes:=1024*used1 + (1024/4)*used2 + (1024/16)*used4 + (1024/64)*used8 + (1024/256)*used16]

neural.results[,c("used1","used2","used4","used8","used16"):=NULL]

setkey(neural.results,data.name,model,ordering,threshold)

lenet.results <- neural.results[model=="lenet" & ordering==4]
lenet.benchmark <- neural.results[model=="lenet" & ordering==1 & threshold==0.8]
scrap <- gen.ppf(lenet.results,lenet.benchmark,"lenet")

dlares.results <- neural.results[model!="lenet" & ordering==4]
dlares.benchmark <- neural.results[model!="lenet" & ordering==1 & threshold==0.8]
scrap <- gen.dlares.ppf(dlares.results,dlares.benchmark)
