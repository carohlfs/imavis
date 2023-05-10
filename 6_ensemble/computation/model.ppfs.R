library(data.table)
library(tidyr)
library(scales)

gen.ppf.mnist <- function(){

	assign.var <- function(i){
		spec <- priorities[i]
		d <- spec[,dataset]
		c <- spec[,candidates]
		p <- spec[,phat]
		v <- spec[,variable]
		benchmark <- spec[,benchmark]
		pmax <- spec[,pmax]
		thresh <- as.numeric(spec[,list(`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
		bname <- paste(d,c,p,v,"bench",sep=".")
		mname <- paste(d,c,p,v,"pmax",sep=".")
		tname <- paste(d,c,p,v,"thresh",sep=".")
		assign(bname,benchmark,envir=globalenv())
		assign(mname,pmax,envir=globalenv())
		assign(tname,thresh,envir=globalenv())
	}

	priorities <- fread('priorities.small.txt')
	numerics <- c("benchmark","pmax","0.7","0.8","0.9","0.95","0.975","0.99")
	priorities[variable=="flops",(numerics):=lapply(.SD,`*`,1000),.SDcols=numerics]

	lapply(1:nrow(priorities),assign.var)

	graph.name <- paste0("mnist_ppf.svg")

	yticks_val <- pretty_breaks(n=4)(c(0.85,1))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0.85,1)
	xlims=c(0,0.5)	
	matplot(mnist.full.unadj.flops.thresh,mnist.full.unadj.accuracy.thresh,type="n",col="red3",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.8,0.8),lty="dashed",col="gray30")
	lines(xlims,c(0.85,0.85),lty="dashed",col="gray30")
	lines(xlims,c(0.9,0.9),lty="dashed",col="gray30")
	lines(xlims,c(0.95,0.95),lty="dashed",col="gray30")	
	lines(mnist.full.unadj.flops.thresh,mnist.full.unadj.accuracy.thresh,col="red3",lwd=10,lty=1)
	lines(kmnist.full.unadj.flops.thresh,kmnist.full.unadj.accuracy.thresh,col="tan1",lwd=10,lty=1)
	lines(fashionmnist.full.unadj.flops.thresh,fashionmnist.full.unadj.accuracy.thresh,col="green",lwd=10,lty=1)

	points(mnist.full.unadj.flops.bench,mnist.full.unadj.accuracy.bench,col="red3",pch=8,cex=2.0)
	points(kmnist.full.unadj.flops.bench,kmnist.full.unadj.accuracy.bench,col="tan1",pch=8,cex=2.0)
	points(fashionmnist.full.unadj.flops.bench,fashionmnist.full.unadj.accuracy.bench,col="green",pch=8,cex=2.0)
	points(mnist.full.unadj.flops.pmax,mnist.full.unadj.accuracy.pmax,col="red3",pch=9,cex=2.0)
	points(kmnist.full.unadj.flops.pmax,kmnist.full.unadj.accuracy.pmax,col="tan1",pch=9,cex=2.0)
	points(fashionmnist.full.unadj.flops.pmax,fashionmnist.full.unadj.accuracy.pmax,col="green",pch=9,cex=2.0)

	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="f", digits=1))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=0, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","",""), lwd=1)
	title(ylab="% Test Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="MMACs", mgp = c(4.0, 1, 0))
	legend(x = "topright",
      ncol = 2,cex=1.6,inset=c(-0.82,-0.10),y.intersp=2,
      bg = "transparent",bty="n",pt.cex=2,
      legend = c("MNIST \n- Benchmark","KMNIST \n- Benchmark","Fashion MNIST \n- Benchmark",
      	"MNIST -\nThreshold-based","Fashion MNIST -\nThreshold-based","MNIST \n- Max Prob",
      	"KMNIST \n- Max Prob","Fashion MNIST \n- Max Prob","KMNIST -\nThreshold-based")
  ,  # Legend texts
      lty = c(NA, NA, NA, 1, 1, NA, NA,NA,1),           # Line types
      pch = c(8,8,8,NA,NA,9,9,9,NA),
      col = c('red3','tan1','green','red3','green','red3','tan1','green','tan1'),           # Line colors
	  lwd = c(NA,NA,NA,10,10,NA,NA,NA,10))
	dev.off()	
}


gen.ppf.rgb <- function(){

	assign.var <- function(i){
		spec <- priorities[i]
		d <- spec[,dataset]
		c <- spec[,candidates]
		p <- spec[,phat]
		v <- spec[,variable]
		benchmark <- spec[,benchmark]
		pmax <- spec[,pmax]
		thresh <- as.numeric(spec[,list(`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
		bname <- paste(d,c,p,v,"bench",sep=".")
		mname <- paste(d,c,p,v,"pmax",sep=".")
		tname <- paste(d,c,p,v,"thresh",sep=".")
		assign(bname,benchmark,envir=globalenv())
		assign(mname,pmax,envir=globalenv())
		assign(tname,thresh,envir=globalenv())
	}

	priorities <- fread('priorities.small.txt')

	lapply(1:nrow(priorities),assign.var)

	graph.name <- paste0("rgb_ppf.svg")

	yticks_val <- pretty_breaks(n=4)(c(0.85,1))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0.80,1)
	xlims=c(0,1.5)	
	matplot(svhn.full.unadj.flops.thresh,svhn.full.unadj.accuracy.thresh,type="n",col="dodgerblue",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.8,0.8),lty="dashed",col="gray30")
	lines(xlims,c(0.85,0.85),lty="dashed",col="gray30")
	lines(xlims,c(0.9,0.9),lty="dashed",col="gray30")
	lines(xlims,c(0.95,0.95),lty="dashed",col="gray30")	
	lines(svhn.full.unadj.flops.thresh,svhn.full.unadj.accuracy.thresh,col="dodgerblue",lwd=10,lty=1)
	lines(cifar.full.unadj.flops.thresh,cifar.full.unadj.accuracy.thresh,col="violet",lwd=10,lty=1)

	points(svhn.full.unadj.flops.bench,svhn.full.unadj.accuracy.bench,col="dodgerblue",pch=8,cex=2.0)
	points(cifar.full.unadj.flops.bench,cifar.full.unadj.accuracy.bench,col="violet",pch=8,cex=2.0)
	points(svhn.full.unadj.flops.pmax,svhn.full.unadj.accuracy.pmax,col="dodgerblue",pch=9,cex=2.0)
	points(cifar.full.unadj.flops.pmax,cifar.full.unadj.accuracy.pmax,col="violet",pch=9,cex=2.0)

	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="f", digits=1))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=0, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","",""), lwd=1)
	title(ylab="% Test Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="GMACs", mgp = c(4.0, 1, 0))
	legend(x = "topright",
       ncol = 2,cex=1.6,inset=c(-0.82,-0.10),y.intersp=2,
       bg = "transparent",bty="n",pt.cex=2,
       legend = c("SVHN \n- Benchmark","SVHN \n- Max Prob","SVHN -\nThreshold-based",
       	"CIFAR \n- Benchmark","CIFAR -\nThreshold-based","CIFAR \n- Max Prob")
   ,  # Legend texts
       lty = c(NA, NA, 1, NA, NA,1),           # Line types
       pch = c(8,9,NA,8,9,NA),
       col = c('dodgerblue','dodgerblue','dodgerblue','violet','violet','violet','violet'),           # Line colors
	   lwd = c(NA,NA,10,NA,NA,10))
	dev.off()	
}



gen.gmacs.ppf.inet <- function(){

	assign.var <- function(i){
		spec <- priorities[i]
		s <- spec[,sample]
		c <- spec[,candidates]
		v <- spec[,variable]
		benchmark <- spec[,benchmark]
		pmax <- spec[,pmax]
		thresh <- as.numeric(spec[,list(`0.5`,`0.6`,`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
		bname <- paste(s,c,v,"bench",sep=".")
		mname <- paste(s,c,v,"pmax",sep=".")
		tname <- paste(s,c,v,"thresh",sep=".")
		assign(bname,benchmark,envir=globalenv())
		assign(mname,pmax,envir=globalenv())
		assign(tname,thresh,envir=globalenv())
	}

	priorities <- fread('priorities.inet.txt')
	lapply(1:nrow(priorities),assign.var)

	graph.name <- paste0("inet_gmac_ppf.svg")

	yticks_val <- pretty_breaks(n=5)(c(0.60,0.85))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0.60,0.85)
	xlims=c(0,50)	
	matplot(valid.all1.flops.thresh,valid.all1.accuracy.thresh,type="n",col="red3",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.65,0.65),lty="dashed",col="gray30")
	lines(xlims,c(0.7,0.7),lty="dashed",col="gray30")
	lines(xlims,c(0.75,0.75),lty="dashed",col="gray30")
	lines(xlims,c(0.80,0.80),lty="dashed",col="gray30")
	lines(valid.all1.flops.thresh,valid.all1.accuracy.thresh,col="red3",lwd=10,lty=1)
	lines(valid.select2.flops.thresh,valid.select2.accuracy.thresh,col="green",lwd=10,lty=1)
	lines(valid.select4.flops.thresh,valid.select4.accuracy.thresh,col="violet",lwd=10,lty=1)

	lines(test.all1.flops.thresh,test.all1.accuracy.thresh,col="red3",lwd=10,lty=3)
	lines(test.select2.flops.thresh,test.select2.accuracy.thresh,col="green",lwd=10,lty=3)
	lines(test.select4.flops.thresh,test.select4.accuracy.thresh,col="violet",lwd=10,lty=3)

	points(valid.all1.flops.bench,valid.all1.accuracy.bench,col="dodgerblue",pch=8,cex=2.0)
	points(test.all1.flops.bench,test.all1.accuracy.bench,col="violet",pch=8,cex=2.0)
	points(valid.select4.flops.pmax,valid.select4.accuracy.pmax,col="dodgerblue",pch=9,cex=2.0)
	points(test.select4.flops.pmax,test.select4.accuracy.pmax,col="violet",pch=9,cex=2.0)

	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="d", digits=0))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=0, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="% Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="GMACs", mgp = c(4.0, 1, 0))
	 legend(x = "topright",
        ncol = 2,cex=1.4,inset=c(-0.75,-0.17),y.intersp=2.0,x.intersp=2.0,
        bg = "transparent",bty="n",pt.cex=2,
        legend = c("ImageNet \n- Benchmark","ImageNet \n- Max Prob","ImageNet -\nThreshold-\nbased - All",
        	"ImageNet -\nThreshold-\nbased - Set 1","ImageNet -\nThreshold-\nbased - Set 2",

			"ImageNet-V2 \n- Benchmark","ImageNet-V2 \n- Max Prob","ImageNet-V2 -\nThreshold-\nbased - All",
        	"ImageNet-V2 -\nThreshold-\nbased - Set 1","ImageNet-V2 -\nThreshold-\nbased - Set 2")
    ,  # Legend texts
        lty = c(NA, NA, 1, 1, 1, NA, NA,3, 3, 3),           # Line types
        pch = c(8,9,NA,NA,NA,8,9,NA,NA,NA),
        col = c('dodgerblue','dodgerblue','red3','green','violet','violet','violet','red3','green','violet'),           # Line colors
	    lwd = c(NA,NA,10,10,10,NA,NA,10,10,10))
	dev.off()	
}


gen.msecs.ppf.inet <- function(){

	assign.var <- function(i){
		spec <- priorities[i]
		s <- spec[,sample]
		c <- spec[,candidates]
		v <- spec[,variable]
		benchmark <- spec[,benchmark]
		pmax <- spec[,pmax]
		thresh <- as.numeric(spec[,list(`0.5`,`0.6`,`0.7`,`0.8`,`0.9`,`0.95`,`0.975`,`0.99`)])
		bname <- paste(s,c,v,"bench",sep=".")
		mname <- paste(s,c,v,"pmax",sep=".")
		tname <- paste(s,c,v,"thresh",sep=".")
		assign(bname,benchmark,envir=globalenv())
		assign(mname,pmax,envir=globalenv())
		assign(tname,thresh,envir=globalenv())
	}

	priorities <- fread('priorities.inet.txt')
	lapply(1:nrow(priorities),assign.var)

	graph.name <- paste0("inet_msec_ppf.svg")

	yticks_val <- pretty_breaks(n=6)(c(0.55,0.85))

	svg(graph.name,width=31,height=11)

	par(mar=c(5.5,7.5,2,30), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0.55,0.85)
	xlims=c(0,5)	
	matplot(valid.all2.msecs.thresh,valid.all2.accuracy.thresh,type="n",col="dodgerblue",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.60,0.60),lty="dashed",col="gray30")
	lines(xlims,c(0.65,0.65),lty="dashed",col="gray30")
	lines(xlims,c(0.7,0.7),lty="dashed",col="gray30")
	lines(xlims,c(0.75,0.75),lty="dashed",col="gray30")
	lines(xlims,c(0.80,0.80),lty="dashed",col="gray30")
	lines(valid.all2.msecs.thresh,valid.all2.accuracy.thresh,col="red3",lwd=10,lty=1)
	lines(valid.select1.msecs.thresh,valid.select1.accuracy.thresh,col="green",lwd=10,lty=1)
	lines(valid.select3.msecs.thresh,valid.select3.accuracy.thresh,col="dodgerblue",lwd=10,lty=1)

	lines(test.all2.msecs.thresh,test.all2.accuracy.thresh,col="red3",lwd=10,lty=3)
	lines(test.select1.msecs.thresh,test.select1.accuracy.thresh,col="green",lwd=10,lty=3)
	lines(test.select3.msecs.thresh,test.select3.accuracy.thresh,col="dodgerblue",lwd=10,lty=3)

	points(valid.all2.msecs.bench,valid.all2.accuracy.bench,col="dodgerblue",pch=8,cex=2.0)
	points(test.all2.msecs.bench,test.all2.accuracy.bench,col="violet",pch=8,cex=2.0)
	points(valid.select3.msecs.pmax,valid.select3.accuracy.pmax,col="dodgerblue",pch=9,cex=2.0)
	points(test.select3.msecs.pmax,test.select3.accuracy.pmax,col="violet",pch=9,cex=2.0)

	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="d", digits=0))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=0, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","","",""), lwd=1)
	title(ylab="% Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="Miliseconds", mgp = c(4.0, 1, 0))
	 legend(x = "topright",
        ncol = 2,cex=1.65,inset=c(-0.82,-0.17),y.intersp=1.7,x.intersp=1.5,
        bg = "transparent",bty="n",pt.cex=2,
        legend = c("ImageNet \n- Benchmark","ImageNet \n- Max Prob","ImageNet -\nThreshold-\nbased - All",
        	"ImageNet -\nThreshold-\nbased - Set 3",

			"ImageNet-V2 \n- Benchmark","ImageNet-V2 \n- Max Prob","ImageNet-V2 -\nThreshold-\nbased - All",
        	"ImageNet-V2 -\nThreshold-\nbased - Set 3")
    ,  # Legend texts
        lty = c(NA, NA, 1, 1, NA, NA,3, 3),           # Line types
        pch = c(8,9,NA,NA,8,9,NA,NA),
        col = c('dodgerblue','dodgerblue','red3','dodgerblue','violet','violet','red3','dodgerblue'),           # Line colors
	    lwd = c(NA,NA,10,10,NA,NA,10,10))
	dev.off()	
}

gen.ppf.mnist()
gen.ppf.rgb()
gen.gmacs.ppf.inet()
gen.msecs.ppf.inet()

