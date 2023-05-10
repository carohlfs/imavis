library(data.table)
library(tidyr)

cleaned.data[,pcat:=floor(10*prob)]
cleaned.data[pcat==10,pcat:=9]
cleaned.data[,pcat:=paste0("p",pcat)]
cleaned.data[,count:=as.numeric(1)]
cleaned.data[sample=="valid",sample:="imagenet"]
cleaned.data[sample=="test",sample:="imagenetv2"]
vars <- paste0("p",0:9)
count.data <- cleaned.data[,sum(count),by=list(sample,size,model,pcat)]
count.data <- spread(count.data,pcat,V1)
count.data[,psum:=Reduce(`+`,mget(vars))]
count.data[,(vars):=lapply(.SD,`/`,psum),.SDcols=vars]
count.data[,psum:=NULL]

correct.data <- cleaned.data[,mean(correct),by=list(sample,size,model,pcat)]
correct.data <- spread(correct.data,pcat,V1)

gen.imagenet <- function(variable,source){

	get.vec <- function(variable,source,side,m){
		if(variable=="count"){
			graph.data <- copy(count.data)
		} else if(variable=="actual"){
			graph.data <- copy(correct.data)
		}
		graph.vec <- graph.data[sample==source & size==side & model==m,list(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9)]
		return(as.numeric(t(graph.vec)))
	}

	if(variable=="count"){
		graph.ylabel <- "% of Validation Sample"
	} else if(variable=="actual"){
		graph.ylabel = "% Validation Cases Correct"
	}

	graph.name <- paste0(source,"_",variable,"_pcats.svg")
	yticks_val <- pretty_breaks(n=5)(c(0,1))

	x.labels <- c("\n\n\u2264 10%",
		"\n\n10% -\n20%",
		"\n\n20% -\n30%",
		"\n\n30% -\n40%",
		"\n\n40% -\n50%",
		"\n\n50% -\n60%",
		"\n\n60% -\n70%",
		"\n\n70% -\n80%",
		"\n\n80% -\n90%",">90%")

	pcats <- factor(x.labels,levels=x.labels)	

	sides <- c(256,128,64,32,16)

	model.names <- c("ResNeXt101-32x8d","Wide ResNet-101-2","EfficientNet-b0","EfficientNet-b7",
		"ResNet-101","DenseNet-201","VGG-19-bn","MobileNet v3 large",
		"MobileNet v3 small","GoogLeNet","Inception v3","AlexNet")

	model.names <- factor(model.names,levels=model.names)

	svg(graph.name,width=83,height=11)
	par(mfrow=c(1,5))
	for(i in 1:5){
		resnext101_32x8d <- get.vec(variable,source,sides[i],"resnext101_32x8d")
		wide_resnet101_2 <- get.vec(variable,source,sides[i],"wide_resnet101_2")
		efficientnet_b0 <- get.vec(variable,source,sides[i],"efficientnet_b0")
		efficientnet_b7 <- get.vec(variable,source,sides[i],"efficientnet_b7")
		resnet101 <- get.vec(variable,source,sides[i],"resnet101")
		densenet201 <- get.vec(variable,source,sides[i],"densenet201")
		vgg19_bn <- get.vec(variable,source,sides[i],"vgg19_bn")
		mobilenet_v3_large <- get.vec(variable,source,sides[i],"mobilenet_v3_large")
		mobilenet_v3_small <- get.vec(variable,source,sides[i],"mobilenet_v3_small")
		googlenet <- get.vec(variable,source,sides[i],"googlenet")
		inception_v3 <- get.vec(variable,source,sides[i],"inception_v3")
		alexnet <- get.vec(variable,source,sides[i],"alexnet")

		par(mar=c(6,6,5,2), mgp=c(5,2,0), cex=2.1, cex.main=4.0, cex.lab=2.0)
		ylims=c(0,1)
		matplot(pcats,resnext101_32x8d,type="n",col="red3",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
		abline(h = seq(0.2,1.0,0.2), lty = "dashed", col = "gray30")
		lines(pcats,resnext101_32x8d,col="red3",lwd=10,lty=1)
		lines(pcats,wide_resnet101_2,col="tan1",lwd=10,lty=1)
		lines(pcats,efficientnet_b0,col="green",lwd=10,lty=1)
		lines(pcats,efficientnet_b7,col="dodgerblue",lwd=10,lty=1)
		lines(pcats,resnet101,col="violet",lwd=10,lty=1)
		lines(pcats,densenet201,col="burlywood4",lwd=10,lty=1)
		lines(pcats,vgg19_bn,col="red3",lwd=10,lty=3)
		lines(pcats,mobilenet_v3_large,col="tan1",lwd=10,lty=3)
		lines(pcats,mobilenet_v3_small,col="green",lwd=10,lty=3)
		lines(pcats,googlenet,col="dodgerblue",lwd=10,lty=3)
		lines(pcats,inception_v3,col="violet",lwd=10,lty=3)
		lines(pcats,alexnet,col="burlywood4",lwd=10,lty=3)
		axis(2, at=yticks_val, lab=percent(yticks_val), pos=1, lwd=0, las=1)
		axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
		title(main=sides[i])
		title(ylab=graph.ylabel, mgp = c(3.5, 1, 0))
		title(xlab="Model Forecasted Probability of Chosen Class", mgp = c(4, 1, 0))
		if(variable=="count"){
			if(i==1){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[1:3],  # Legend texts
				     lty = c(1, 1, 1),           # Line types
				     col = c('red3', 'tan1', 'green'), # Line colors
				     lwd = c(10,10,10))                 # Line width			
			} else if(i==2){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[4:6],  # Legend texts
				     lty = c(1, 1, 1),           # Line types
				     col = c('dodgerblue', 'violet', 'burlywood4'), # Line colors
				     lwd = c(10,10,10))     		
			} else if(i==3){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[7:9],  # Legend texts
				     lty = c(3, 3, 3),           # Line types
				     col = c('red3', 'tan1', 'green'), # Line colors
				     lwd = c(10,10,10))                 # Line width			
			} else if(i==4){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[10:12],  # Legend texts
				     lty = c(3, 3, 3),           # Line types
				     col = c('dodgerblue', 'violet', 'burlywood4'), # Line colors
				     lwd = c(10,10,10))     		
			}
		} else if(variable=="actual"){
			if(i==3){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[1:3],  # Legend texts
				     lty = c(1, 1, 1),           # Line types
				     col = c('red3', 'tan1', 'green'), # Line colors
				     lwd = c(10,10,10))                 # Line width			
			} else if(i==4){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[4:7],  # Legend texts
				     lty = c(1, 1, 1, 3),           # Line types
				     col = c('dodgerblue','violet', 'burlywood4', 'red3'), # Line colors
				     lwd = c(10,10,10,10))     		
			} else if(i==5){
				legend(x = "topleft",cex=2,
				     ncol = 1, bty="n",
				     bg = "transparent",
				     legend = model.names[8:12],  # Legend texts
				     lty = c(3, 3, 3, 3,3),           # Line types
				     col = c('tan1', 'green','dodgerblue', 'violet', 'burlywood4'), # Line colors
				     lwd = c(10,10,10,10,10))                # Line width			
			}
		}
	}
	dev.off()
}

scrap <- gen.imagenet("count","imagenet")
scrap <- gen.imagenet("actual","imagenet")
scrap <- gen.imagenet("count","imagenetv2")
scrap <- gen.imagenet("actual","imagenetv2")

