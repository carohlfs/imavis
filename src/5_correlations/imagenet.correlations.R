library(data.table)
library(tidyr)
library(scales)
library(svglite)

cleaned.data <- readRDS('cleaned.data.RDS')
correlation <- cleaned.data[,list(Correlation=cor(prob,correct)),by=list(sample,size,model)]
correlation[sample=="valid",sample:="imagenet"]
correlation[sample=="test",sample:="imagenetv2"]
correlation[,size:=factor(size,levels=c(256,128,64,32,16))]
setkey(correlation,sample,size,model)

gen.imagenet.correl.graph <- function(source){

	resnext101_32x8d <- correlation[model=="resnext101_32x8d" & sample==source,Correlation]
	wide_resnet101_2 <- correlation[model=="wide_resnet101_2" & sample==source,Correlation]
	efficientnet_b0 <- correlation[model=="efficientnet_b0" & sample==source,Correlation]
	efficientnet_b7 <- correlation[model=="efficientnet_b7" & sample==source,Correlation]
	resnet101 <- correlation[model=="resnet101" & sample==source,Correlation]
	densenet201 <- correlation[model=="densenet201" & sample==source,Correlation]
	vgg19_bn <- correlation[model=="vgg19_bn" & sample==source,Correlation]
	mobilenet_v3_large <- correlation[model=="mobilenet_v3_large" & sample==source,Correlation]
	mobilenet_v3_small <- correlation[model=="mobilenet_v3_small" & sample==source,Correlation]
	googlenet <- correlation[model=="googlenet" & sample==source,Correlation]
	inception_v3 <- correlation[model=="inception_v3" & sample==source,Correlation]
	alexnet <- correlation[model=="alexnet" & sample==source,Correlation]

	x.labels <- c(256,128,64,32,16)
	sizes <- factor(x.labels,levels=x.labels)

	graph.name <- paste0(source,"_correl.svg")

	yticks_val <- pretty_breaks(n=5)(c(-0.2,0.8))

	svg(graph.name,width=17,height=11)

	par(mar=c(7,10,1,1), mgp=c(5,2.5,0), cex=2.1, cex.lab=2.0, cex.axis=1.5)
	ylims=c(-0.2,0.8)
	matplot(sizes,resnext101_32x8d,type="n",col="red3",xlab="",ylab="",ylim=ylims,lty=0, yaxt="n", xaxs="i",yaxs="i")
	abline(h = seq(0.0,0.8,0.2), lty = "dashed", col = "gray30")
	lines(sizes,resnext101_32x8d,col="red3",lwd=10,lty=1)
	lines(sizes,wide_resnet101_2,col="tan1",lwd=10,lty=1)
	lines(sizes,efficientnet_b0,col="green",lwd=10,lty=1)
	lines(sizes,efficientnet_b7,col="dodgerblue",lwd=10,lty=1)
	lines(sizes,resnet101,col="violet",lwd=10,lty=1)
	lines(sizes,densenet201,col="burlywood4",lwd=10,lty=1)
	lines(sizes,vgg19_bn,col="red3",lwd=10,lty=3)
	lines(sizes,mobilenet_v3_large,col="tan1",lwd=10,lty=3)
	lines(sizes,mobilenet_v3_small,col="green",lwd=10,lty=3)
	lines(sizes,googlenet,col="dodgerblue",lwd=10,lty=3)
	lines(sizes,inception_v3,col="violet",lwd=10,lty=3)
	lines(sizes,alexnet,col="burlywood4",lwd=10,lty=3)
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="Correlation between\nPropensity and Accuracy", mgp = c(4.6, 1, 0))
	title(xlab="Size of Validation Image", mgp = c(4.9, 1, 0))
	if(source=="imagenet"){
		legend(x="bottomleft",
			ncol = 1, bty="n",cex=1.5,
			bg = "transparent",
			legend = model.names[1:6],  # Legend texts
			lty = c(1, 1, 1, 1, 1, 1),           # Line types
			col = c('red3', 'tan1', 'green', 'dodgerblue', 'violet', 'burlywood4'),           # Line colors
			lwd = c(10,10,10,10,10,10))
	} else if(source=="imagenetv2"){
		legend(x="bottomleft",
			ncol = 1, bty="n",cex=1.5,
			bg = "transparent",
			legend = model.names[7:12],  # Legend texts
			lty = c(3, 3, 3, 3, 3, 3),           # Line types
			col = c('red3', 'tan1', 'green', 'dodgerblue', 'violet', 'burlywood4'),           # Line colors
			lwd = c(10,10,10,10,10,10))		
	}
	dev.off()
}

model.names <- c("ResNeXt101-32x8d","Wide ResNet-101-2","EfficientNet-b0","EfficientNet-b7",
	"ResNet-101","DenseNet-201","VGG-19-bn","MobileNet v3 large",
	"MobileNet v3 small","GoogLeNet","Inception v3","AlexNet")

model.names <- factor(model.names,levels=model.names)

scrap <- gen.imagenet.correl.graph("imagenet")
scrap <- gen.imagenet.correl.graph("imagenetv2")