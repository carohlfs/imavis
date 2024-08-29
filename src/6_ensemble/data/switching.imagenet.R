
library(data.table)
library(tidyr)
library(scales)

switching.inet <- function(i){
	o <- specifications[i,order]
	threshold <- specifications[i,threshold]

	results <- copy(results.table)

	results[,used16:=as.numeric(0)]
	results[,used32:=as.numeric(0)]
	results[,used64:=as.numeric(0)]
	results[,used128:=as.numeric(0)]
	results[,used256:=as.numeric(0)]

	if(o==1){
		results[,chosen:=16]
		results[,used16:=1]
		results[,correct:=c16]
		results[,bytes:=bytes16]
		results[p16<threshold,chosen:=32]
		results[p16<threshold,used32:=1]
		results[p16<threshold,correct:=c32]
		results[p16<threshold,bytes:=bytes32]
	}
	if(o==2){
		results[,chosen:=32]
		results[,used32:=1]
		results[,correct:=c32]
		results[,bytes:=bytes32]	
	}
	if(o<=2){
		results[p32<threshold & used32==1,chosen:=64]
		results[p32<threshold & used32==1,used64:=1]
		results[p32<threshold & used32==1,correct:=c64]	
		results[p32<threshold & used32==1,bytes:=bytes64]
		results[p64<threshold & used64==1,chosen:=128]
		results[p64<threshold & used64==1,used128:=1]
		results[p64<threshold & used64==1,correct:=c128]
		results[p64<threshold & used64==1,bytes:=bytes128]		
	}
	if(o==3){
		results[,chosen:=64]
		results[,used64:=1]
		results[,correct:=c64]
		results[,bytes:=bytes64]	
	}

	results[p64<threshold & used64==1,chosen:=128]
	results[p64<threshold & used64==1,used128:=1]
	results[p64<threshold & used64==1,correct:=c128]
	results[p64<threshold & used64==1,bytes:=bytes128]
	results[p128<threshold & used128==1,chosen:=256]
	results[p128<threshold & used128==1,used256:=1]
	results[p128<threshold & used128==1,correct:=c256]
	results[p128<threshold & used128==1,bytes:=bytes256]

	results[,bytes:=bytes16*used16+bytes32*used32+bytes64*used64+bytes128*used128+bytes256*used256]
	results <- results[,list(correct=mean(correct),bytes=mean(bytes)),by=list(model,sample)]
	results[,order:=o]
	results[,threshold:=threshold]
	return(results)
}

gen.inet.ppf <- function(source){

	resnext101_32x8d.correct <- spec.results[model=="resnext101_32x8d" & order==3 & sample==source,correct]
	resnext101_32x8d.bytes <- spec.results[model=="resnext101_32x8d" & order==3 & sample==source,bytes]
	wide_resnet101_2.correct <- spec.results[model=="wide_resnet101_2" & order==3 & sample==source,correct]
	wide_resnet101_2.bytes <- spec.results[model=="wide_resnet101_2" & order==3 & sample==source,bytes]
	efficientnet_b0.correct <- spec.results[model=="efficientnet_b0" & order==3 & sample==source,correct]
	efficientnet_b0.bytes <- spec.results[model=="efficientnet_b0" & order==3 & sample==source,bytes]
	efficientnet_b7.correct <- spec.results[model=="efficientnet_b7" & order==3 & sample==source,correct]
	efficientnet_b7.bytes <- spec.results[model=="efficientnet_b7" & order==3 & sample==source,bytes]
	resnet101.correct <- spec.results[model=="resnet101" & order==3 & sample==source,correct]
	resnet101.bytes <- spec.results[model=="resnet101" & order==3 & sample==source,bytes]
	densenet201.correct <- spec.results[model=="densenet201" & order==3 & sample==source,correct]
	densenet201.bytes <- spec.results[model=="densenet201" & order==3 & sample==source,bytes]
	vgg19_bn.correct <- spec.results[model=="vgg19_bn" & order==3 & sample==source,correct]
	vgg19_bn.bytes <- spec.results[model=="vgg19_bn" & order==3 & sample==source,bytes]
	mobilenet_v3_large.correct <- spec.results[model=="mobilenet_v3_large" & order==3 & sample==source,correct]
	mobilenet_v3_large.bytes <- spec.results[model=="mobilenet_v3_large" & order==3 & sample==source,bytes]
	mobilenet_v3_small.correct <- spec.results[model=="mobilenet_v3_small" & order==3 & sample==source,correct]
	mobilenet_v3_small.bytes <- spec.results[model=="mobilenet_v3_small" & order==3 & sample==source,bytes]
	googlenet.correct <- spec.results[model=="googlenet" & order==3 & sample==source,correct]
	googlenet.bytes <- spec.results[model=="googlenet" & order==3 & sample==source,bytes]
	inception_v3.correct <- spec.results[model=="inception_v3" & order==3 & sample==source,correct]
	inception_v3.bytes <- spec.results[model=="inception_v3" & order==3 & sample==source,bytes]
	alexnet.correct <- spec.results[model=="alexnet" & order==3 & sample==source,correct]
	alexnet.bytes <- spec.results[model=="alexnet" & order==3 & sample==source,bytes]

	resnext101_32x8d.benchmark.correct <- benchmark.results[model=="resnext101_32x8d" & sample==source,correct]
	resnext101_32x8d.benchmark.bytes <- benchmark.results[model=="resnext101_32x8d" & sample==source,bytes]
	wide_resnet101_2.benchmark.correct <- benchmark.results[model=="wide_resnet101_2" & sample==source,correct]
	wide_resnet101_2.benchmark.bytes <- benchmark.results[model=="wide_resnet101_2" & sample==source,bytes]
	efficientnet_b0.benchmark.correct <- benchmark.results[model=="efficientnet_b0" & sample==source,correct]
	efficientnet_b0.benchmark.bytes <- benchmark.results[model=="efficientnet_b0" & sample==source,bytes]
	efficientnet_b7.benchmark.correct <- benchmark.results[model=="efficientnet_b7" & sample==source,correct]
	efficientnet_b7.benchmark.bytes <- benchmark.results[model=="efficientnet_b7" & sample==source,bytes]
	resnet101.benchmark.correct <- benchmark.results[model=="resnet101" & sample==source,correct]
	resnet101.benchmark.bytes <- benchmark.results[model=="resnet101" & sample==source,bytes]
	densenet201.benchmark.correct <- benchmark.results[model=="densenet201" & sample==source,correct]
	densenet201.benchmark.bytes <- benchmark.results[model=="densenet201" & sample==source,bytes]
	vgg19_bn.benchmark.correct <- benchmark.results[model=="vgg19_bn" & sample==source,correct]
	vgg19_bn.benchmark.bytes <- benchmark.results[model=="vgg19_bn" & sample==source,bytes]
	mobilenet_v3_large.benchmark.correct <- benchmark.results[model=="mobilenet_v3_large" & sample==source,correct]
	mobilenet_v3_large.benchmark.bytes <- benchmark.results[model=="mobilenet_v3_large" & sample==source,bytes]
	mobilenet_v3_small.benchmark.correct <- benchmark.results[model=="mobilenet_v3_small" & sample==source,correct]
	mobilenet_v3_small.benchmark.bytes <- benchmark.results[model=="mobilenet_v3_small" & sample==source,bytes]
	googlenet.benchmark.correct <- benchmark.results[model=="googlenet" & sample==source,correct]
	googlenet.benchmark.bytes <- benchmark.results[model=="googlenet" & sample==source,bytes]
	inception_v3.benchmark.correct <- benchmark.results[model=="inception_v3" & sample==source,correct]
	inception_v3.benchmark.bytes <- benchmark.results[model=="inception_v3" & sample==source,bytes]
	alexnet.benchmark.correct <- benchmark.results[model=="alexnet" & sample==source,correct]
	alexnet.benchmark.bytes <- benchmark.results[model=="alexnet" & sample==source,bytes]

	model.benchmark <- paste0(model.names,"\n- Benchmark")
	model.benchmark <- factor(model.benchmark,levels=model.benchmark)
	model.threshold <- paste0(model.names," -\nThreshold-Based")
	model.threshold <- factor(model.threshold,levels=model.threshold)

	graph.name <- paste0(source,"_ppf.svg")

	yticks_val <- pretty_breaks(n=5)(c(0,1))

	svg(graph.name,width=32.5,height=11)

	par(mar=c(5.5,7.5,2,33), mgp=c(5,1,0), cex=2.1, cex.lab=2.0, cex.axis=1.5,xpd=TRUE)
	ylims=c(0,1)
	xlims <- c(0,350000)
	matplot(resnext101_32x8d.bytes,resnext101_32x8d.correct,type="n",col="red3",xlab="",ylab="",ylim=ylims,xlim=xlims,lty=0, yaxt="n", xaxt="n", xaxs="i",yaxs="i")
	rect(par("usr")[1], par("usr")[3],
	    par("usr")[2], par("usr")[4],
	    col = "white") # Color
	lines(xlims,c(0.2,0.2),lty="dashed",col="gray30")
	lines(xlims,c(0.4,0.4),lty="dashed",col="gray30")
	lines(xlims,c(0.6,0.6),lty="dashed",col="gray30")
	lines(xlims,c(0.8,0.8),lty="dashed",col="gray30")

	lines(resnext101_32x8d.bytes,resnext101_32x8d.correct,col="red3",lwd=10,lty=1)
	lines(wide_resnet101_2.bytes,wide_resnet101_2.correct,col="tan1",lwd=10,lty=1)
	lines(efficientnet_b0.bytes,efficientnet_b0.correct,col="green",lwd=10,lty=1)
	lines(efficientnet_b7.bytes,efficientnet_b7.correct,col="dodgerblue",lwd=10,lty=1)
	lines(resnet101.bytes,resnet101.correct,col="violet",lwd=10,lty=1)
	lines(densenet201.bytes,densenet201.correct,col="burlywood4",lwd=10,lty=1)
	lines(vgg19_bn.bytes,vgg19_bn.correct,col="red3",lwd=10,lty=3)
	lines(mobilenet_v3_large.bytes,mobilenet_v3_large.correct,col="tan1",lwd=10,lty=3)
	lines(mobilenet_v3_small.bytes,mobilenet_v3_small.correct,col="green",lwd=10,lty=3)
	lines(googlenet.bytes,googlenet.correct,col="dodgerblue",lwd=10,lty=3)
	lines(inception_v3.bytes,inception_v3.correct,col="violet",lwd=10,lty=3)
	lines(alexnet.bytes,alexnet.correct,col="burlywood4",lwd=10,lty=3)

	points(resnext101_32x8d.benchmark.bytes,resnext101_32x8d.benchmark.correct,col="red3",pch=8,cex=2.0)
	points(wide_resnet101_2.benchmark.bytes,wide_resnet101_2.benchmark.correct,col="tan1",pch=8,cex=2.0)
	points(efficientnet_b0.benchmark.bytes,efficientnet_b0.benchmark.correct,col="green",pch=8,cex=2.0)
	points(efficientnet_b7.benchmark.bytes,efficientnet_b7.benchmark.correct,col="dodgerblue",pch=8,cex=2.0)
	points(resnet101.benchmark.bytes,resnet101.benchmark.correct,col="violet",pch=8,cex=2.0)
	points(densenet201.benchmark.bytes,densenet201.benchmark.correct,col="burlywood4",pch=8,cex=2.0)
	points(vgg19_bn.benchmark.bytes,vgg19_bn.benchmark.correct,col="red3",pch=9,cex=2.0)
	points(mobilenet_v3_large.benchmark.bytes,mobilenet_v3_large.benchmark.correct,col="tan1",pch=9,cex=2.0)
	points(mobilenet_v3_small.benchmark.bytes,mobilenet_v3_small.benchmark.correct,col="green",pch=9,cex=2.0)
	points(googlenet.benchmark.bytes,googlenet.benchmark.correct,col="dodgerblue",pch=9,cex=2.0)
	points(inception_v3.benchmark.bytes,inception_v3.benchmark.correct,col="violet",pch=9,cex=2.0)
	points(alexnet.benchmark.bytes,alexnet.benchmark.correct,col="burlywood4",pch=9,cex=2.0)

	axis(1, at=axTicks(1), labels=formatC(axTicks(1), format="d", big.mark=','))
	axis(2, at=yticks_val, lab=percent(yticks_val), pos=1.1, lwd=0, las=1)
	axis(2, at=yticks_val, lab=c("","","","","",""), lwd=1)
	title(ylab="% Validation Cases Correct", mgp = c(4.6, 1, 0))
	title(xlab="Bytes Read per Validation Case", mgp = c(4.0, 1, 0))
	if(source=="imagenet"){
		legend(x = "topright",
	      ncol = 2,cex=1.6,inset=c(-1.02,-0.10),y.intersp=2,
	      bg = "transparent",bty="n",pt.cex=2,
	      legend = c(model.benchmark[1:6],model.threshold[1:6])
	  ,  # Legend texts
	      lty = c(NA, NA, NA, NA, NA,NA,1,1,1,1,1,1),           # Line types
	      pch = c(8,8,8,8,8,8,NA,NA,NA,NA,NA,NA),
	      col = c('red3','tan1','green','dodgerblue','violet','burlywood4','red3','tan1','green','dodgerblue','violet','burlywood4'),           # Line colors
		  lwd = c(NA,NA,NA,NA,NA,NA,10,10,10,10,10,10))
	} else if(source=="imagenetv2"){
		legend(x = "topright",
	      ncol = 2,cex=1.6,inset=c(-1.02,-0.10),y.intersp=2,
	      bg = "transparent",bty="n",pt.cex=2,
	      legend = c(model.benchmark[7:12],model.threshold[7:12])
	  ,  # Legend texts
	      lty = c(NA, NA, NA, NA, NA,NA,3,3,3,3,3,3),           # Line types
	      pch = c(9,9,9,9,9,9,NA,NA,NA,NA,NA,NA),
	      col = c('red3','tan1','green','dodgerblue','violet','burlywood4','red3','tan1','green','dodgerblue','violet','burlywood4'),           # Line colors
		  lwd = c(NA,NA,NA,NA,NA,NA,10,10,10,10,10,10))
	}
	dev.off()
}


valid.image.dims <- fread('valid_shapes.txt')
setnames(valid.image.dims,c("width","height"))

test.image.dims <- fread('test_shapes.csv')
setnames(test.image.dims,c("width","height"))

valid.image.dims[,obs:=.I]
valid.image.dims[,sample:="imagenet"]
test.image.dims[,obs:=.I]
test.image.dims[,sample:="imagenetv2"]
image.dims <- rbind(valid.image.dims,test.image.dims)
setkey(image.dims,sample,obs)

cleaned.data <- readRDS('cleaned.data.RDS')
cleaned.data[sample=="valid",sample:="imagenet"]
cleaned.data[sample=="test",sample:="imagenetv2"]

setkey(cleaned.data,sample,obs)
cleaned.data <- image.dims[cleaned.data]

probs <- cleaned.data[,list(sample,obs,model,size,prob)]
probs[,size:=paste0("p",size)]
probs[,size:=factor(size,levels=c("p16","p32","p64","p128","p256"))]
probs <- spread(probs,size,prob)

correct <- cleaned.data[,list(sample,obs,model,size,correct)]
correct[,size:=paste0("c",size)]
correct[,size:=factor(size,levels=c("c16","c32","c64","c128","c256"))]
correct <- spread(correct,size,correct)

bytes <- image.dims[,list(sample,obs,width,height)]
bytes[,shorter:=pmin(width,height)]
bytes[,longer:=pmax(width,height)]
bytes[,bytes256:=3*(256/shorter)^2*shorter*longer]
bytes[shorter<256,bytes256:=3*width*height]
bytes[,bytes128:=3*(128/shorter)^2*shorter*longer]
bytes[shorter<128,bytes128:=3*width*height]
bytes[,bytes64:=3*(64/shorter)^2*shorter*longer]
bytes[shorter<64,bytes64:=3*width*height]
bytes[,bytes32:=3*(32/shorter)^2*shorter*longer]
bytes[shorter<32,bytes32:=3*width*height]
bytes[,bytes16:=3*(16/shorter)^2*shorter*longer]
bytes[shorter<16,bytes16:=3*width*height]
bytes <- bytes[,list(sample,obs,bytes16,bytes32,bytes64,bytes128,bytes256)]

results.table <- cbind(probs,correct[,list(c16,c32,c64,c128,c256)])
setkey(results.table,sample,obs)
setkey(bytes,sample,obs)

results.table <- bytes[results.table]
setcolorder(results.table,c("sample","obs","model","p16","p32","p64","p128","p256",
	"c16","c32","c64","c128","c256","bytes16","bytes32","bytes64","bytes128","bytes256"))

source.list <- c("imagenet","imagenetv2")

# model.names <- c("resnext101_32x8d","wide_resnet101_2","efficientnet_b0","efficientnet_b7",
#	"resnet101","densenet201","vgg19_bn","mobilenet_v3_large",
#	"mobilenet_v3_small","googlenet","inception_v3","alexnet")

# model.names <- factor(model.names,levels=model.names)

model.names <- c("ResNeXt101-32x8d","Wide ResNet-101-2","EfficientNet-b0","EfficientNet-b7",
	"ResNet-101","DenseNet-201","VGG-19-bn","MobileNet v3 large",
	"MobileNet v3 small","GoogLeNet","Inception v3","AlexNet")

model.names <- factor(model.names,levels=model.names)

benchmark <- c(1)
order1 <- c(16,32,64,128,256)
order2 <- c(32,64,128,256)
order3 <- c(64,128,256)
order <- list(benchmark,order1,order2,order3)

thresholds <- c(0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,0.999)

specifications <- data.table(expand.grid(order=1:3,threshold=thresholds))

spec.results <- lapply(1:nrow(specifications),switching.inet)
spec.results <- rbindlist(spec.results)

benchmark.results <- results.table[,list(correct=mean(c256),bytes=mean(bytes256)),by=list(model,sample)]
benchmark.results[,order:=0]
benchmark.results[,threshold:=0]

spec.results <- rbind(benchmark.results,spec.results)
spec.results <- spec.results[threshold>=0.5 & threshold<=0.975]

benchmark.results[,c("order","threshold"):=NULL]
write.table(benchmark.results,"benchmark.txt",sep='\t',row.names=FALSE)

o3.results <- spec.results[order==3,list(model,sample,correct,bytes,threshold)]
o3.results <- melt(o3.results,id=c("model","sample","threshold"))
o3.results <- spread(o3.results,threshold,value)
setkey(o3.results,sample,model)

write.table(o3.results,"o3.results.txt",sep='\t',row.names=FALSE)


