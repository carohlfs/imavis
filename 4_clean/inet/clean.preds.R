library(data.table)
library(tidyr)
library(scales)

clean.preds <- function(model.name){
	sizes <- c(16,32,64,128,256)
	model.files <- paste0(model.name,".",sizes,".csv")
	model.data <- lapply(model.files,read.csv,header=FALSE)
	model.data <- lapply(model.data,data.table)
	model.data <- rbindlist(model.data)
	model.data[,variable:="prob"]
	model.data[,id:=.I]
	model.data[id %% 2 == 0,variable:="pred"]
	model.data[,id:=floor((id+1)/2)]
	model.data <- spread(model.data,variable,V1)
	model.data[,sample:=rep(c(rep("valid",50000),rep("test",10000)),5)]
	model.data[,obs:=rep(c(1:50000,1:10000),5)]
	model.data[,size:=rep(sizes,each=60000)]
	model.data[,id:=NULL]
	model.data[,model:=model.name]
	return(model.data)
}

model.names <- c("resnext101_32x8d","wide_resnet101_2","efficientnet_b0","efficientnet_b7",
	"resnet101","densenet201","vgg19_bn","mobilenet_v3_large",
	"mobilenet_v3_small","googlenet","inception_v3","alexnet")

model.names <- factor(model.names,levels=model.names)

cleaned.data <- lapply(model.names,clean.preds)
cleaned.data <- rbindlist(cleaned.data)

valid.actuals <- data.table(read.csv('valid_actuals.csv',header=FALSE))
test.actuals <- data.table(read.csv('test_actuals.csv',header=FALSE))
valid.actuals[,sample:="valid"]
valid.actuals[,obs:=1:50000]
test.actuals[,sample:="test"]
test.actuals[,obs:=1:10000]

actuals <- rbind(valid.actuals,test.actuals)
setnames(actuals,c("actual","sample","obs"))

setkey(actuals,sample,obs)
setkey(cleaned.data,sample,obs)

cleaned.data <- actuals[cleaned.data]
cleaned.data[,correct:=as.numeric(actual==pred)]
saveRDS(cleaned.data,"cleaned.data.RDS")

rates <- cleaned.data[,mean(correct),by=list(sample,size,model)]
rates <- spread(rates,size,V1)
setcolorder(rates,c("sample","model","256","128","64","32","16"))

write.table(rates,"rates.txt",sep='\t',row.names=FALSE)