library(data.table)
library(tidyr)
library(scales)
library(parallel)
library(rlist)

train.actuals <- fread("train_actuals.csv")
setnames(train.actuals,"actual")

batch.files <- list.files()
batch.files <- batch.files[grepl("batch",batch.files) & grepl("csv",batch.files)]

toRDS <- function(filename){
	data <- fread(filename)
	saveRDS(data,gsub("csv","RDS",filename))
}

# batch.data <- mclapply(batch.files,toRDS,mc.cores=2)

batch.files <- list.files()
batch.files <- batch.files[grepl("batch",batch.files) & grepl("RDS",batch.files) & !grepl("cleaned",batch.files)]
batch.cleaned <- gsub("probs","cleaned",batch.files)
batch.models <- gsub("_batch.*","",batch.files)

for(i in 1:length(batch.files)){
	batch.data <- readRDS(batch.files[i])
	batch.data[,idx:=.I]
	batch.data <- melt(batch.data,id="idx")
	setkey(batch.data,idx,value)
	batch.data <- batch.data[,tail(.SD,1),idx]
	batch.data <- batch.data[,list(variable,value)]
	batch.data[,variable:=as.numeric(gsub("V","",variable))-1]
	batch.vars <- paste0(c("yhat.","phat."),batch.models[i])
	setnames(batch.data,batch.vars)
	saveRDS(batch.data,batch.cleaned[i])
}

batch.data <- lapply(batch.cleaned,readRDS)
batch.data <- list.cbind(batch.data)

batch.data <- cbind(train.actuals,batch.data)

saveRDS(batch.data,"cleaned.train.RDS")

yhat.vars <- paste0("yhat.",batch.models)
accuracy.vars <- paste0("accuracy.",batch.models)

batch.data[,(accuracy.vars):=lapply(.SD,round),.SDcols=yhat.vars]
batch.data[,(accuracy.vars):=lapply(.SD,`==`,round(actual)),.SDcols=yhat.vars]

batch.accuracy <- batch.data[,lapply(.SD,mean),.SDcols=accuracy.vars]
write.table(batch.accuracy,"train.accuracy.txt",sep='\t',row.names=FALSE)