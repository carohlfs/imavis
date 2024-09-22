library('data.table')

# pull lenets
# only do lenet for mnist-style
dataname <- c("mnist","kmnist","fashionmnist")
pullsource <- "test"
ds.rate <- 7
filenames <- paste0("/Volumes/T7 Shield/Effort_Jul2022/",dataname,ds.rate,"_",pullsource,"_probs.csv")
lenet <- lapply(filenames,fread)
n <- sapply(lenet,nrow)
lenet <- rbindlist(lenet)
lenet[,obs:=as.numeric(sapply(n,seq))]
lenet[,dataset:=rep(dataname,n)]
lenet[,model:="lenet"]
lenet[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
lenet[,selection:=as.numeric(0)]
lenet[phat==V2,selection:=1]
lenet[phat==V3,selection:=2]
lenet[phat==V4,selection:=3]
lenet[phat==V5,selection:=4]
lenet[phat==V6,selection:=5]
lenet[phat==V7,selection:=6]
lenet[phat==V8,selection:=7]
lenet[phat==V9,selection:=8]
lenet[phat==V10,selection:=9]
lenet <- lenet[,list(obs,dataset,model,phat,selection)]

actual.filenames <- paste0("/Volumes/T7 Shield/Effort_Jul2022/",dataname,"_",pullsource,"_actuals.csv")
actuals <- rbindlist(lapply(actual.filenames,fread))

lenet <- cbind(lenet,actuals)
setnames(lenet,"V1","actual")

setkey(lenet,dataset,phat)

splits.correct <- split(lenet[actual==selection],lenet[actual==selection,dataset])
splits <- split(lenet,lenet[,dataset])

worst <- rbindlist(lapply(splits,head,2))
best <- rbindlist(lapply(splits.correct,tail,2))

# pull res:
dataname <- c("svhn","cifar")
ds.rate <- 8
filenames <- paste0("/Volumes/T7 Shield/Effort_Jul2022/pytorch-",dataname,"-resnet/",dataname,ds.rate,"_test_resprobs.csv")
resnet <- lapply(filenames,fread)
n <- sapply(resnet,nrow)
resnet <- rbindlist(resnet)
resnet[,obs:=unlist(sapply(n,seq))]
resnet[,dataset:=rep(dataname,n)]
resnet[,model:="resnet"]
resnet[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
resnet[,selection_resnet:=as.numeric(0)]
resnet[phat==V2,selection_resnet:=1]
resnet[phat==V3,selection_resnet:=2]
resnet[phat==V4,selection_resnet:=3]
resnet[phat==V5,selection_resnet:=4]
resnet[phat==V6,selection_resnet:=5]
resnet[phat==V7,selection_resnet:=6]
resnet[phat==V8,selection_resnet:=7]
resnet[phat==V9,selection_resnet:=8]
resnet[phat==V10,selection_resnet:=9]
resnet[,phat_resnet:=exp(phat)/(exp(V1)+exp(V2)+exp(V3)+exp(V4)+exp(V5)+exp(V6)+exp(V7)+exp(V8)+exp(V9)+exp(V10))]
resnet <- resnet[,list(obs,dataset,phat_resnet,selection_resnet)]

# pull dla
filenames <- paste0("/Volumes/T7 Shield/Effort_Jul2022/pytorch-",dataname,"-master/",dataname,ds.rate,"_test_dlaprobs.csv")
dla <- lapply(filenames,fread)
dla <- rbindlist(dla)
dla[,obs:=unlist(sapply(n,seq))]
dla[,dataset:=rep(dataname,n)]
dla[,model:="dla"]
dla[,phat:=pmax(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10)]
dla[,selection_dla:=as.numeric(0)]
dla[phat==V2,selection_dla:=1]
dla[phat==V3,selection_dla:=2]
dla[phat==V4,selection_dla:=3]
dla[phat==V5,selection_dla:=4]
dla[phat==V6,selection_dla:=5]
dla[phat==V7,selection_dla:=6]
dla[phat==V8,selection_dla:=7]
dla[phat==V9,selection_dla:=8]
dla[phat==V10,selection_dla:=9]
dla[,phat_dla:=exp(phat)/(exp(V1)+exp(V2)+exp(V3)+exp(V4)+exp(V5)+exp(V6)+exp(V7)+exp(V8)+exp(V9)+exp(V10))]
dla <- dla[,list(obs,dataset,phat_dla,selection_dla)]


actual.filenames <- paste0("/Volumes/T7 Shield/Effort_Jul2022/",dataname,"_",pullsource,"_actuals.csv")
actuals <- rbindlist(lapply(actual.filenames,fread))
setnames(actuals,"V1","actual")

bestworst <- cbind(resnet,dla[,phat_dla,selection_dla],actuals)
bestworst[,phat:=0.5*(phat_dla+phat_resnet)]
bestworst[,c("phat_dla","phat_resnet"):=NULL]

setkey(bestworst,dataset,phat)

splits.correct <- split(bestworst[actual==selection_resnet & actual==selection_dla],bestworst[actual==selection_resnet & actual==selection_dla,dataset])
splits <- split(bestworst,bestworst[,dataset])

worst <- rbindlist(lapply(splits,head,2))
best <- rbindlist(lapply(splits.correct,tail,2))
