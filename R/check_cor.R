MyData <- read.delim(file="/home/andrey/yandexML/test_intern/con_feat.txt", header=FALSE, sep="\t")
MyData = array(as.numeric(unlist(MyData)), dim=c(605, 26))
print(dim(MyData))
count_var = 26
p <- array(0,dim=c(count_var, count_var))
rho <- array(0,dim=c(count_var,count_var))

for (i in 1:count_var) {
  for (j in 1:count_var){
    ans <- cor.test(MyData[,i], MyData[,j], method = "spearman")
    rho[i,j] <- ans$estimate
    p[i,j] <- ans$p.value
  }
}

boxplot(MyData)

write.table(p, file = "/home/andrey/yandexML/test_intern/p_value.csv",row.names=FALSE, na="",col.names=FALSE, sep="\t")
write.table(rho, file = "/home/andrey/yandexML/test_intern/rho_value.csv",row.names=FALSE, na="",col.names=FALSE, sep="\t")

print(MyData[10])


library("mpmi")
for (i in 1:(count_var-1)) { 
    ans<-cmi.pw(MyData[,i], MyData[,26])
    mi <- ans$mi
  #  print(mi)
    if (mi > 0.05){
      print(i-1)
    }
}
