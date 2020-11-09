x <- read.table("/home/nicolas/Documentos/redes-complexas/medidas-redes-complexas/indegree.txt")
y <- dnorm(x[,1])
plot(x[,1], y, type="o", col="blue", pch="o", lty=1)