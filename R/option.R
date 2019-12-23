library('ggplot2')

question1<-function() {
  St<-c(0:200)
  k = 100
  Y = c()
  for (i in St){
    x <- c(i-k,0)
    Y <- append(max(x), Y)
  }
  df = data.frame(St, Y)
  ggplot(data=df, aes(x=St, y=Y)) + geom_point(size=1) + labs(x="St", y="Y")
}

