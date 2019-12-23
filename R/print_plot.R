print_plot<-function(n, datas){
  for (i in 1:n){
    x <- 1:ncol(datas)
    y <- S[i, ]
    df = data.frame(x, y)
    plot <- ggplot(data=df, aes(x=x, y=y)) + geom_line(size=1) + labs(x="Time", y="Stock Price") + ggtitle(toString(i))
    print(plot)
  }
}