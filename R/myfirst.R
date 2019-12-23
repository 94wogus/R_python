myfunction<-function() {
  x<-norm(100)
  mean(x)
}

second<-function(x) {
  x+rnorm(length(x))
}