option<-function(s, k, t, vol, rf, dt, N){
  M = t/dt
  S <- matrix(0, nrow=N, ncol=M)
  V <- matrix(0, nrow=N, ncol=M)
  
  print("S & V matrix calculation start")
  for (i in 1:nrow(S)){
    for (j in 1:ncol(S)){
      if (j == 1){
        S[i, j] <- s
      }
      else {
        S[i, j] <- S[i, j-1]*exp((rf-0.5*(vol^2))*dt+ vol*sqrt(dt)*rnorm(1, 0, 1))
      }
      
      x = c((S[i, j]-k), 0)
      V[i, j] <- max(x)
    }
    if ((i %% 1000) == 0) {
      p = i/N*100
      cat(p, "% complete", "\n")
    }
  }
  print("S & V matrix calculation end\n")
  print("Get C value")
  C <- rowSums(V)
  C = C*exp(-(rf*t))/N
  C = mean(C[1])
  cat("C: ", C)
  
  return(S)
}