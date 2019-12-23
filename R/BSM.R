BSM<-function(s, k, t, vol, rf) {
  x = vol * sqrt(t)
  d1 = (log(s/k)+(rf+((vol^2)/2))*t)/x
  d2 = d1 - x
  c = (s*pnorm(d1))-(k*exp(-(rf*t))*pnorm(d2))
  return(c)
}