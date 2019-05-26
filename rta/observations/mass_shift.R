# AAAAAAAAAPAAAATAPTTAATTAATAAQ
library(IsoSpecR)

plot.dupa = function(molecule, m){
  res <- IsoSpecify(molecule=molecule,
                    stopCondition= .99,
                    showCounts=TRUE)
  
  res[order(res[,'logProb'], decreasing = T),]
  res = cbind(res, exp(res[,'logProb']))
  colnames(res)[12] = 'prob'
  
  plot(res[,c('mass','prob')], type='h')
  points(m, replicate(length(m), 0), col='red')
  
  mass = res[,'mass']
  pr = res[,'prob']
  return(c('most_prob'=mass[1], 
           'min_mass'=mass[which.min(mass)],
           'ave_mass'=sum(mass*pr)/sum(pr)))
}

molecule = c(C=99, H=166, N=30, O=37)
molecule = c(C=99, H=167, N=30, O=37)
m = c(2368.2185, 2368.2184, 2368.2198, 2368.2182, 2368.2169, 2368.2164, 2368.2175, 2368.2185, 2368.2196, 2368.2178)
plot.dupa(molecule, m)

molecule = c(C=41,H=73,N=13,O=13)
molecule = c(C=41,H=74,N=13,O=13)
m = c(956.5547, 956.5545, 956.5566, 956.5565, 956.5557, 956.5556,
      956.5579, 956.5555, 956.5558, 956.555)
plot.dupa(molecule, m)

molecule = c(C=52,H=89,N=13,O=20)
molecule = c(C=52,H=90,N=13,O=20)
m = c(1216.6434, 1216.6437, 1216.6442, 1216.646 , 1216.6459, 1216.6451,
      1216.6427, 1216.6463, 1216.6478, 1216.6438)
plot.dupa(molecule, m)

mol = c(C=178, H=285, N=43, O=59)
mol = c(C=178, H=286, N=43, O=59)
m = c(3970.0879, 3970.0831, 3970.0842, 3970.0872, 3970.0801, 3970.0789,
      3970.0807, 3970.0793, 3970.0856, 3970.0793)
plot.dupa(mol, m)
# It seems, that the formula is almost equal to the most probable peak divided by mass plus one hydrogen mass.
# Ask Stefan, what is this...