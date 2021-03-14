using Pkg
Pkg.add("Flux")
using Flux
Pkg.add("CSV")
using CSV
Pkg.add("Distributions")
using Distributions
Pkg.add("Random")
using Random
Pkg.add("Statistics")
using Statistics
using Base.Iterators: repeated
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy, throttle, logitcrossentropy
using Plots
using Flux.NNlib
using Flux: throttle, normalise
using Flux: binarycrossentropy
using Base.Iterators: repeated
using Flux: @epochs


a=CSV.read("####################.csv")
a=convert(Matrix,a)


y_train=a[1:25000,25]
Y_test=a[25001:30000,25]
Y = onehotbatch(y_train, 0:1)
 X=a[1:25000,2:24]'
X_test=a[25001:30000,2:24]'
Y_test=onehotbatch(Y_test,0:1)
X=normalise(X, dims=2)
X_test=normalise(X_test,dims=2)

##From tests one hidden layer works best than more than one.  There is no improvement in performance
##with more than one.  
m=Chain(Dense(23,16,sigmoid),
        Dense(16,16,sigmoid),
        Dense(16,2), softmax)


l(x, y) = crossentropy(m(x), y)

##Rate achieved through trial and error.
opt = ADAM(.25)

##Dataset can be repeated if so desired instead of epochs.
dataset = repeated((X,Y),1)
evalcb = () -> @show(l(X, Y))


accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))


##remember to add accuracy in there later.

tots_loss=[]
accu=[]

for j in 1:2
    Flux.train!(l,Flux.params(m),dataset,opt,cb =throttle(evalcb,30))
    push!(tots_loss,l(X,Y))
    push!(accu,accuracy(X,Y))
end

##1e-6 converges too quickly.  2% to 4% improvement in the accuracy rates is achieved in longer run times.
i=2
while abs(tots_loss[i-1]-tots_loss[i])>1e-7
    Flux.train!(l,Flux.params(m),dataset,opt,cb =throttle(evalcb,30))
    push!(tots_loss,l(X,Y))
    push!(accu,accuracy(X,Y))
    i+=1
end

plot(1:length(tots_loss),tots_loss,label="Loss")

plot!(1:length(accu),accu, title = "Accuracy vs. Loss", label = "Accuracy")

plot!(xlabel="Epochs")

##Accuracy of the training set predictions
accuracy(X,Y)

0.8256

##Accuracy of the test set predictions
accuracy(X_test,Y_test)

0.821

onecold(m(X_test[:,16]))-1
1

onecold(Y_test)[16]-1
0

##Single prediction item through the network.  Same as X_Test selection from the test set.
a=onecold(m(normalise(X_test[:,16],dims=1)))-1
1



