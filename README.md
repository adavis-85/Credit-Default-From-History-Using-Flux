# Credit Default From History Using Flux

The credit data used was organized in 30,000 samples of 25 ratings based on the credit limits available, sex, education, marital status and payment amounts and accounts.  The data is used to predict a person defaulting on their credit cards within the next month.  Defaulting is labeled with a 1 and not defaulting is labeled with 0.  The most common defaulting characteristic is female, with a university education and married at 3% of the dataset.  

Flux is a useful package to use for Julia for machine learning.  This is similar to tensorflow or keras in Python.  To model this specific problem the training set was portioned to 25,000 samples and the test set to the remaining 5,000.  The optimal accuracy rate was achieved with one hidden layer.  The model achieved better accuracy with the node amount between the input layer at 16.  The output layer node count was at 2 with a softmax so that the highest probability could be seen for each sample and help to train the model.  In Flux the model is written as follows:
```
m=Chain(Dense(23,16,sigmoid),
        Dense(16,16,sigmoid),
        Dense(16,2), softmax)
```
The model was set to run until the difference in consecutive losses was less than .0000001.  As seen in the following graph of loss versus accuracy of the training set, the model was stopped at precisely 860 epochs.  Both the training and test outputs were one hot encoded and the training data and the test data were both normalized to achieve faster and more consistent results.  

<img width="599" alt="image" src="https://user-images.githubusercontent.com/58529391/111062134-f60efd80-845b-11eb-9cd9-f2d690a2722e.png">

The model achieved its maximum accuracy at 857 epochs.  The graph shows that the number of epochs starts to have less of an effect of causing the loss to get smaller faster as well as the accuracy to continuously increase.  The rate of loss decreases more slowly at around 150 epochs.  The accuracy starts to increase at a much slower rate at around 30 epochs.  The first accuracy test achieved a rate of 77%.  The final training rate was 82.5% and the training accuracy was 82.1%.  This took less than 10 minutes to train.  The loss did not come lower than .39 and tended to fluctuate to greater than 40.  
