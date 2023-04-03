# deep-learning-challenge

The overall goal of this assignment was to create a neural network to look at whether applicants for funding will be successful or not. The model was close to the target accuracy level of 75%+ but ultimately came up 2.3% short. I had tried to fine tune the model using different levels of hidden layers  and even automate the process using keras tuner but could not eclipse that 75% mark. 

The data was attained from https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv and was then cleaned up to drop any unneeded columns. 
I then created dummy variabled for a successful or unsuccessful funding. I tried the initial test with just two hidden layers which did not produce the 75% target, I may have overdone the amount of parameters for each hidden layer and overcomplicated the model. I tried again with a new approach, 3 hidden layers 

layer 1 - 7,
layer 2 - 14,
layer 3 - 21,

The model performed ever so slightly better but still could not eclipse that desired 75% accuracy. 

My next attempt took quite a long time and once again i fear that I overcomplicated the model. Using keras tuner i let the computer try to determine the best amount of layers and parameters. After a few hours of tests the best the model could produce was ~73% still not up to the standard. 

After all this I decided to try one more time with a much simpler approach. still 3 hidden layers however much less parameters.

layer 1 - 7,
layer 2 - 14,
layer 3 - 1

This produced the worst result so far at only a 70% accuracy. 

My last attempt will take out the 3rd hidden layer which i fear has just confused the model. I ran it one more time with just the first two hidden layers 
layer 1 - 7
layer 2 - 14

In one last ditch effort i decided to try changing the activation from 'relu' to 'tanh' and it appeared to work until it leveled off and actually still produced a worse output than the 2 hidden layer attempt. it had an accuracy of 72.1%

Even with this model being simplified the best I could produce was 72.7% accuracy. not terrible but I decided to call it there. I am unsure how to make this any better. After all of these different models were tested. This was the best output I could produce using this snippet of code...

     #fine tune the model
     # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
     number_input_features = len(X_train[0])
     hidden_nodes_layer1 =  7
     hidden_nodes_layer2 = 14
     #hidden_nodes_layer3 = 1

     nn = tf.keras.models.Sequential()

     # First hidden layer
     nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

     # Second hidden layer
     nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

     # Third hidden layer
     #nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

     # Output layer

     nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

     # Check the structure of the model
     nn.summary()

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_11 (Dense)            (None, 7)                 308       

     dense_12 (Dense)            (None, 14)                112       

     dense_13 (Dense)            (None, 1)                 15        

    =================================================================
    Total params: 435
    Trainable params: 435
    Non-trainable params: 0
    _________________________________________________________________

    268/268 - 1s - loss: 0.5531 - accuracy: 0.7275 - 611ms/epoch - 2ms/step
    Loss: 0.5530908107757568, Accuracy: 0.7274635434150696
