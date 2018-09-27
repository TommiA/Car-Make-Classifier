# Car-Make-Classifier
Transfer learning on VGG16 based convolutional network model to infer car make from image. Training code loads the basic VGG16 with imagenet weights leaving the top out. The code adds 512-256-81 classifier NN on top of it as my data contained 81 different car make models, i.e. classes. The trials with my training and testing data with this model reached validation accuracy of ~85%. Proper filtering to improve the training and testing data quality could improve that.
![Alt text](plot_E100_IMG400_BS300_INVLR10000.png?raw=true "Title")
