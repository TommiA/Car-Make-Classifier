# Car-Make-Classifier
Transfer train VGG16 based model to infer car make from image. Training code loads the basic VGG16 with imagenet weights leaving the top out. The code adds 512-256-81 classifier NN on top of it as my data contained 81 different car make models, i.e. classes. The trials with my training and testing data with this model reached validation accuracy of ~85%.
