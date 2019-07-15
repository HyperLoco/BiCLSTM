import tflearn
import tflearn.data_utils as du
import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot='True')
print(X)
X = X.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])
X, mean = du.featurewise_zero_center(X)
testX = du.featurewise_zero_center(testX, mean)

#resnet
net = tflearn.input_data(shape=[None,28,28,1])
net = tflearn.conv_2d(net, 64, 3,activation='relu', bias='Flase')

#resnet block
net = tflearn.residual_bottleneck(net,3,16,64)

net = tflearn.residual_bottleneck(net,1,32,128,downsample=True)
net = tflearn.residual_bottleneck(net,2,32,128)

net = tflearn.residual_bottleneck(net,1,64,256,downsample=True)
net = tflearn.residual_bottleneck(net,2,64,256)

net = tflearn.batch_normalization(net)
net = tflearn.activation(net,'relu')
net = tflearn.global_avg_pool(net)
#net = tflearn.fully_connected(net,1024,activation='relu')
net = tflearn.fully_connected(net,10,activation='softmax')

net = tflearn.regression(net,optimizer='momentum',learning_rate=0.1)

model = tflearn.DNN(net,checkpoint_path='model_resnet_mnist')
model.fit(X,Y,n_epoch=10,validation_set=(testX,testY),
          show_metric=True,batch_size=256)
