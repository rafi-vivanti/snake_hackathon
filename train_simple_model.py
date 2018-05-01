from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

sample_batch_train_data = np.load(r"D:\projects\RL\snake\hackathon\rafi\logs\feturesAndRewards.npy")
#loading data
np.random.shuffle(sample_batch_train_data)
half_sz = int(sample_batch_train_data.shape[0]/2)
input_dim = 43
X_train = sample_batch_train_data[1:half_sz, :input_dim]
y_train = sample_batch_train_data[1:half_sz, input_dim]
X_test  = sample_batch_train_data[half_sz:, :input_dim]
y_test  = sample_batch_train_data[half_sz:, input_dim]
mx = np.max((y_train.max(), y_test.max()))
mn = np.min((y_train.min(), y_test.min()))
y_test = (y_test-mn)/(mx-mn)
y_train = (y_train-mn)/(mx-mn)
num_train_examples = X_train.shape[0]
num_eval_examples = X_test.shape[0]

init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
activ = 'relu'# LeakyReLU(alpha=0.1)
# create model
model = Sequential()
model.add(Dense(12, input_dim=input_dim, kernel_initializer=init, activation=activ))
model.add(Dense(8,kernel_initializer=init, activation=activ))
model.add(Dense(1,kernel_initializer=init, activation='sigmoid'))

# Compile model
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])


# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=200 )

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X_test)
plt.plot(y_test, '.')
plt.hold(1)
plt.plot(predictions, '.')
plt.show()

a=1
