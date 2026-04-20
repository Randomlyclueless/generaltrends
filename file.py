# Q.1)  
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

funct = {
    'Sigmoid': 1 / (1 + np.exp(-x)),
    'ReLU': np.maximum(0, x),
    'Tanh': np.tanh(x),
    'Leaky ReLU': np.where(x > 0, x, 0.1 * x)
}

plt.figure(figsize=(10, 6))
for i, (n, y) in enumerate(funct.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(x, y)
    plt.title(n)
    plt.grid(True)


plt.show()
# ---------------------------------------------------------------------
# Q2
#question 2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[ord(str(i))] for i in range(10)])
y = np.array([1 if i%2==0 else 0 for i in range(10)])

X = X / 100.0

model = Sequential([
    Dense(8, activation='relu', input_dim=1),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=200, verbose=0)

x_line = np.linspace(min(X), max(X), 100)
y_line = model.predict(x_line.reshape(-1,1), verbose=0)

plt.scatter(X, y, c=y)
plt.plot(x_line, y_line, color='red')
plt.show()

# ------------------------
# q3
import numpy as np

def steps(n):
  if n>=0:
    return 1
  else:
    return 0
print("And Not fucnction ")
inputs = [(0,0),(0,1),(1,0),(1,1)]
w1 = 1
w2 = -1
b = -0.5
expected = [0,0,1,0]
outputs = []
for x1,x2 in inputs:
  y = steps(x1*w1 + x2*w2+b)
  outputs.append(y)
  print(x1,x2,"->",y)

  
if outputs == expected:
  print("Verified ! ")
# --------------------------------
# q5
#question 5
# build and train fnn and feedbackward netwrok using keras

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

x = np.array([[0,1],[0,0],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential([
    Dense(2,activation='sigmoid',input_dim = 2),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer = 'adam',loss = 'mse',metrics = ['accuracy'])

model.fit(x,y,epochs = 500,verbose=0)

loss,acc = model.evaluate(x,y)
print("Accuracy : ",acc)

pred = model.predict(x)
print("Prediction : ",pred)

# -------------------------------------------------
#  q 6 n 7

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0

# Model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc)

# Predict
pred = model.predict(x_test)
print("Prediction:", pred)
# -------------------------------------------
# q8
# question 8
# sentiment analysis using RNN
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from sklearn.metrics import confusion_matrix
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = 10000)
x_train = pad_sequences(x_train,maxlen = 200)
x_test = pad_sequences(x_test, maxlen = 200)

model = Sequential([
    Embedding(input_dim = 10000, output_dim = 32, input_length = 200),
    SimpleRNN(32),
    Dense(1,activation = 'sigmoid')
    ])
  
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train,y_train,epochs = 3,batch_size = 64,verbose = 1)
loss,acc = model.evaluate(x_test,y_test)
print("Test Accuracy : ",acc)

y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)
print("confusion_matrix : ",confusion_matrix(y_test,y_pred_classes))

# question 9
# Time series prediction using lstm

import numpy as np
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential

x = np.array([
    [10, 20, 30], 
    [20, 30, 40], 
    [30, 40, 50], 
    [40, 50, 60], 
    [50, 60, 70]
])

x = x.reshape((5,3,1))
y = np.array([40,50,60,70,80])
model = Sequential([
    LSTM(50,activation = 'relu',input_shape=(3,1)),
    Dense(1)
])
model.compile(
    optimizer = 'adam',
    loss = 'mse'
)
model.fit(x,y,epochs = 200,verbose = 0)
new_w = np.array([60,70,80]).reshape((1,3,1))
pred = model.predict(new_w, verbose = 0)
print("Forecasted : ",np.round(pred[0][0]))

# q10

# Question 10 - Image Denoising

import numpy as np
from tensorflow.keras.layers import Input,Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,784)/255
x_test = x_test.reshape(-1,784)/255

noise = 0.3
x_test_noisy = x_test + noise * np.random.normal(size = x_test.shape)

inp = Input(shape = (784, ))
enc = Dense(64, activation = 'relu')(inp)
dec = Dense(784,activation = 'sigmoid')(enc)
model = Model(inp, dec)
model.compile(optimizer='adam', loss='mse')

model.fit(x_train,x_train,epochs = 10, verbose = 0)

pred = model.predict(x_test_noisy)

n = 3
for i in range(n):

    # Noisy
    plt.subplot(3,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28), cmap='gray')

    # Output
    plt.subplot(3,n,i+1+n)
    plt.imshow(pred[i].reshape(28,28), cmap='gray')

    # Original
    plt.subplot(3,n,i+1+2*n)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')

plt.show()
#q11
# Question 11 - GAN

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------------
# 1. Generator
# -------------------------------
gen = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(20, activation='sigmoid')
])

# -------------------------------
# 2. Discriminator
# -------------------------------
disc = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

disc.compile(optimizer='adam', loss='binary_crossentropy')

# -------------------------------
# 3. GAN (Combined Model)
# -------------------------------
disc.trainable = False
gan = models.Sequential([gen, disc])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# -------------------------------
# 4. Training Loop (Reduced)
# -------------------------------
for i in range(200):

    # Generate real & fake data
    real = np.random.rand(16, 20)
    noise = np.random.rand(16, 10)
    fake = gen.predict(noise, verbose=0)

    # Combine data
    X = np.vstack((real, fake))
    y = np.vstack((np.ones((16,1)), np.zeros((16,1))))

    # Train Discriminator (less frequent)
    if i % 2 == 0:
        disc.trainable = True
        disc.train_on_batch(X, y)

    # Train Generator
    disc.trainable = False
    gan.train_on_batch(noise, np.ones((16,1)))

# -------------------------------
# 5. Generate Image
# -------------------------------
sample = gen.predict(np.random.rand(1,10))

plt.imshow(sample.reshape(4,5))

plt.show()