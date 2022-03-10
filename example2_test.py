import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# Load up the dataset and separate training and test data
ds = tfds.load(name='example2')
training, testing = ds["train"], ds["test"]
builder = tfds.builder('example2')
#dsinfo = builder.info
#print(dsinfo) # Print dataset information

# Extract features (images and labels)
for extract in training.batch(1600):
  trainx = extract['image'].numpy().astype('float32')
  trainy = extract['label']
  
for extract2 in testing.batch(200):
  testx = extract2['image'].numpy().astype('float32')
  testy = extract2['label']

# Crop out whitespace from matplotlib
trainx = tf.image.central_crop(trainx, 0.98)
testx = tf.image.central_crop(testx, 0.98)

# Divide pixels by 255 to be 0 to 1
trainx = trainx/255.0
testx = testx/255.0

# Set labels as "categories"
trainy = tf.keras.utils.to_categorical(trainy,6)
testy = tf.keras.utils.to_categorical(testy,6)

# Resizing boosts accuracy through testing
trainx = tf.image.resize(trainx, [20,20]).numpy()
testx = tf.image.resize(testx, [20,20]).numpy()

# Model
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(25, 5, input_shape=(20,20,3),padding='same', activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
cnn.add(tf.keras.layers.Conv2D(50, 5, padding='same', activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
cnn.add(tf.keras.layers.Conv2D(100, 5, padding='same', activation="relu"))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(200, activation='relu'))
cnn.add(tf.keras.layers.Dense(6, activation='softmax'))
cnn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(trainx, trainy, validation_split=0.20, epochs=15, validation_data=(testx, testy))

print("-----------------------EVALUATE CNN------------------------------")
check = cnn.evaluate(testx, testy)

