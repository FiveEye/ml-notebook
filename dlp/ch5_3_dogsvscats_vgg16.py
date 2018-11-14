from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = '/home/han/code/data/dogsvscats/small/train'
val_dir = '/home/han/code/data/dogsvscats/small/validation'
test_dir = '/home/han/code/data/dogsvscats/small/test'

def build_model():
	conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150,150,3))
	
	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(256,activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	print("trainable weights:", len(model.trainable_weights))
	conv_base.trainable = False
	print("freezing, trainable weights:", len(model.trainable_weights))
	
	conv_base.trainable = True
	set_trainable = False
	for layer in conv_base.layers:
		if layer.name == 'block5_conv1':
			set_trainable = True
		layer.trainable = set_trainable
	print("fine-tuning, trainable weights:", len(model.trainable_weights))
	

	model.compile(
		loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(lr=1e-5),
		metrics=['acc'])
	return model
	
def create_data_gen():
    train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        test_dir,
		target_size=(150,150),
		batch_size=32,
		class_mode='binary')
    return train_generator, val_generator, test_generator

def train(model):
	train_generator, val_generator, test_generator = create_data_gen()
	history = model.fit_generator(
		train_generator,
		steps_per_epoch = 100,
		epochs = 100,
		validation_data=val_generator,
		validation_steps=50)
	return history

model = build_model()
history = train(model)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("test acc:", test_acc)

