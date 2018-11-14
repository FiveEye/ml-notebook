import keras as ks
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = ks.Input(shape=(latent_dim,))

x = layers.Dense(128*16*16)(generator_input)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16,16,128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = ks.models.Model(generator_input, x)

generator.summary()

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128,3)(discriminator_input)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)
#x = layers.Dense(2, activation='softmax')(x)

discriminator = ks.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = ks.optimizers.RMSprop(lr = 0.0008, 
											 clipvalue=1.0, 
											 decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = ks.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = ks.models.Model(gan_input, gan_output)

gan_optimizer = ks.optimizers.RMSprop(lr=0.0004, 
								   clipvalue=1.0, 
								   decay=1e-8)

gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


#encoder_input = layers.Input(shape=(height, width, channels))
encoder_input = discriminator_input
x = layers.Conv2D(128,3)(encoder_input)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
#x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(latent_dim, activation='sigmoid')(x)

encoder = ks.models.Model(discriminator_input, x)


#encoder.trainable = False

#generator.trainable = True

encoder_gan = ks.models.Model(discriminator_input, generator(encoder(discriminator_input)))

encoder_gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

#encoder.trainable = True

#generator.trainable = False

decoder_gan = ks.models.Model(generator_input, encoder(generator(generator_input)))

decoder_gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')



gan.load_weights('gan.h5')
encoder.load_weights('encoder.h5')


import os
from keras.preprocessing import image

(x_train, y_train), (_, _) = ks.datasets.cifar10.load_data()

x_train = x_train[y_train.flatten() == 6]
print(x_train.shape)
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
print(x_train.shape)

iterations = 100000
batch_size = 20
d_epochs = 1
a_epochs = 2
save_dir = 'gan'

start = 0

#d_labels = np.zeros((batch_size * 2, 2), dtype='float32')
#d_labels[0:batch_size,0] = 1.0
#d_labels[batch_size:,1] = 1.0

#r_labels = np.zeros((batch_size, 2), dtype='float32')
#r_labels[:,1] = 1.0

#g_labels = np.zeros((batch_size, 2), dtype='float32')
#g_labels[:,0] = 1.0

#a_labels = np.zeros((batch_size, 2), dtype='float32')
#a_labels[:,1] = 1.0

#def noise_labels(org_labels):
#	labels = org_labels + 0.1 * np.random.random(org_labels.shape)
#	sums = np.sum(labels, axis=1)
#	sums = sums.reshape((labels.shape[0], 1))
#	sums = np.concatenate([sums, sums], axis = 1)
#	return labels / sums

r_labels = np.zeros((batch_size, 1), dtype='float32')

g_labels = np.ones((batch_size, 1), dtype='float32')

def noise_labels(org_labels):
	if org_labels[0] == 0.0:
		labels = org_labels + 0.1 * np.random.random(org_labels.shape)
	else:
		labels = org_labels + 0.1 * np.random.random(org_labels.shape)
		labels /= np.max(labels)
	return labels

step = 0
#for step in range(1,iterations+1):
while True:
	step += 1
	#print('step:', step)
	stop = start + batch_size
	real_images = x_train[start:stop]
	d_loss = 0.0
	for _ in range(d_epochs):
		d_random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
		generated_images = generator.predict(d_random_latent_vectors)
		#encoder_gan.train_on_batch(generated_images, generated_images)
		#encoder_gan.train_on_batch(real_images, real_images)
		d_loss += discriminator.train_on_batch(generated_images, noise_labels(g_labels))
		
		d_loss += discriminator.train_on_batch(real_images, noise_labels(r_labels))
		start += batch_size
		if start > len(x_train) - batch_size:
			start = 0
			np.random.shuffle(x_train)
		
	d_loss /= (d_epochs * 2)
	

	
	#combined_images = np.concatenate([generated_images, real_images])
	
	#labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
	
	#labels = d_labels + 0.025 * np.random.random(d_labels.shape)
	#sums = np.sum(labels, axis=1)
	#sums = sums.reshape((batch_size * 2, 1))
	#sums = np.concatenate([sums, sums], axis = 1)
	#labels /= sums
	
	#d_loss = discriminator.train_on_batch(combined_images, labels)
	
	a_loss = 0.0
	for _ in range(a_epochs):
		a_random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
		decoder_gan.train_on_batch(a_random_latent_vectors, a_random_latent_vectors)
		encoder_gan.train_on_batch(real_images, real_images)
		a_random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
		a_loss += gan.train_on_batch(a_random_latent_vectors, r_labels)
		#decoder_gan.train_on_batch(a_random_latent_vectors, a_random_latent_vectors)
	a_loss /= a_epochs
	#for i in range(1, 100):
	#	random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
	#	a_loss = gan.train_on_batch(random_latent_vectors, r_labels)
	#	if a_loss < d_loss + i:
	#		break
	#print('step:', step)
	#print('d_loss:', d_loss)
	#print('a_loss:', a_loss)
	if step % 100 == 0:
		print('step:', step)
		print('d_loss:', d_loss)
		print('a_loss:', a_loss)
		d_err = discriminator.predict(generated_images).ravel()
		print(d_err, d_err.mean())
		
		generated_images_after = generator.predict(d_random_latent_vectors)
		d_err_after = discriminator.predict(generated_images_after).ravel()
		print(d_err_after, d_err_after.mean())
		print(d_err_after - d_err)
		a_err = gan.predict(a_random_latent_vectors).ravel()
		print(a_err, a_err.mean())
		if step % 500 == 0:
			gan.save_weights('gan.h5')
			encoder.save_weights('encoder.h5')
			
			ind = np.argmin(d_err_after)
			img = image.array_to_img(generated_images[ind] * 255.0, scale=False)
			img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
			

			img = image.array_to_img(generated_images_after[ind] * 255.0, scale=False)
			img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '_after.png'))
			
			
			#img = image.array_to_img(real_images[0] * 255., scale=False)
			#img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png')) 

