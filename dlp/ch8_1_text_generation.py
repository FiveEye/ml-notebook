import numpy as np
import keras as ks
from keras import layers

def reweight_dist(org_dist, temp=0.5):
	dist = np.log(org_dist) / temp
	dist = np.exp(dist)
	return dist / np.sum(dist)

path = ks.utils.get_file(
	'nietzsche.txt',
	origin='http://s3.amazonaws.com/text-datasets/nietzsche.txt')

text=open(path).read().lower()

print('Corpus length:', len(text))

maxlen = 60

step = 3

sentences = []

next_chars = []

for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i:i+maxlen])
	next_chars.append(text[i+maxlen])

print('Number of seq:', len(sentences))

chars = sorted(list(set(text)))
print('unique char:', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i,t,char_indices[char]] = 1
	y[i,char_indices[next_chars[i]]] = 1

model = ks.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = ks.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temp=0.1):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temp
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1,preds,1)
	return np.argmax(probas)
	
import random
import sys

model = ks.models.load_model("text_gen.h5")

for epoch in range(1,60):
	print('epoch', epoch)
	model.fit(x, y, batch_size=128, epochs=1)
	model.save("text_gen.h5")
	start_index = random.randint(0, len(text) - maxlen - 1)
	generated_text = text[start_index:start_index+ maxlen]
	print(generated_text)
	
	for temp in [0.2, 0.5, 1.0, 1.2]:
		print('temp:', temp)
		generated_text = text[start_index:start_index+ maxlen]
		sys.stdout.write(generated_text)
		for i in range(400):
			sampled = np.zeros((1,maxlen, len(chars)))
			for i, char in enumerate(generated_text):
				sampled[0,t, char_indices[char]] - 1.0
			preds = model.predict(sampled, verbose=0)[0]
			next_index = sample(preds, temp)
			next_char = chars[next_index]
			
			generated_text += next_char
			generated_text = generated_text[1:]
			sys.stdout.write(next_char)
		print('')