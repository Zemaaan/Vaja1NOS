import math
import random
import numpy
from PIL import Image, ImageDraw
from numpy import asarray
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers


# kranjc david

class DataGenerator(tf.keras.utils.Sequence):

	def __init__(self, x_set=None, y_set=None, batch_size=16):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		return IzdejalSlike()


def IzdejalSlike():
	# Create empty black canvas
	im = Image.new('RGB', (256, 256))

	# Draw red and yellow triangles on it and save
	#
	SeznamOblikov = ['Trikotnik'] * 2
	SeznamOblikov += ['Stirikotnik'] * 2
	SeznamOblikov += ['Elipsa'] * 2
	SeznamOblikov += ['Zvezda'] * 2

	random.shuffle(SeznamOblikov)
	# print(SeznamOblikov)

	for lik in SeznamOblikov:
		if lik == 'Trikotnik':
			draw = ImageDraw.Draw(im)
			PointOneX = random.randrange(0, 255)
			PointOneY = random.randrange(0, 255)

			PointTwoX = random.randrange(0, 255)
			PointTwoY = random.randrange(0, 255)

			PointThreeX = random.randrange(0, 255)
			PointThreeY = random.randrange(0, 255)

			BarvaRKomponenta = random.randrange(0, 255)
			BarvaGKomponenta = random.randrange(0, 255)
			BarvaBKomponenta = random.randrange(0, 255)

			draw.polygon([(PointOneX, PointOneY), (PointTwoX, PointTwoY), (PointThreeX, PointThreeY)],
						 fill=(BarvaRKomponenta, BarvaGKomponenta, BarvaBKomponenta))
			im.save('256.png')

		elif lik == 'Stirikotnik':

			draw = ImageDraw.Draw(im)

			PointOneXStirikotnik = random.randrange(0, 255)
			PointOneYStirikotnik = random.randrange(0, 255)

			PointTwoXStirikotnik = random.randrange(0, 255)
			PointTwoYStirikotnik = random.randrange(0, 255)

			PointThreeXStirikotnik = random.randrange(0, 255)
			PointThreeYStirikotnik = random.randrange(0, 255)

			PointFourXStirikotnik = random.randrange(0, 255)
			PointFourYStirikotnik = random.randrange(0, 255)

			BarvaRKomponenta = random.randrange(0, 255)
			BarvaGKomponenta = random.randrange(0, 255)
			BarvaBKomponenta = random.randrange(0, 255)

			draw.polygon([(PointOneXStirikotnik, PointOneYStirikotnik), (PointTwoXStirikotnik, PointTwoYStirikotnik),
						  (PointThreeXStirikotnik, PointThreeYStirikotnik),
						  (PointFourXStirikotnik, PointFourYStirikotnik)],
						 fill=(BarvaRKomponenta, BarvaGKomponenta, BarvaBKomponenta))
			im.save(r'/home/hrvoje/PycharmProjects/Vaja1NOS/256.png')

		elif lik == 'Elipsa':

			PointOneXElipsa = random.randrange(0, 255)  # Gori desno
			PointOneYElipsa = random.randrange(0, 255)  # Gori desno

			PointTwoXElipsa = random.randrange(0, 255)  # doli Levo
			PointTwoYElipsa = random.randrange(0, 255)  # doli Levo

			shape = [(PointOneXElipsa, PointOneYElipsa), (PointTwoXElipsa, PointTwoYElipsa)]

			# create ellipse image
			img1 = ImageDraw.Draw(im)
			img1.ellipse(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
			im.save(r'/home/hrvoje/PycharmProjects/Vaja1NOS/256.png')

		elif lik == 'Zvezda':
			draw = ImageDraw.Draw(im)

			PointOneXZvezda = random.randrange(0, 255)
			PointOneYZvezda = random.randrange(0, 255)

			PointTwoXZvezda = random.randrange(0, 255)
			PointTwoYZvezda = random.randrange(0, 255)

			PointThreeXZvezda = random.randrange(0, 255)
			PointThreeYZvezda = random.randrange(0, 255)

			PointFourXZvezda = random.randrange(0, 255)
			PointFourYZvezda = random.randrange(0, 255)

			PointFiveXZvezda = random.randrange(0, 255)
			PointFiveYZvezda = random.randrange(0, 255)

			AverageX = (PointOneXZvezda + PointTwoXZvezda + PointThreeXZvezda + PointFourXZvezda + PointFiveXZvezda) / 5
			AverageY = (PointOneYZvezda + PointTwoYZvezda + PointThreeYZvezda + PointFourYZvezda + PointFiveYZvezda) / 5

			shape = [(PointOneXZvezda, PointOneYZvezda), (AverageX, AverageY)]
			draw.line(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

			shape = [(PointTwoXZvezda, PointTwoYZvezda), (AverageX, AverageY)]
			draw.line(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

			shape = [(PointThreeXZvezda, PointThreeYZvezda), (AverageX, AverageY)]
			draw.line(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

			shape = [(PointFourXZvezda, PointFourYZvezda), (AverageX, AverageY)]
			draw.line(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

			shape = [(PointFiveXZvezda, PointFiveYZvezda), (AverageX, AverageY)]
			draw.line(shape, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

			im.save(r'/home/hrvoje/PycharmProjects/Vaja1NOS/256.png')

	NumpyMatrikaKoncneSlike = asarray(cv2.imread(r'/home/hrvoje/PycharmProjects/Vaja1NOS/256.png'))
	print(NumpyMatrikaKoncneSlike)
	return NumpyMatrikaKoncneSlike


def DodajSum(SlikaMatrika):
	numpy.set_printoptions(precision=3)
	# print(numpy.around(SlikaMatrika, 3))
	row, col, ch = SlikaMatrika.shape
	SumAditivni = numpy.random.normal(0, 0.2, (row, col, ch))
	SumMultiplikativni = numpy.random.normal(1, 0.2, (row, col, ch))

	AditivniSum = SlikaMatrika + (SumAditivni * 255)
	MultiplikativniSum = SlikaMatrika * SumMultiplikativni

	cv2.imwrite(r'/home/hrvoje/PycharmProjects/Vaja1NOS/SlikaCorruptedAditivni256.png', AditivniSum)
	cv2.imwrite(r'/home/hrvoje/PycharmProjects/Vaja1NOS/SlikaCorruptedMultiplikativni256.png', MultiplikativniSum)
	SlikaMultiplikativniSum = cv2.imread(r'/home/hrvoje/PycharmProjects/Vaja1NOS/SlikaCorruptedAditivni256.png')
	SlikaAdiitivniSum = cv2.imread(r'/home/hrvoje/PycharmProjects/Vaja1NOS/SlikaCorruptedMultiplikativni256.png')

	return SlikaMultiplikativniSum, SlikaAdiitivniSum


# DodajSum(IzdejalSlike())

inputShape = (256, 256, 3)  # Tukaj imamo vhod, kateri se potem deli na dve veje

VhodPrvaVeja = keras.Input(shape=inputShape)  # [H, W, 3]
b = VhodPrvaVeja[:, :, :, 0:1]  # [N, H, W, 1]
g = VhodPrvaVeja[:, :, :, 1:2]  # [N, H, W, 1]
r = VhodPrvaVeja[:, :, :, 2:3]  # [N, H, W, 1]

layer = tensorflow.keras.layers.Conv2D(8, (11, 11), activation=tf.keras.activations.linear, name='BarvnaConv')

b_filts = layer(b)
g_filts = layer(g)
r_filts = layer(r)


VhodiSlojDrugaVeja = tensorflow.keras.layers.Input(batch_shape=(1, 256, 256, 3))

PrviResnet = tensorflow.keras.layers.Conv2D(32, (3, 3), activation=tensorflow.keras.activations.relu)(VhodiSlojDrugaVeja)
PrviResnetDropOut = tensorflow.keras.layers.Dropout(0.5)(PrviResnet)

DrugiResnet = tensorflow.keras.layers.Conv2D(32, (3, 3), activation=tensorflow.keras.activations.relu)(PrviResnetDropOut)
DrugiResnetDropOut = tensorflow.keras.layers.Dropout(0.5)(DrugiResnet)

TreciResnet = tensorflow.keras.layers.Conv2D(32, (3, 3), activation=tensorflow.keras.activations.relu)(DrugiResnetDropOut)
TreciResnet = tensorflow.keras.layers.Dropout(0.5)(TreciResnet)

Rezultat = tensorflow.keras.layers.Conv2D(8, (3, 3), activation=tensorflow.keras.activations.relu)(TreciResnet)
Rezultat = tensorflow.keras.layers.Softmax(3)(Rezultat)

# fali pretvorba z 32 kanalov v 8 razredov
# uporabimo 3 da dobimo 4 element - koji?? (visina, sirina, stevilo slik, ??)
# tensorflow.keras.layers.Conv2D(8, 11, activation='relu', input_shape=input_shape[1:])(x)

# ZdruzitevniSloj = tf.keras.layers.Concatenate()

ZdruzitevRdeca = tf.keras.layers.multiply([b_filts, Rezultat])
ZdruzitevZelena = tf.keras.layers.multiply([g_filts, Rezultat])
ZdruzitevPlava = tf.keras.layers.multiply([r_filts, Rezultat])


model = keras.Model(
	inputs=[VhodiSlojDrugaVeja, VhodPrvaVeja],
	outputs=[Rezultat],
)
# print((model.get_layer("BarvnaConv").weights))
# funkcija sum - keras
# funkcija concatenate
print(model.summary())
keras.utils.plot_model(model, "main.png", show_shapes=True)
