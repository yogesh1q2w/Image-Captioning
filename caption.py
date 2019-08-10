import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from encoder_decoder import Encoder, Decoder, Attention
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu


nTrain = 50
miniBatchSize = 4


a = pd.read_csv('flickr30k_images/results.csv', dtype = 'str', delimiter = '|', header = 0)
captions = np.array(a)

captionCollect = []
pathsCollect = []

for i,cap in enumerate(captions):
    cap[2] = '<s> ' + str(cap[2]) + ' <e>'
    imageid = cap[0]
    path = 'flickr30k_images/flickr30k_images/' + imageid
    captionCollect.append(cap[2])
    pathsCollect.append(path)

images, captions = shuffle(pathsCollect,captionCollect,random_state = 1)
images = images[:nTrain]
captions = captions[:nTrain]

def image_loader(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,(299,299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, path

imageTrain = images
captionTrain = captions

model = tf.keras.applications.InceptionV3(include_top=False,weights = 'imagenet')

newInput = model.input
hiddenLayer = model.layers[-1].output

extractedFeaturesModel = tf.keras.Model(newInput,hiddenLayer)

imageSet = sorted(set(images))
imageDataset = tf.data.Dataset.from_tensor_slices(imageSet)
imageDataset = imageDataset.map(image_loader,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(miniBatchSize)


for image,path in imageDataset:
    batchFeatures = extractedFeaturesModel(image)
    batchFeatures = tf.reshape(batchFeatures,(batchFeatures.shape[0],-1,batchFeatures.shape[3]))
    
    for b,p in zip(batchFeatures,path):
        pathFeature = p.numpy().decode('utf-8')
        np.save(pathFeature, b.numpy())


outVocabSize = 500
# print(captionTrain)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = outVocabSize, oov_token = '<unk>', filters= '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(captionTrain)
# print('--------------------------------------------------------------------------')
# print(captionTrain)
trainSeqCaptions = tokenizer.texts_to_sequences(captionTrain)
# print(trainSeqCaptions)
# print(tokenizer.index_word[3])

print(tokenizer.word_index['<s>'])

tokenizer.word_index['<p>'] = 0
tokenizer.index_word[0] = '<p>'

captionPadded = tf.keras.preprocessing.sequence.pad_sequences(trainSeqCaptions,padding='post')

maxAttenLen = max(len(t) for t in trainSeqCaptions)

imageNameTrain, imageNameTest, captionTrain, captionTest = train_test_split(images,captionPadded,test_size = .2, random_state = 1)

# print(len(imageNameTrain),len(imageNameTest),len(captionTrain),len(captionTest))

def mapping(imageName, caption):
    imageVec = np.load(imageName.decode('utf-8')+'.npy')
    return imageVec, caption


bufferSize = 1000

embedDim = 256
decoderNumCells = 512
inVocabSize = len(tokenizer.word_index) + 1
numTrain = len(imageNameTrain) + 0.
featureInpSize = 2048
attentionFeatureShape = 64

trainData = tf.data.Dataset.from_tensor_slices((imageNameTrain,captionTrain))

trainData = trainData.map(lambda i, c: tf.numpy_function(mapping,[i,c],[tf.float32,tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
trainData = trainData.shuffle(buffer_size = bufferSize).batch(miniBatchSize)
trainData = trainData.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

encoder = Encoder(embedDim)
decoder = Decoder(embedDim, decoderNumCells, inVocabSize, True)

optimizer = tf.keras.optimizers.Adam()

def loss(YPred,YTrue):
    YTrue = tf.cast(YTrue, dtype = tf.float32)
    logits = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(YPred,YTrue))
    return logits

checkPointPath = "model"
chkpt = tf.train.Checkpoint(encoder = encoder, decoder = decoder)
chkptMgr = tf.train.CheckpointManager(chkpt, checkPointPath, max_to_keep = 10)

if chkptMgr.latest_checkpoint:
    start = int(chkptMgr.latest_checkpoint.split('-')[-1])
else:
    start = 0

train = 1
test = 1
epochs = 50


if train:
    for epoch in range(epochs):
        epochLoss = 0
        lossImage = 0
        # print("HEYYYYYYY")
        for (batch, (trainBatchImage,trainBatchTarget)) in enumerate(trainData):

            hidden = decoder.reset_cell(trainBatchTarget.shape[0])
            # print("HEYYYYYYY")
            decoderInp = tf.expand_dims([tokenizer.word_index['<s>']] * miniBatchSize, 1)

            with tf.GradientTape() as tape:

                encoderOut = encoder.encode(trainBatchImage)

                for i in range(1,trainBatchTarget.shape[1]):
                    nextInp, nextState, _ = decoder.decode(decoderInp, encoderOut, hidden)

                    lossImage += loss(trainBatchTarget[:,i], nextInp)

                    decoderInp = tf.expand_dims(trainBatchTarget[:,i],1)

            totalLoss = lossImage/float(trainBatchTarget.shape[1])

            trainVars = encoder.trainable_variables + decoder.trainable_variables

            grads = tape.gradient(lossImage, trainVars)

            optimizer.apply_gradients(zip(grads, trainVars))
            
            epochLoss += lossImage

            if epoch % 3 == 0:
                chkptMgr.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, epochLoss/numTrain))


def eval(image):
    # print(image)
    hidden = decoder.reset_cell(1)
    inp = tf.expand_dims(image_loader(image)[0], 0)
    imageVec = extractedFeaturesModel(inp)
    imageVec = tf.reshape(imageVec,(imageVec.shape[0],-1,imageVec.shape[3]))
    
    encoderOut = encoder.encode(imageVec)

    decoderInp = tf.expand_dims([tokenizer.word_index['<s>']],0)
    result = []

    for i in range(maxAttenLen):
        nextInp, nextState, _ = decoder.decode(decoderInp, encoderOut, hidden)
        predWordID = tf.argmax(nextInp[0]).numpy()
        result.append(tokenizer.index_word[predWordID])

        if tokenizer.index_word[predWordID] == '<e>':
            return result

        decoderInp = tf.expand_dims([predWordID],0)

    return result


if test:
    score = 0
    for i in range(len(imageNameTest)):
        print('----------------------------------')
        image = imageNameTest[i]
        realCaption = []
        for (j,x) in enumerate(imageNameTest):
            if x == image:
                realCaption.append(list(captionTest[j]))

        result = eval(image)
        print (result)
        print (image)
        for (i,x) in enumerate(realCaption):
            print ('\nCaption ',i+1)
            for j in x:
                print (tokenizer.index_word[j], end=' ')
                if tokenizer.index_word[j] == '<e>':
                    break

    
        # plt.imshow(mpimg.imread(image))
        # plt.show()
        score += sentence_bleu(realCaption, result)
        print('\n',score)
        print('-------------------------------')
    
    print('BLEU Score = ', (score/(1.*len(imageNameTest))))
    
