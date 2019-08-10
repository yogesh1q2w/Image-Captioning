import tensorflow as tf 

class Encoder(tf.keras.Model):
    def __init__(self, embedDim):
        super(Encoder, self).__init__()
        self.fullyconn = tf.keras.layers.Dense(units = embedDim, activation=tf.nn.relu)

    def encode(self, embedding):
        embedding = self.fullyconn(embedding)
        return embedding
        
class Decoder(tf.keras.Model):
    def __init__(self, embedDim, numberOfCells, vocabSize, attentionMech):
        super(Decoder, self).__init__()
        self.numberOfCells = numberOfCells

        self.embedding = tf.keras.layers.Embedding(vocabSize, embedDim)
        self.gruCells = tf.keras.layers.GRU(self.numberOfCells, return_sequences=True, return_state=True,recurrent_initializer='glorot_uniform')
        self.fullyConn1 = tf.keras.layers.Dense(self.numberOfCells)
        self.fullyConn2 = tf.keras.layers.Dense(vocabSize)
        self.attentionMech = attentionMech

        if attentionMech:
            self.attention = Attention(self.numberOfCells)

    def decode(self, X, encoderOut, hidden):
        if self.attentionMech:
            contextVec, attenWts = self.attention.attention(encoderOut, hidden)
        
            X = self.embedding(X)
            X = tf.concat([tf.expand_dims(contextVec,1),X], axis = -1)
            finalOut, finalState = self.gruCells(X)
            X = self.fullyConn1(finalOut)
            X = tf.reshape(X,(-1,X.shape[2]))
            X = self.fullyConn2(X)

            return X, finalState, attenWts
        
        else:
            X = self.embedding(X)
            finalOut, finalState = self.gruCells(X)
            X = self.fullyConn1(finalOut)
            X = tf.reshape(X,(-1,X.shape[2]))
            X = self.fullyConn2(X)

            return X, finalState

    def reset_cell(self, N):
        return tf.zeros((N, self.numberOfCells))


class Attention(tf.keras.Model):
    def __init__(self, hiddenLayerSize):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(hiddenLayerSize)
        self.W2 = tf.keras.layers.Dense(hiddenLayerSize)
        self.out = tf.keras.layers.Dense(1)

    def attention(self, encoderOut, hidden):
        hiddenExtended = tf.expand_dims(hidden,1)
        attenVal = tf.nn.tanh(self.W1(encoderOut) + self.W2(hiddenExtended))
        attenWts = tf.nn.softmax(self.out(attenVal), axis = 1)
        contextVec = attenWts * encoderOut
        contextVec = tf.reduce_sum(contextVec, axis = 1)
        
        return contextVec, attenWts
        