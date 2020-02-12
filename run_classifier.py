

from modeling import PreTrainedBertModel, BertConfig, BERTModel
import argparse
from data_utils import *
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import losses
from keras.layers import BatchNormalization, Input, Dropout, Add , Dense, Layer, Softmax, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Adadelta
import random
import os
np.random.seed(112)
random.seed(112)


# ++++++++++++++++++++++++        CLASSIFIER WITH PRE-TRAINED BERT MODEL           +++++++++++++++++++++++++++++++


class BertTokenizerForClassification(BertTokenizer):

    def __init__(self, model_dir=None, input_file=None,input_list=[],max_seq_length=128, labels=None):

        super().__init__(model_dir=model_dir, input_file=input_file,\
                         input_list=input_list,max_seq_length=max_seq_length)
        self.labels=np.array(labels)
        self.data_train=None
        self.data_test = None
        self.labels_train=None
        self.labels_test=None
        self.labels_validation = None
        self.data_validation = None




class BERTClassifier(object):
        """
        Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
        """

        def __init__(self, bert_path=None, input_file=None, input_list=[], \
                     num_classes=2, labels=None, \
                     Verbose=False, dummy=False, \
                     split_train_test=True, test_size=0.1, val_size=0.1,batch_size=32,\
                     epochs=30, max_seq_length=128, \
                     learning_rate=1e-4, use_dropout=True, dropout_prob=0.1,max_delay=3, **kwargs):

            self.input_file=input_file
            self.input_list=input_list
            self.num_classes = num_classes
            self.dummy=dummy
            self.bert_path=bert_path
            self.Verbose=Verbose
            self.labels=labels
            self.split_train_test=split_train_test
            self.test_size=test_size
            self.val_size=val_size
            self.batch_size=batch_size
            self.max_seq_length=max_seq_length
            self.learning_rate=learning_rate
            self.use_dropout=use_dropout
            self.epochs=epochs
            self.max_delay=max_delay
            self.dropout_prob=dropout_prob
            
            # Prepare data
            self._generate_tokens()

            # Design and compile model
            self._add_input_op()
            self._build_model()
            self._add_loss()
            self._get_callbacks()



        def _get_callbacks(self):


            early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    patience=self.max_delay,
                                                    mode='min')

            self.callbacks = [early_stopping_callback]

        def _encode_labels(self):
            if self.Verbose:
                print('Shape of labels (BEFORE one-hot encoding): %s' % str(self.data.labels.shape))
            self.data.labels = to_categorical(self.data.labels)
            if self.Verbose:
                print('Shape of labels (AFTER  one-hot encoding): %s\n' % str(self.data.labels.shape))


        def _generate_tokens(self):
            """
            Generates tokens for the bert model.

            :return: after this method has been called, the class contains an attribute .data that includes:

            - all_input_ids
            - labels

            - data_train
            - data_test
            - labels_train
            - labels_test
            - data_validation
            - labels_validation

            """

            if self.Verbose:
                print("Generating tokens for BERT...")

            self.data = BertTokenizerForClassification(self.bert_path, \
                                                       input_file=self.input_file,input_list=self.input_list,\
                                           max_seq_length=self.max_seq_length, labels=self.labels)

            if self.Verbose:
                print("Bert tokens generated with shape :", self.data.all_input_ids.shape )

            self._encode_labels()


            if self.split_train_test:
                assert self.labels is not None

                # create random train/test split
                indices = np.arange(self.data.all_input_ids.shape[0])
                num_training_indices = int((1 - self.test_size) * self.data.all_input_ids.shape[0])
                np.random.shuffle(indices)
                train_indices = indices[:num_training_indices]
                test_indices = indices[num_training_indices:]


                # split the actual data
                self.data.data_train, self.data.data_test = self.data.all_input_ids[train_indices,:,:],\
                                                  self.data.all_input_ids[test_indices,:,:]
                self.data.labels_train, self.data.labels_test = self.data.labels[train_indices], self.data.labels[test_indices]


                assert self.data.data_train.shape[0]==self.data.labels_train.shape[0]
                assert self.data.data_test.shape[0] == self.data.labels_test.shape[0]

                if self.Verbose:
                    print("Training data set size ", self.data.data_train.shape)
                    if self.data.data_train.shape[0]<15:
                        print(self.input_list[train_indices])
                        print(self.data.labels[train_indices])

                    print("Test data set size ", self.data.data_test.shape)
                    if self.data.data_test.shape[0] < 15:
                        print(self.input_list[test_indices])
                        print(self.data.labels[test_indices])



        def _add_input_op(self):
            self.input = Input(shape=(self.data.all_input_ids.shape[1],\
                                      self.data.all_input_ids.shape[2],),\
                               dtype='int32', name='input')

        def _build_model(self):
            if self.Verbose:
                print("\n Loading BERT Embeddings...")
            self.bert = PreTrainedBertModel(bert_path=self.bert_path, Verbose=True, \
                                            dummy=self.dummy, return_layer_list=False)
            if self.dummy:
                self.bert.build((self.bert.config.max_seq_len,))
            print("Bert built")
            if self.use_dropout:
                self.dropout = Dropout(self.dropout_prob, trainable=True)
                print("Dropout built with dropout prob ", self.dropout_prob)
            self.dense = Dense(input_shape=(int(self.bert.config.hidden_size),), units=self.num_classes, \
                                    trainable=True, activation='softmax')
            print("Dense built")
            self.classifier = Sequential()
            self.classifier.add(self.bert)
            
            if self.use_dropout:
                self.classifier.add(self.dropout)
            self.classifier.add(self.dense)


        def _add_loss(self):

            self.model = Model(inputs=self.input,
                                       outputs=self.classifier(self.input) )

            print("Model defined")
            my_opt=Adam(lr=self.learning_rate)

            self.model.compile(loss=losses.binary_crossentropy, optimizer=my_opt,\
                               metrics=['accuracy'])

            if self.Verbose:
                print("\n ------ Model architecture: sequential")
                print(self.classifier.summary(90))
                print("\n ------ Model architecture: full")
                print(self.model.summary(90))

        def train(self, batch_size=32, validation_split=0.1):

            self.model.fit(x=self.data.data_train, y=self.data.labels_train,\
                           validation_split=validation_split,\
                           epochs=self.epochs, batch_size=batch_size, verbose=1,
                           callbacks=self.callbacks)


        def evaluate(self):
            res = self.model.evaluate(x=self.data.data_test, y=self.data.labels_test)
            return res

        def save(self, path):
            self.model.save(path)






def load_dataset(data_dir=None, task_name='SST2', sample=False, sample_size=1000):



    if data_dir is not None and not os.path.exists(data_dir):
        raise ValueError('Please provide valid data dir or set the value to None to automatically download data')
    elif data_dir is None:
        data_dir='data/'
        corpus_path = data_dir + task_name + '.pkl'
        if not os.path.exists(corpus_path):
            print("Downloading dataset for "+task_name+" ...")
            os.system('git clone https://github.com/AcademiaSinicaNLPLab/sentiment_dataset.git '+data_dir)
            print('Extracting data...')
            cur_dir = os.getcwd()
            os.chdir(data_dir)
            cmd='./preprocess.py -m corpus.yaml '+task_name
            os.system(cmd)
            os.chdir(cur_dir)
    else:
        corpus_path = data_dir + task_name + '.pkl'

    print("Loading %s dataset .... " %task_name)
    corpus = pd.read_pickle(corpus_path)
    sentences, labels = np.array(list(corpus.sentence)), np.array([int(l) for l in list(corpus.label)])

    if sample:
        idx = np.hstack([np.random.choice(np.where(labels == l)[0], int(np.floor(sample_size/2)), replace=False)
                      for l in np.unique(labels)])
        np.random.shuffle(idx)
        sentences = sentences[idx]
        labels = labels[idx]

        labels = labels[sentences != '']
        sentences = sentences[sentences != '']

    print("Sentences and labels shape:")
    print(sentences.shape)
    print(labels.shape)
    return sentences, labels



# ++++++++++++++++++++++++ main


def main():

    # Read input args
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="Directory where the data are stored")
    parser.add_argument("--model_dir", default=None, type=str, required=False,
                        help="Directory where the pre-trained weights are stored")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--task_name", default='', type=str, help="Dataset name")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Training epochs")
    parser.add_argument("--num_classes", default=2, type=int, help="Classes")

    args = parser.parse_args()

    # Read data
    print("Reading data....")
    sentences, labels = load_dataset(args.data_dir, task_name=args.task_name, sample=False, sample_size=1000)

    print("Building model....")
    model = BERTClassifier(bert_path=args.model_dir, input_list=sentences, \
                           num_classes=args.num_classes,\
                           batch_size= args.batch_size, \
                           labels=labels, \
                           Verbose=True, \
                           dummy=False,
                           epochs=args.num_train_epochs,\
                           learning_rate=args.learning_rate, max_delay=5, use_dropout=True, \
                           dropout_prob=0.1, test_size=0.1)
    print("Training model....")
    model.train(batch_size=args.batch_size, validation_split=0.15)

    print("Predicting....")
    res = model.evaluate()

    print("Loss, accuracy " + "\n")
    print(str(res))

    print("Saving to: ", args.output_file)

    
    model.save(args.output_file)
    print("Done.")



if __name__ == '__main__':
    main()
