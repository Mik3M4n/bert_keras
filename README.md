# Keras pretrained BERT

This repository contains an implementation in Keras of BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art pre-training model for Natural Language Procesing [released by Google AI](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) and avaiable in the [original Tensorflow implementation](https://github.com/google-research/bert) and in a [re-implementation in pytorch](https://github.com/huggingface/pytorch-pretrained-BERT).
Check out [the original paper](https://arxiv.org/abs/1810.04805) for details.


**Important: at this stage, the implementation works only as a feature extractor, i.e. it's not yet possible to re-train or fine-tune the pre-trained weights. See the following for details.**



## Overview and code organization

The package provides:

* Two `Keras BERT layers`, contained in `modeling.py`:
	* BertModel - raw BERT Transformer model (not pre-trained),
	* PreTrainedBertModel - raw BERT Transformer model (fully pre-trained)

* One `Binary classifier` BERTClassifier - Keras model for binary classification with pre-trained BERT embeddings (contained in `run_classifier.py`)

* Two examples of usage (see section examples)
	* extract_features.py - Show how to extract hidden states from an instance of PreTrainedBertModel,
	* run_classifier.py - Show how to fine-tune an instance of BERTClassifier on SST2 classification task
	
 * Explanatory `notebooks` (see dedicated section)
    
* Outils for tokenization in `tokenization.py`.

* A configuration class (in the `modeling.py` file):

	* BertConfig - Configuration class to store the configuration of a BertModel with utilities to read and write from JSON configuration files.
	

The first time the code is executed, some folders will be created: 
    
* `pre_trained_models` is where the pre-trained weights from Google will be saved. They will be automatically fetched from the release of Oct 18, 2018. To change the default model, edit the BERT_URL variable in data_utils.py. Alternatively, one can download different releases (see the [BERT release on GitHub](https://github.com/google-research/bert) for updates) and pass the corresponding folder with the ```model_dir``` argument (see Usage and Examples).

* `bert_google` is the folder where the [implementation in tensorflow](https://github.com/google-research/bert) will be cloned for comparison.

* `data` is the folder where data for classifications will be be downloaded.



More documentation and comments are provided inside the files.



## Usage

Here we provide an example of usage. For an example text, we extract the output of last layer of BERT

```
# Example input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"

# Load model tokenizer and get input features
bert_tokenizer = BertTokenizer(model_dir='pre_trained_models/eng_embeddings/uncased_L-12_H-768_A-12/',
                           input_file=None, 
                           input_list=[text],
                           max_seq_length=128)
                           
# Load BERT model with pre-trained weights
model_dir='pre_trained_models/eng_embeddings/uncased_L-12_H-768_A-12/'
model_pre_trained = PreTrainedBertModel(model_dir, Verbose=True)


# Compute embeddings
bert_embeddings = model_pre_trained(K.variable(bert_tokenizer.all_input_ids))

# Evaluate last layer
cls_layer = K.eval(bert_embeddings[-1])

```

## Doc

### 1. Feature extraction

The layer `PreTrainedBertModel` can be used as a feature extractor as shown in the section Usage. 


### 2. Binary classification

For binary classification tasks, see the class `BERTClassifier` contained in run_classifier.py

The usage of BERT implemented in this version is as simple as a regular Keras embedding layer. The structre for binary classification is just Embedding-Dropout-Dense with output dimension of the dense layer equal to the number of classes.

The usage of the BERT pre-trained model is just analogue to a binary classifier with an embedding layer:


```
bert = PreTrainedBertModel(model_dir= < path to saved BERT pre-trained weights >)

dropout = Dropout(self.dropout_prob, trainable=True)

dense = Dense(input_shape=(768,), 
							units= <number of classes >, \
                           trainable=True, 
                           activation='softmax')
                        

classifier = Sequential()
classifier.add(self.bert)
classifier.add(self.dropout)
classifier.add(self.dense)

model = Model(inputs= < input >, outputs=classifier(<input>) )

model.compile(loss= < loss >, optimizer= < optimizer >, metrics=['accuracy'])


```

**In this version, only the parameters of the final dense layer are trainable.**

## Examples


### Feature extraction

In `extract_features.py` we provide a working example of feature extraction and save the output in a .json file. Usage: 



```
> DATA_DIR=data/
> RES_DIR=results/
> mkdir $DATA_DIR
> mkdir $RES_DIR
> echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' >$DATA_DIR'input_dummy.txt'
>  python extract_features.py \
  --input_file $DATA_DIR'input_dummy.txt' \
  --output_file $RES_DIR'output_dummy_keras.jsonl' \
  --layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12\
  --max_seq_length 128 \
  --batch_size 1
  
```


### Binary single-sentence classification

In `run_classifier.py` we provide an example of binary text classification with the Stanford Sentiment Treebank 2 (SST2) dataset. 
The data are downloaded from the GItHub repository [AcademiaSinicaNLPLab](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset.git). To adapt the classifier to a different dataset, edit the function ```load_dataset``` in `run_classifier.py`.

Usage: 


```
> python run_classifier.py \
--output_file 'results/test_classifier.h5' \
--max_seq_length=128 \
--batch_size=32 \
--task_name 'SST2' \
--learning_rate 1e-4 \
--num_train_epochs 30 \
--num_classes 2 
```

Note that in case the data and model have already been downloaded, one can provide as input arguments also --model_dir and --data_dir tho specify their location. 

Example of output:


```
....

Epoch 30/30
60934/60934 [==============================] - 1099s 18ms/step - loss: 0.4843 - acc: 0.7716 - val_loss: 0.4286 - val_acc: 0.8163
Predicting....
7966/7966 [==============================] - 112s 14ms/step

Loss, accuracy:
[0.42064712371863283, 0.8135827266328933]

Saving to:  results/test_classifier.h5
Done.

```


## Notebooks

We provide some explanatory jupyter notebooks

* test\_extract_features extracts the hidden states of a full sequence on each layers of the TensorFlow and the Keras models and computes the standard deviation between them. We get std of order e-07 (*note that this has been obtained with Google's release of November 2018. We don't assure mantainance in the future.*)

	Output:

	```
	STANDARD DEVIATION TF-KERAS: 
	N. of layer, shape google layer, shape keras layer, standard deviation: 
	('Layer #1', (16, 768), (16, 768), 5.831007720901227e-07)
	('Layer #2', (16, 768), (16, 768), 6.765222033798338e-07)
	('Layer #3', (16, 768), (16, 768), 6.32635649157223e-07)
	('Layer #4', (16, 768), (16, 768), 7.203898864785463e-07)
	('Layer #5', (16, 768), (16, 768), 6.91689413529853e-07)
	('Layer #6', (16, 768), (16, 768), 6.629126073640179e-07)
	('Layer #7', (16, 768), (16, 768), 7.279752170693516e-07)
	('Layer #8', (16, 768), (16, 768), 6.503329475787774e-07)
	('Layer #9', (16, 768), (16, 768), 6.095085323779151e-07)
	('Layer #10', (16, 768), (16, 768), 7.068190001371195e-07)
	('Layer #11', (16, 768), (16, 768), 6.75920475783594e-07)
	('Layer #12', (16, 768), (16, 768), 5.382504451160329e-07)

	```

* run\_classifier\_loss_plot performs a binary classification task on the SST2 dataset and plots the loss and accuracy at each epoch.


## Comparison to other existing implementations


As already pointed out in the introduction, in this version is not possible to fine-tune the pre-treined weights. To this end, one would need to manually set the parameters to 'trainable' in each layer in `modeling.py`.

This said, the code organization follows closely that of the original tensorflow implementation. 

#### Tensorflow and pytorch

In the [original Tensorflow implementation](https://github.com/google-research/bert) and in the [pytorch version](https://github.com/huggingface/pytorch-pretrained-BERT), more tasks are implemented, while here we focus only on feature extraction and binary classification. However, the code structure allows to easily implement other tasks (e.g. question answering), by inheriting from the BertModel (or PreTrainedBertModel) class and implementing the call method. This also allows to compare to existing results in tensorflow and pytorch, as shown for the feature extraction case.

In particular, we obtain a ~10^-7 standard deviation between our model and the tensorflow version by Google, which indicates very good agreement (see the Notebooks section). The pytorh implementation has also been verified to have the same order of magnitude standard deviation with respect to Google's verison. We then conclude that the three are compatible.


#### Other keras implementations

Other implementations in Keras exist. [This version](https://github.com/Separius/BERT-keras) fully re-implements the model, althought the structure of the code is different from the original one and generalization to different tasks seems less straightforward. [This work](https://github.com/strongio/keras-bert) integrates BERT in Keras with a tensorflow hub. Finally, [this version](https://github.com/CyberZHG/keras-bert) exists. 

We didn't check the agreement of our version with Keras implementations, having already obtained a very good agreement with the original version.

To our knowledge, this Keras version is the one following most closely the original one, thus allowing easy generalization to different tasks and comparison of results. On the other hand, in this version we still need to make the BERT layer weights trainable in order to perform fine-tuning tasks.

## Acknowledgements

This work has been released during the School of Artificial Intelligence held by [Pi SCHOOL](https://picampus-school.com/) in Rome, Italy, from October to December 2018. It is part of a project sponsored by Cloud-Care SrL.