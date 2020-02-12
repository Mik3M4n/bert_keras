
import json
import six
import math
import tensorflow as tf
import copy
import os

from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Dropout, Add , Dense, Layer, Softmax, Activation
import keras.backend as K
from keras.initializers import Zeros, Ones
import numpy as np
from data_utils import *
import random

np.random.seed(112)
random.seed(112)
BERT_TRAINABLE=False

# ++++++++++++++++++++++++        CONFIGURATION           +++++++++++++++++++++++++++++++

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02,
                max_seq_len=128):
        """Constructs BertConfig.
        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.max_seq_len = max_seq_len

        self.trainable = BERT_TRAINABLE

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




# ++++++++++++++++++++++++        EMBEDDING LAYER           +++++++++++++++++++++++++++++++



class BERTLayerNorm(Layer):

    def __init__(self, config, variance_epsilon=1e-12, **kwargs):
        self.variance_epsilon = variance_epsilon
        self.trainable = False
        super(BERTLayerNorm, self).__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        if isinstance(input_shape, tuple) and input_shape[0] is None:
            input_shape = [input_shape[1]]

        elif (isinstance(input_shape, list) and len(input_shape) < 2) or type(input_shape) == int:
             input_shape = (input_shape, )

        self.beta = self.add_weight(name='beta',
                                    shape=np.array(input_shape),
                                    initializer=Zeros(),
                                    trainable=False)
        self.gamma = self.add_weight(name='gamma',
                                     shape=np.array(input_shape),
                                     initializer=Ones(),
                                     trainable=False)

        super(BERTLayerNorm, self).build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / K.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta



    

class BERTEmbeddings(Layer):
    
    def __init__(self, config, **kwargs):

        self.trainable = False
        super(BERTEmbeddings, self).__init__(**kwargs)

        self.config = config

        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size,
                                           name='token_type_embeddings', 
                                               trainable=False)
        
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size,
                                           name='position_embeddings', 
                                             trainable=False)
        
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size,
                                           name='token_embeddings',
                                         trainable=False)
        
        self.LayerNorm = BERTLayerNorm(config, trainable=False)
        
        self.dropout = Dropout(config.hidden_dropout_prob, name='EmbeddingDropOut', trainable=False)

    def build(self, input_shape):

        self.token_type_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)
        self.word_embeddings.build(input_shape)
        self.LayerNorm.build(self.config.hidden_size)
        self.dropout.build(self.config.hidden_size)

        super(BERTEmbeddings, self).build(self.config.hidden_size)

    def call(self, x, **kwargs):

        input_ids, token_type_ids, position_ids = x

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
        



# ++++++++++++++++++++++++        ENCODER LAYER(S)           +++++++++++++++++++++++++++++++



# --------- Multi-head attention layer

class BERTSelfAttention(Layer):
    
    def __init__(self, config, **kwargs):
        self.trainable = False
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config=config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(input_shape=(config.hidden_size,), units=self.all_head_size, trainable=False)
        self.key = Dense(input_shape=(config.hidden_size,), units=self.all_head_size, trainable=False)
        self.value = Dense(input_shape=(config.hidden_size,), units=self.all_head_size, trainable=False)

        self.dropout = Dropout(config.attention_probs_dropout_prob, trainable=False)

 
    def transpose_for_scores(self, x, k: bool = False):
        x_shape = list(x.shape)
        new_x_shape = [-1] + x_shape[-2:-1] + [self.num_attention_heads, self.attention_head_size]
        new_x = K.reshape(x, new_x_shape)
        if k:
            return K.permute_dimensions(new_x, [0, 2, 3, 1])
        else:
            return K.permute_dimensions(new_x, [0, 2, 1, 3])
    
    def build(self, input_shape):

        self.query.build((self.all_head_size,self.config.hidden_size))
        self.key.build((self.all_head_size,self.config.hidden_size))
        self.value.build((self.all_head_size,self.config.hidden_size))
        self.dropout.build(input_shape)
                
        super(BERTSelfAttention, self).build(input_shape)
        
    
    def call(self, x, **kwargs):
        
        hidden_states, attention_mask = x

        hidden_states_r = K.reshape(hidden_states, (-1, hidden_states.shape[-1]))

        # `query_layer` = [B*F, N*H]
        mixed_query_layer = self.query(hidden_states_r)

        # `key_layer` = [B*T, N*H]
        mixed_key_layer = self.key(hidden_states_r)

        # `value_layer` = [B*T, N*H]
        mixed_value_layer = self.value(hidden_states_r)

        mixed_query_layer_r = K.reshape(mixed_query_layer, (-1, self.config.max_seq_len, self.config.hidden_size))
        mixed_key_layer_r = K.reshape(mixed_key_layer, (-1, self.config.max_seq_len, self.config.hidden_size))
        mixed_value_layer_r = K.reshape(mixed_value_layer, (-1, self.config.max_seq_len, self.config.hidden_size))


        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(mixed_query_layer_r, k=False)
        
        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(mixed_key_layer_r, k=True)
        
        value_layer = self.transpose_for_scores(mixed_value_layer_r, k=False)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = K.batch_dot(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask is (precomputed for all layers in call to BertModel)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        context_layer = K.batch_dot(attention_probs, value_layer)
        context_layer = K.permute_dimensions(context_layer, [0, 2, 1, 3])
        new_context_layer_shape = [-1, self.config.max_seq_len, self.all_head_size]
        context_layer = K.reshape(context_layer, new_context_layer_shape)
        return context_layer


# --------- Sub-layers for encoder

def gelu(x):
    return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))


class BERTSelfOutput(Layer):
    
    def __init__(self, config, **kwargs):
        self.trainable = False
        super().__init__(**kwargs)   
        self.config=config
        self.dense = Dense(input_shape=(self.config.hidden_size,), units=self.config.hidden_size, trainable=False)
        self.LayerNorm = BERTLayerNorm(self.config, trainable=False)
        self.dropout = Dropout(self.config.hidden_dropout_prob, trainable=False)

    def build(self, input_shape):

        if isinstance(input_shape, tuple) and input_shape[0] is None:
            dense_input_shape = (self.config.hidden_size, input_shape[1])
        else:
            dense_input_shape = (self.config.hidden_size, input_shape)

        self.dense.build(dense_input_shape)

        self.LayerNorm.build(self.config.hidden_size)
        self.dropout.build(self.config.hidden_size)
        super(BERTSelfOutput, self).build(input_shape)
    
    def call(self, x, **kwargs):
        input_tensor, hidden_states = x

        original_shape = hidden_states.shape
        hidden_states_r = K.reshape(hidden_states, (-1, hidden_states.shape[-1]))

        hidden_states = self.dense(hidden_states_r)
        hidden_states = self.dropout(hidden_states)

        hidden_states_r = K.reshape(hidden_states, (-1, original_shape[1], original_shape[2]))
        hidden_states = self.LayerNorm(hidden_states_r + input_tensor)
        return hidden_states


class BERTAttention(Layer):
    
    def __init__(self, config, **kwargs):
        self.config=config
        self.trainable = False
        super().__init__( **kwargs)
        self.self = BERTSelfAttention(config, trainable=False)
        self.my_output = BERTSelfOutput(config, trainable=False)

    def build(self, input_shape):
        self.self.build(input_shape)
        self.my_output.build(input_shape)
        super(BERTAttention, self).build(input_shape)

        
    def call(self, x, **kwargs):
        input_tensor, attention_mask = x
        self_output = self.self(inputs=[input_tensor, attention_mask])
        attention_output = self.my_output(inputs=[input_tensor, self_output])
        return attention_output


class BERTIntermediate(Layer):
    def __init__(self, config, **kwargs):
        self.config=config
        self.trainable = False
        super().__init__( **kwargs)
        self.dense = Dense(input_shape=(self.config.hidden_size,), units=self.config.intermediate_size, trainable=False)
        self.intermediate_act_fn = gelu

    def build(self, input_shape):
        self.dense.build((self.config.intermediate_size, self.config.hidden_size))
        super(BERTIntermediate, self).build(input_shape)
        
    def call(self, x, **kwargs):
        hidden_states = x
        original_shape = hidden_states.shape
        hidden_states_r = K.reshape(hidden_states, (-1, hidden_states.shape[-1]))

        hidden_states = self.dense(hidden_states_r)
        hidden_states_r = K.reshape(hidden_states, (-1, original_shape[1], hidden_states.shape[-1]))
        hidden_states = self.intermediate_act_fn(hidden_states_r)
        return hidden_states

    

class BERTOutput(Layer):
    def __init__(self, config, **kwargs):
        self.config=config
        self.trainable = False
        super().__init__( **kwargs)
        self.dense = Dense(input_shape=(config.intermediate_size,), units= config.hidden_size, trainable=False)
        self.LayerNorm = BERTLayerNorm(config, trainable=False)
        self.dropout = Dropout(config.hidden_dropout_prob, trainable=False)

    def build(self, input_shape):
        self.dense.build((self.config.hidden_size, self.config.intermediate_size))
        self.LayerNorm.build(self.config.hidden_size)
        self.dropout.build(self.config.hidden_size)
        super(BERTOutput, self).build(self.config.hidden_size)
    
    def call(self, x, **kwargs):
        
        input_tensor, hidden_states = x

        original_shape = hidden_states.shape
        hidden_states_r = K.reshape(hidden_states, (-1, hidden_states.shape[-1]))

        hidden_states = self.dense(hidden_states_r)
        hidden_states = self.dropout(hidden_states)
        hidden_states_r = K.reshape(hidden_states, (-1, original_shape[1], hidden_states.shape[-1]))

        hidden_states = self.LayerNorm(hidden_states_r + input_tensor)
        return hidden_states





class BERTLayer(Layer):
       
    def __init__(self, config, **kwargs):
        self.trainable = False
        super().__init__(**kwargs)
        self.attention = BERTAttention(config, trainable=False)
        self.intermediate = BERTIntermediate(config, trainable=False)
        self.my_output = BERTOutput(config, trainable=False)
        
    def build(self, input_shape):
        self.attention.build(input_shape)
        self.intermediate.build(input_shape)
        self.my_output.build(input_shape)
        super(BERTLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        hidden_states, attention_mask = x
        attention_output = self.attention(inputs=[hidden_states, attention_mask])
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.my_output(inputs=[attention_output, intermediate_output])
        return layer_output



# --------- Encoder layer

class BERTEncoder(Layer):
    
    def __init__(self, config, **kwargs):
        self.trainable = False
        super().__init__(**kwargs)
        self.config=config
        self.model = Sequential()        
        for i in range(config.num_hidden_layers):
            self.model.add(BERTLayer(config, trainable=False))
    
    def build(self, input_shape):
        for i in range(self.config.num_hidden_layers):
            self.model.layers[i].build(input_shape)
        super(BERTEncoder, self).build(input_shape)
    
    def call(self, x, **kwargs):
        hidden_states, attention_mask = x
        all_encoder_layers_list = []
        for layer_module in self.model.layers:
            hidden_states = layer_module(inputs=[hidden_states, attention_mask])
            all_encoder_layers_list.append(hidden_states)

        return all_encoder_layers_list


    def compute_output_shape(self, input_shape):
        output_shapes = [[input_shape[1][0], input_shape[1][-1] , self.config.hidden_size] for _ in range(self.config.num_hidden_layers) ]
        return output_shapes



# ++++++++++++++++++++++++        POOLER LAYER           +++++++++++++++++++++++++++++++

class BERTPooler(Layer):
    def __init__(self, config, **kwargs):
        self.trainable = False
        self.config=config
        super().__init__(**kwargs)
        self.dense=Dense(input_shape=[config.hidden_size,], units= config.hidden_size, trainable=False, activation='tanh')

    def build(self, input_shape):

        if isinstance(input_shape, tuple) and input_shape[0] is None:
            pooler_input_shape = [self.config.hidden_size, input_shape[1]]
        else:
            pooler_input_shape = [self.config.hidden_size, input_shape]

        self.dense.build(pooler_input_shape)
        super(BERTPooler, self).build(input_shape)
        
    def call(self, hidden_states, **kwargs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)

        return [pooled_output]




# ++++++++++++++++++++++++        FULL BERT MODEL           +++++++++++++++++++++++++++++++


class BERTModel(Layer):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = K.variable(value=[1, 2, 3])
    input_mask = K.variable(value=[1, 1, 1])
    token_type_ids = K.variable(value=[0, 0, 1])

    config = BertConfig(vocab_size=32000, hidden_size=24,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertModel(config=config)
    model.build(config.hidden_size)

    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig, Verbose=True, return_layer_list=True,**kwargs):
        """Constructor for BertModel.
        Args:
            config: `BertConfig` instance.
        """
        self.config=config
        self.log_path = 'results/log.txt'
        self.trainable = False
        super(BERTModel, self).__init__(**kwargs)
        self.Verbose = Verbose
        self.feature_size = self.config.hidden_size
        self.return_layer_list=return_layer_list
        self._build_model()

        if self.Verbose:
            print('------ BERT model initialized. ')

    def _build_model(self):

        self.embeddings = BERTEmbeddings(self.config, trainable=False)
        self.encoder = BERTEncoder(self.config, trainable=False)
        self.pooler = BERTPooler(self.config, trainable=False)

    def build(self, input_shape):
        self.embeddings.build(input_shape)
        self.encoder.build(self.config.hidden_size)
        self.pooler.build(self.config.hidden_size)
        super(BERTModel, self).build(input_shape)

    def parse_input(self, x):

        input_ids = K.squeeze(K.slice(x, [0, 0, 0], [-1, 1, self.config.max_seq_len]), 1)
        attention_mask = K.cast(K.squeeze(K.slice(x, [0, 1, 0], [-1, 1, self.config.max_seq_len]), 1),
                                dtype="float32")
        token_type_ids = K.squeeze(K.slice(x, [0, 2, 0], [-1, 1, self.config.max_seq_len]), 1)

        position_ids = K.squeeze(K.slice(x, [0, 3, 0], [-1, 1, self.config.max_seq_len]), 1)

        return input_ids, attention_mask, token_type_ids, position_ids


    def call(self, x,  **kwargs):

        input_ids, attention_mask, token_type_ids, position_ids = self.parse_input(x)

        extended_attention_mask = K.expand_dims(attention_mask, axis=1)
        extended_attention_mask = K.expand_dims(extended_attention_mask, axis=2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(inputs=[input_ids, token_type_ids, position_ids])

        all_encoder_layers = self.encoder(inputs=[embedding_output, extended_attention_mask])

        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if self.return_layer_list:
            return all_encoder_layers+[pooled_output]
        else:
            return pooled_output

    def compute_output_shape(self, input_shape):

        output_shape_layers = [(input_shape[0], input_shape[-1], self.config.hidden_size) for _ in range(self.config.num_hidden_layers)]
        output_shape_pooler = [( input_shape[0], self.config.hidden_size )]
        out_shape = output_shape_layers+output_shape_pooler

        if self.return_layer_list:
            return out_shape
        else:
            return output_shape_pooler



# ++++++++++++++++++++++++        PRE-TRAINED BERT MODEL           +++++++++++++++++++++++++++++++




def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def get_weights_dict(names, arrays):
    d = {}
    for my_n, my_vals in zip(names, arrays):
        keys = my_n.split('/')[1:]
        nested_set(d, keys, my_vals)
    return d


def getFromDict(dataDict, mapList):
    dataDict1 = copy.deepcopy(dataDict)
    for k in mapList: dataDict = dataDict[k]
    return dataDict


def multi_getattr(obj, attr, default=None):
    """
    Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
    equivalent to x.a.b.c.d. When a default argument is given, it is
    returned when any attribute in the chain doesn't exist; without
    it, an exception is raised when a missing attribute is encountered.

    """
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj


class PreTrainedBertModel(BERTModel):

    def __init__(self, model_dir = None,
                 bert_url=BERT_URL,
                 Verbose=True, dummy=False, return_layer_list=True,**kwargs):
        """Constructor for PreTrainedBertModel.
        Args:
            config: `BertConfig` instance.
        """

        if model_dir is not None and not os.path.exists(model_dir):
            raise ValueError('Please provide a valid path to BERT pre-trained weights or set the value to None to automatically download the model weights')

        elif model_dir is None:

            folder_name = 'pre_trained_models/'
            model_name = bert_url.split('/')[-1]
            cmd = "mkdir "+folder_name+"  ; "+ \
                  "wget "+ bert_url+ " -d "+folder_name+"  ;" +\
                "unzip" +model_name+ " -d "+folder_name

            model_dir = folder_name + model_name.split('.')[0] + '/'
            if not os.path.exists(model_dir):
                print ("Downloading pre-trained weights...")
                os.system(cmd)



        self.dummy=dummy
        self.model_path = model_dir+'bert_model.ckpt'
        self.config_path = model_dir+'bert_config.json'
        self.vocab_path = model_dir+'vocab.txt'
        self.Verbose = Verbose
        self.return_layer_list=return_layer_list
        self.config = BertConfig.from_json_file(self.config_path)
        super().__init__(config=self.config, Verbose=Verbose, \
                         trainable=False,return_layer_list=self.return_layer_list, **kwargs)


        self.build((self.config.max_seq_len, ))
        if not self.dummy:
            if self.Verbose:
                print('------ Filling with pre-trained weights... ')
            self.get_BERT_weights(self.model_path, Verbose=False)
        else:
            print("Using randomly initialized weights")

    def _swap_dense_params(self, names):
        my_names = copy.deepcopy(names)
        all_dense = [i for i, s in enumerate(names) if '/bias' in s]
        for index in all_dense:
            next_index = index + 1
            my_names[index], my_names[next_index] = my_names[next_index], my_names[index]

        return my_names

    def _load_BERT(self, model_path):
        """
        Load tf model.
        Return  - names: a list of names of layers (206)
            - arrays: list of correstponding weights
        """

        init_vars = tf.train.list_variables(model_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            # print("Loading {} with shape {}".format(name, shape))
            array = tf.train.load_variable(model_path, name)
            # print("Numpy array shape {}".format(array.shape))
            names.append(name)
            arrays.append(array)
        return self._swap_dense_params(names), arrays

    def _get_BERT_layer(self, my_n_split):
        """
        given a model and a layer name in the form of a list ['encoder', 'layer_7', 'intermediate', 'dense'],
        returns corresponding model layer and name
        """
        if any('layer_' in string for string in my_n_split):

            layer_index = [i for i, s in enumerate(my_n_split) if 'layer_' in s][0]
            layer_number = int(my_n_split[layer_index].split('_')[1])
            my_layer_name_temp = ('.').join(my_n_split[1:layer_index]) + '.model'  # +'layer_'+str(layer_number)

            my_temp_layer = multi_getattr(self, my_layer_name_temp)

            my_temp_layer_1 = my_temp_layer.layers[layer_number]
            my_layer_name = ('.').join(my_n_split[layer_index + 1:])

            my_layer = multi_getattr(my_temp_layer_1, my_layer_name)
        else:
            my_layer_name = ('.').join(my_n_split[1:])
            my_layer = multi_getattr(self, my_layer_name)

        return my_layer, my_layer_name

    def get_BERT_weights(self, path, Verbose=False):
        """
        Fills model with pre-trained bert weights
        """

        if Verbose:
            print('------ Loading bert pre-trained weights.....')

        names, arrays = self._load_BERT(path)
        weights_dict_full = get_weights_dict(names, arrays)

        if Verbose:
            print('------ Filling layers with bert pre-trained weights..........')

        filled = []

        for my_n in names:
            layer_type = None
            my_n_split = my_n.split('/')
            my_n_split_or = my_n_split
            my_n_split = [word if word != 'output' else 'my_output' for word in my_n_split]

            if my_n_split[0] == 'bert':

                if 'LayerNorm' in my_n_split or 'dense' in my_n_split \
                        or 'key' in my_n_split or 'query' in my_n_split or 'value' in my_n_split:
                    if 'LayerNorm' in my_n_split:
                        layer_type = 'LayerNorm'
                    else:
                        layer_type = 'Dense'
                    name = copy.deepcopy(my_n_split[:-1])
                    name_or = copy.deepcopy(my_n_split_or[:-1])
                else:
                    name = copy.deepcopy(my_n_split)
                    name_or = copy.deepcopy(my_n_split_or)

                my_layer, my_layer_name = self._get_BERT_layer(name)

                if name_or not in filled:
                    if Verbose:
                        print("Now filling %s" % name_or[1:])

                    my_vals = getFromDict(weights_dict_full, name_or[1:])

                    if layer_type == 'LayerNorm':
                        my_vals_list = [my_vals['beta'], my_vals['gamma']]
                    elif layer_type == 'Dense':
                        my_vals_list = [my_vals['bias'], my_vals['kernel']]
                    else:
                        my_vals_list = [my_vals]

                    if len(my_vals_list) < 2:
                        input_shape = my_vals_list[0].shape
                    else:
                        if layer_type == 'LayerNorm':
                            input_shape = my_vals['beta'].shape
                        elif layer_type == 'Dense':
                            input_shape = my_vals['bias'].shape[::-1]

                    if Verbose:
                        print("Input shape:")
                        print(input_shape)

                    my_layer.set_weights(my_vals_list)
                    my_layer.trainable = False

                    filled.append(name_or)
                else:
                    if Verbose: print("Skipping %s" % my_n_split_or[1:])
        print("------  Filled. \n ")





