"""Extract pre-computed feature vectors from a Keras BERT model."""


from modeling import PreTrainedBertModel, BertConfig, BERTModel
import tokenization as tokenization
import re
import keras.backend as K
import numpy as np
import argparse
import collections
import json
import os
from data_utils import *



# ++++++++++++++++++++++++++++++++++




class BertFeatureExtractor(BertTokenizer):

    def __init__(self, model_dir=None, input_file=None, input_list=[], max_seq_length=128):

        super().__init__(model_dir=model_dir, input_file=input_file, \
                         input_list=input_list, max_seq_length=max_seq_length)
        self.model_dir=model_dir

    def feature_extactor(self, dummy=False):
        if not dummy:
            model_pre_trained = PreTrainedBertModel(self.model_dir, Verbose=True)
        else:
            print("Using randomly initialized model...")
            model_pre_trained=BERTModel(config=BertConfig(), Verbose = True, trainable = False)
            model_pre_trained.build(input_shape=(self.batch_size,self.max_seq_length))

        print("Computing embeddings...")
        self.bert_embeddings = model_pre_trained(K.variable(self.all_input_ids))
        print("Evaluating...")
        self.all_encoder_layers = np.array([K.eval(emb) for emb in self.bert_embeddings[:-1]])
        print("Output BERT shape: ", self.all_encoder_layers.shape)




# ++++++++++++++++++++++++++++++++++ MAIN FUNCTION +++++++++++++++++++++++++++++++++++++++


def main():


    # 1. Read input args

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=False,
                        help="Directory where the pre-trained weights are stored")


    parser.add_argument("--layers", default="-1", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")

    args = parser.parse_args()

    layer_indexes = [int(x) for x in args.layers.split(",")]

    # 2. Instantiate class BertFeatureExtractor to prepare tokens
    my_emb = BertFeatureExtractor(model_dir=args.model_dir, input_file=args.input_file,\
                                  max_seq_length=args.max_seq_length)

    # Fill bert model with pre-trained weights and compute embeddings
    my_emb.feature_extactor( dummy=False)


    # 3. save results to file.
    print()
    print("Saving results to ", args.output_file)

    with open(args.output_file, "w", encoding='utf-8') as writer:

        for example_index in range(len(my_emb.all_input_ids)):

            feature = my_emb.features[example_index]
            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = my_emb.all_encoder_layers[int(layer_index)]
                    layer_output = layer_output[example_index] # example_index is number of feature
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                            round(x.item(), 6) for x in layer_output[i] # i is token
                        ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            output_json["features"] = all_out_features
            writer.write(json.dumps(output_json) + "\n")

    print("Saved.")



if __name__=='__main__':
    main()


