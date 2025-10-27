import pyconll
import tensorflow as tf
import keras
import numpy as np
import pathlib

class ReadUDTreebank(object):
    def read(self, link):
        path_to_downloaded_file = tf.keras.utils.get_file(
             origin = link,
             extract = True,
             )
        corpus = pyconll.load_from_file(path_to_downloaded_file)
        return corpus

    def remove_multi_empty(self, corpus):
        inputs = []
        targets = []
        for sentence in corpus:
          tokens = []
          pos = []
          for token in sentence:
            if sentence.__len__()<128:
              if not (token.is_multiword() or token.is_empty_node()):
                tokens = np.append(tokens, token.form)
                pos = np.append(pos, token.upos)
          inputs.append(" ".join(tokens))
          targets.append(pos)

        inputs = np.array(inputs, dtype= object)
        targets = np.array(targets, dtype = object)

        return inputs, targets