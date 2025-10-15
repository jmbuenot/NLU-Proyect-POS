# Practical Assignment 1 - Part-of-speech (PoS) Tagging

In this assignment, you will implement a neural PoS sequence labeling model to label the words of a given input sentence according to their morphological information. Table 1 shows a simple example:
| <!-- -->      | <!-- -->      | <!-- -->      | <!-- -->      | <!-- -->      | <!-- -->      | <!-- -->      | <!-- -->      |
|:-------------:|:---------------:|:-------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| **Inputs** | Google | is | a     | nice | search      | engine        | .            |
| **Outputs**| PROPN | AUX | DET| ADJ | NOUN | NOUN | PUNCT |

Table 1. Example of an input sentence and the ground-truth output, extracted from the test set of the UD_English_EWT dataset.

Specifically, your task involves the implementation of a neural model utilizing long short-term memory  networks (LSTMs). We provide a suggested default architecture. The fundamental model should take word-level embeddings as its input. Subsequently, this representation will be passed through a word￾level LSTM layer, followed by a dense layer that assigns a label to each token using a softmax activation  function.
Figure 1 illustrates the high-level structure of the network. You are granted the flexibility to adjust the  network's dimensions, experiment with hyperparameters for enhanced performance, and extend the  foundational model.
![Figure 1: High-level architecture of the PoS tagging model](image.png)
Figure 1: High-level architecture of the PoS tagging model

The functionalities that must be supported are:
1. Train a model and ensure it converges successfully.
2. Evaluate the trained model and provide tagging accuracy metrics for both the  validation/development and test sets specified below.
3. Develop a function to compute part-of-speech tags for newly input sentences by the user that have not been seen before.

Specifically, for this assignment you must train and evaluate your model on the English_EWT dataset from the https://universaldependencies.org/ (UD) collection, using the corresponding training,  development and test files, represented in CoNLL-U format. Here, we will use and learn to predict the  morphological information stored in the UPOS column. Furthermore, we request that you train and  evaluate the model on two other language datasets from UD of your choosing. To simplify the process,  sentences longer than 128 words can be preprocessed and removed during the training and evaluation  of your models. 

Preprocessing the treebank: Ignore sentences longer than 128 from the training, development and test sets. This means an input to the model will have the shape (max_sentence_length, embedding_size).
Recommendations:
1. Use the Tokenizer or TextVectorizer layer approaches.
2. Implement your models using the functional API.
3. Make sure and extra ID is used for unknown tokens (read the docs for the Tokenizer or the TextVectorizer layer)

Main steps to train and evaluate the word-level model:
1. Transform the input sentences (strings) to a list of numerical IDs.
2. Pad the input sentences, so they all have the same length (required for Keras models).
3. Convert the output labels to IDs as well, so we can train the model with the fit function.



### Submission
The assignment should comprise a concise user manual detailing the steps to execute the code for training, evaluation, and label generation. Additionally, it should feature a brief discussion covering  implementation choices, potential variations across the explored models, and an analysis of performance across the assessed datasets. 

### Organize your code
Create folder reports and put requiered manuals and reports there. 
- Include a user manual explaining how to run.
- Define classes to create reusable pieces of code. 
- Your main notebook should make easy to relaunch all the relevant tasks: training, 
evaluation and generation of labels, etc. 
- Consider saving and loading your good trained models. 
- Include a brief discussion of the implementation decisions, differences across the 
evaluated models that you might have explored, as well as an analysis of the performance across the evaluated datasets: In a separate PDF not exceeding 3 pages and written using Calibri font style and a size of minimum 11pt. Put your names on it. 
