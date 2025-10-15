#!/usr/bin/env python3
"""
Tokenization and Mapping to IDs for PoS Tagging
================================================
This script extends the existing CoNLL-U data processor (conllu_processor.py)
by implementing tokenization, numerical encoding, and padding operations
required for training a neural sequence labeling model (e.g., LSTM PoS tagger).

Author: Student Implementation for NLU Lab 1
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from conllu_processor import load_ud_english_data

# ------------------------------------------------------------
# 1. Load already processed UD English data
# ------------------------------------------------------------
print("Loading preprocessed UD English EWT datasets...")
train_path = "data/ud_english_ewt/en_ewt-ud-train.conllu"
dev_path = "data/ud_english_ewt/en_ewt-ud-dev.conllu"
test_path = "data/ud_english_ewt/en_ewt-ud-test.conllu"

datasets = load_ud_english_data(train_path, dev_path, test_path)
train_data = datasets["train"].get_word_pos_pairs()
dev_data = datasets["dev"].get_word_pos_pairs()
test_data = datasets["test"].get_word_pos_pairs()

print(f"Loaded {len(train_data)} training sentences, "
      f"{len(dev_data)} dev sentences, {len(test_data)} test sentences.\n")

# ------------------------------------------------------------
# 2. Extract words and tags from sentences
# ------------------------------------------------------------
def split_words_tags(sentences):
    """Split list of (word, tag) pairs into two separate lists."""
    X = [[w for (w, t) in sent] for sent in sentences]
    y = [[t for (w, t) in sent] for sent in sentences]
    return X, y

X_train, y_train = split_words_tags(train_data)
X_dev, y_dev = split_words_tags(dev_data)
X_test, y_test = split_words_tags(test_data)

# ------------------------------------------------------------
# 3. Tokenizer for words
# ------------------------------------------------------------
print("Fitting Keras Tokenizer on training data (word-level)...")

word_tokenizer = Tokenizer(oov_token="<OOV>")
word_tokenizer.fit_on_texts(X_train)

word_index = word_tokenizer.word_index
vocab_size = len(word_index) + 1  # +1 for padding index 0

print(f"Vocabulary size: {vocab_size}")

# ------------------------------------------------------------
# 4. Encode POS tags manually (label mapping)
# ------------------------------------------------------------
print("Creating POS tag mapping...")

unique_tags = sorted(set(tag for sent in y_train for tag in sent))
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}
num_tags = len(tag2id)


print(f"Number of unique POS tags: {num_tags}")
print(f"Tags: {unique_tags}\n")

# ------------------------------------------------------------
# 5. Convert sentences and tags to integer sequences
# ------------------------------------------------------------
def encode_sentences_and_tags(X, y, word_tokenizer, tag2id):
    """Convert tokens and tags to integer sequences."""
    X_encoded = word_tokenizer.texts_to_sequences(X)
    y_encoded = [[tag2id[tag] for tag in tags] for tags in y]
    return X_encoded, y_encoded

X_train_ids, y_train_ids = encode_sentences_and_tags(X_train, y_train, word_tokenizer, tag2id)
X_dev_ids, y_dev_ids = encode_sentences_and_tags(X_dev, y_dev, word_tokenizer, tag2id)
X_test_ids, y_test_ids = encode_sentences_and_tags(X_test, y_test, word_tokenizer, tag2id)

# ------------------------------------------------------------
# 6. Pad sequences to uniform length
# ------------------------------------------------------------
MAX_LEN = 128

X_train_padded = pad_sequences(X_train_ids, maxlen=MAX_LEN, padding='post', truncating='post')
y_train_padded = pad_sequences(y_train_ids, maxlen=MAX_LEN, padding='post', truncating='post')

X_dev_padded = pad_sequences(X_dev_ids, maxlen=MAX_LEN, padding='post', truncating='post')
y_dev_padded = pad_sequences(y_dev_ids, maxlen=MAX_LEN, padding='post', truncating='post')

X_test_padded = pad_sequences(X_test_ids, maxlen=MAX_LEN, padding='post', truncating='post')
y_test_padded = pad_sequences(y_test_ids, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Padded sequences to length {MAX_LEN}")
print(f"Example encoded + padded sentence:")
print(X_train_padded[0][:20])
print(f"Example tag IDs:")
print(y_train_padded[0][:20], "\n")

# ------------------------------------------------------------
# 7. Validation checks
# ------------------------------------------------------------
assert X_train_padded.shape[0] == len(y_train_padded)
assert X_dev_padded.shape[0] == len(y_dev_padded)
assert X_test_padded.shape[0] == len(y_test_padded)

print("Validation passed: All input and tag sequences aligned correctly.\n")


# =============================================================================
# DEMO: Verifying the word-to-ID mapping for a single sentence
#
# This block demonstrates that our tokenization and padding process is fully
# reversible. It takes a single sentence from the training set, retrieves its
# padded integer representation, reverses the process, and verifies that the
# decoded result perfectly matches the original data.
# =============================================================================
print("\n" + "="*60)
print("DEMO: Mapping a Sentence to IDs and Back")
print("="*60)

# 1. Select a sample sentence from the training dataset.
#    We use a fixed index (e.g., 43) for reproducibility of this demonstration.
sample_idx = 43
original_words = X_train[sample_idx]
original_tags = y_train[sample_idx]
original_length = len(original_words)

# 2. Retrieve the corresponding padded ID sequences for both words and tags.
padded_word_ids = X_train_padded[sample_idx]
padded_tag_ids = y_train_padded[sample_idx]

# 3. Remove the padding to isolate the IDs of the original sentence.
word_ids_no_padding = padded_word_ids[:original_length]
tag_ids_no_padding = padded_tag_ids[:original_length]

# 4. Decode the integer IDs back into their original string representations.
#    The Keras Tokenizer provides an `index_word` attribute for reverse mapping of word IDs.
decoded_words = [word_tokenizer.index_word.get(i, "<UNK>") for i in word_ids_no_padding]
#    We use our previously created 'id2tag' dictionary for the PoS tag IDs.
decoded_tags = [id2tag.get(i, "<UNK>") for i in tag_ids_no_padding]

# 5. Display a side-by-side comparison to visually verify the mappings.
print(f"--- Analyzing Sentence #{sample_idx} ---")
print(f"Original Sentence: {' '.join(original_words)}\n")

print(f"{'WORD':<18} | {'WORD_ID':<10} | {'PoS TAG':<15} | {'POS_ID':<10}")
print("-" * 65)

for i in range(original_length):
    word = original_words[i]
    w_id = word_ids_no_padding[i]
    tag = original_tags[i]
    t_id = tag_ids_no_padding[i]
    print(f"{word:<18} | {w_id:<10} | {tag:<15} | {t_id:<10}")

# 6. Perform a final programmatic check to confirm the bidirectionality of the mapping.
print("\n--- Verification ---")
print(f"Original sentence matches decoded sentence: {original_words == decoded_words}")
print(f"Original tags match decoded tags: {original_tags == decoded_tags}")

if original_words == decoded_words and original_tags == decoded_tags:
    print("\nSuccess! The round-trip mapping is working correctly. ✅")
else:
    print("\nWarning! A discrepancy was found in the mapping process. ⚠️")

print("="*60)