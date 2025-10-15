#!/usr/bin/env python3
"""
CoNLL-U Data Processor for PoS Tagging
======================================

This module provides functions to process UD English treebank data in CoNLL-U format.
It handles multiword token removal, sentence length filtering, and data extraction
for neural PoS tagging models.

Author: Student Implementation for NLU Lab 1
"""

import re
from typing import List, Tuple, Dict, Optional
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoNLLUProcessor:
    """
    Processor class for handling CoNLL-U format data from Universal Dependencies.
    
    Features:
    - Parse CoNLL-U files
    - Remove multiword tokens and empty nodes
    - Filter sentences by length
    - Extract clean (word, UPOS) pairs for training
    - Provide detailed statistics
    """
    
    def __init__(self):
        """Initialize the processor with empty data structures."""
        self.sentences = []  # List of sentences, each sentence is list of (word, upos) tuples
        self.raw_sentences = []  # Keep raw data for debugging
        self.removed_multiword_count = 0
        self.removed_empty_nodes = 0
        self.removed_long_sentences = 0
        self.total_sentences_processed = 0
        self.max_sentence_length = 128
        
    def load_conllu_file(self, filepath: str, max_sentence_length: int = 128) -> None:
        """
        Load and process a CoNLL-U file.
        
        Args:
            filepath (str): Path to the .conllu file
            max_sentence_length (int): Maximum allowed sentence length (default: 128)
        """
        self.max_sentence_length = max_sentence_length
        logger.info(f"Loading CoNLL-U file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                self._parse_conllu_content(file.readlines())
            
            logger.info(f"Successfully loaded {filepath}")
            self.print_statistics()
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {str(e)}")
            raise
    
    def _parse_conllu_content(self, lines: List[str]) -> None:
        """
        Parse the content of a CoNLL-U file line by line.
        
        Args:
            lines (List[str]): All lines from the file
        """
        current_sentence = []
        current_raw_sentence = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines (sentence boundaries)
            if not line:
                if current_sentence:
                    self._process_completed_sentence(current_sentence, current_raw_sentence)
                    current_sentence = []
                    current_raw_sentence = []
                continue
            
            # Skip comment lines (metadata)
            if line.startswith('#'):
                current_raw_sentence.append(line)
                continue
            
            # Parse token lines
            try:
                token_data = self._parse_token_line(line)
                if token_data:
                    current_sentence.append(token_data)
                    current_raw_sentence.append(line)
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {line}")
                logger.warning(f"Error details: {str(e)}")
                continue
        
        # Process the last sentence if file doesn't end with empty line
        if current_sentence:
            self._process_completed_sentence(current_sentence, current_raw_sentence)
    
    def _parse_token_line(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Parse a single token line from CoNLL-U format.
        
        Args:
            line (str): A line from the .conllu file
            
        Returns:
            Optional[Tuple[str, str]]: (word, upos_tag) tuple or None if line should be skipped
        """
        parts = line.split('\t')
        
        # CoNLL-U format should have exactly 10 columns
        if len(parts) != 10:
            logger.warning(f"Line doesn't have 10 columns: {line}")
            return None
        
        token_id = parts[0]
        word_form = parts[1]
        upos_tag = parts[3]
        
        # Skip multiword tokens (e.g., "1-2", "19-20")
        if self._is_multiword_token(token_id):
            self.removed_multiword_count += 1
            return None
        
        # Skip empty nodes (e.g., "10.1")
        if self._is_empty_node(token_id):
            self.removed_empty_nodes += 1
            return None
        
        # Validate that we have valid word and UPOS tag
        if not word_form or not upos_tag or upos_tag == '_':
            logger.warning(f"Invalid word or UPOS tag in line: {line}")
            return None
        
        return (word_form, upos_tag)
    
    def _is_multiword_token(self, token_id: str) -> bool:
        """
        Check if a token ID represents a multiword token (contains hyphen).
        
        Args:
            token_id (str): The token ID from column 1
            
        Returns:
            bool: True if this is a multiword token
        """
        return '-' in token_id and not token_id.startswith('#')
    
    def _is_empty_node(self, token_id: str) -> bool:
        """
        Check if a token ID represents an empty node (contains dot).
        
        Args:
            token_id (str): The token ID from column 1
            
        Returns:
            bool: True if this is an empty node
        """
        return '.' in token_id
    
    def _process_completed_sentence(self, sentence: List[Tuple[str, str]], 
                                   raw_sentence: List[str]) -> None:
        """
        Process a completed sentence and decide whether to keep it.
        
        Args:
            sentence (List[Tuple[str, str]]): List of (word, upos) pairs
            raw_sentence (List[str]): Original lines for debugging
        """
        self.total_sentences_processed += 1
        
        # Filter out sentences that are too long
        if len(sentence) > self.max_sentence_length:
            self.removed_long_sentences += 1
            logger.debug(f"Removed long sentence with {len(sentence)} tokens")
            return
        
        # Keep sentences with at least 1 token
        if len(sentence) > 0:
            self.sentences.append(sentence)
            self.raw_sentences.append(raw_sentence)
    
    def get_word_pos_pairs(self) -> List[List[Tuple[str, str]]]:
        """
        Get the processed sentences as lists of (word, UPOS) pairs.
        
        Returns:
            List[List[Tuple[str, str]]]: List of sentences, each containing (word, upos) tuples
        """
        return self.sentences
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get vocabulary statistics from the processed data.
        
        Returns:
            Dict[str, int]: Word frequency dictionary
        """
        word_counts = Counter()
        for sentence in self.sentences:
            for word, _ in sentence:
                word_counts[word] += 1
        return dict(word_counts)
    
    def get_pos_tags(self) -> Dict[str, int]:
        """
        Get POS tag statistics from the processed data.
        
        Returns:
            Dict[str, int]: POS tag frequency dictionary
        """
        pos_counts = Counter()
        for sentence in self.sentences:
            for _, pos in sentence:
                pos_counts[pos] += 1
        return dict(pos_counts)
    
    def get_sentence_length_distribution(self) -> Dict[int, int]:
        """
        Get distribution of sentence lengths.
        
        Returns:
            Dict[int, int]: Dictionary mapping sentence length to count
        """
        length_counts = Counter()
        for sentence in self.sentences:
            length_counts[len(sentence)] += 1
        return dict(length_counts)
    
    def print_statistics(self) -> None:
        """Print detailed statistics about the processed data."""
        print(f"\n{'='*60}")
        print("CoNLL-U Processing Statistics")
        print(f"{'='*60}")
        print(f"Total sentences processed: {self.total_sentences_processed}")
        print(f"Sentences kept: {len(self.sentences)}")
        print(f"Sentences removed (length > {self.max_sentence_length}): {self.removed_long_sentences}")
        print(f"Multiword tokens removed: {self.removed_multiword_count}")
        print(f"Empty nodes removed: {self.removed_empty_nodes}")
        
        if self.sentences:
            total_tokens = sum(len(sent) for sent in self.sentences)
            avg_length = total_tokens / len(self.sentences)
            max_length = max(len(sent) for sent in self.sentences)
            min_length = min(len(sent) for sent in self.sentences)
            
            print(f"\nSentence Statistics:")
            print(f"Total tokens: {total_tokens}")
            print(f"Average sentence length: {avg_length:.2f}")
            print(f"Min sentence length: {min_length}")
            print(f"Max sentence length: {max_length}")
            
            # POS tag statistics
            pos_tags = self.get_pos_tags()
            print(f"\nUnique POS tags found: {len(pos_tags)}")
            print("Most frequent POS tags:")
            for tag, count in sorted(pos_tags.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {tag}: {count}")
            
            # Vocabulary statistics
            vocab = self.get_vocabulary()
            print(f"\nVocabulary size: {len(vocab)}")
            print(f"{'='*60}")


def load_ud_english_data(train_path: str, dev_path: str, test_path: str, 
                        max_sentence_length: int = 128) -> Dict[str, CoNLLUProcessor]:
    """
    Load all three UD English treebank files.
    
    Args:
        train_path (str): Path to training .conllu file
        dev_path (str): Path to development .conllu file
        test_path (str): Path to test .conllu file
        max_sentence_length (int): Maximum sentence length to keep
        
    Returns:
        Dict[str, CoNLLUProcessor]: Dictionary with 'train', 'dev', 'test' processors
    """
    processors = {}
    
    # Load training data
    print("Loading training data...")
    train_processor = CoNLLUProcessor()
    train_processor.load_conllu_file(train_path, max_sentence_length)
    processors['train'] = train_processor
    
    # Load development data
    print("\nLoading development data...")
    dev_processor = CoNLLUProcessor()
    dev_processor.load_conllu_file(dev_path, max_sentence_length)
    processors['dev'] = dev_processor
    
    # Load test data
    print("\nLoading test data...")
    test_processor = CoNLLUProcessor()
    test_processor.load_conllu_file(test_path, max_sentence_length)
    processors['test'] = test_processor
    
    return processors


def demonstrate_usage():
    """Demonstrate how to use the CoNLLU processor."""
    print("CoNLL-U Processor Demonstration")
    print("=" * 50)
    
    # File paths
    train_path = "data/ud_english_ewt/en_ewt-ud-train.conllu"
    dev_path = "data/ud_english_ewt/en_ewt-ud-dev.conllu"
    test_path = "data/ud_english_ewt/en_ewt-ud-test.conllu"
    
    try:
        # Load all data
        processors = load_ud_english_data(train_path, dev_path, test_path)
        
        # Show example sentences
        print("\nExample sentences from training data:")
        train_sentences = processors['train'].get_word_pos_pairs()
        for i, sentence in enumerate(train_sentences[:3]):
            print(f"\nSentence {i+1}:")
            for word, pos in sentence:
                print(f"  {word} -> {pos}")
        
        print("\n" + "="*50)
        print("Data loading completed successfully!")
        
        return processors
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the data files are in the correct location.")
        return None


if __name__ == "__main__":
    # Run demonstration
    processors = demonstrate_usage()
