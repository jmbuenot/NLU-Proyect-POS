#!/usr/bin/env python3
"""
Usage Example: CoNLL-U Processor for PoS Tagging
================================================

This example demonstrates how to use the CoNLL-U processor for your PoS tagging assignment.
"""

from conllu_processor import CoNLLUProcessor, load_ud_english_data

def example_basic_usage():
    """Example of basic usage - loading a single file."""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # Create processor instance
    processor = CoNLLUProcessor()
    
    # Load development data (smaller file for quick testing)
    processor.load_conllu_file("data/ud_english_ewt/en_ewt-ud-dev.conllu", max_sentence_length=128)
    
    # Get processed sentences
    sentences = processor.get_word_pos_pairs()
    
    print(f"Loaded {len(sentences)} sentences")
    
    # Show first sentence
    if sentences:
        print("\nFirst sentence:")
        for word, pos_tag in sentences[0]:
            print(f"  {word:15s} -> {pos_tag}")
    
    return sentences

def example_full_dataset():
    """Example of loading all three dataset files."""
    print("\n=== FULL DATASET LOADING ===")
    
    # Define file paths
    train_path = "data/ud_english_ewt/en_ewt-ud-train.conllu"
    dev_path = "data/ud_english_ewt/en_ewt-ud-dev.conllu"
    test_path = "data/ud_english_ewt/en_ewt-ud-test.conllu"
    
    # Load all datasets
    processors = load_ud_english_data(train_path, dev_path, test_path, max_sentence_length=128)
    
    # Access individual datasets
    train_sentences = processors['train'].get_word_pos_pairs()
    dev_sentences = processors['dev'].get_word_pos_pairs()
    test_sentences = processors['test'].get_word_pos_pairs()
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_sentences)} sentences")
    print(f"  Development: {len(dev_sentences)} sentences")
    print(f"  Test:       {len(test_sentences)} sentences")
    
    return processors

def example_data_analysis():
    """Example of analyzing the processed data."""
    print("\n=== DATA ANALYSIS EXAMPLE ===")
    
    # Load just development data for analysis
    processor = CoNLLUProcessor()
    processor.load_conllu_file("data/ud_english_ewt/en_ewt-ud-dev.conllu", max_sentence_length=128)
    
    # Get vocabulary and POS statistics
    vocabulary = processor.get_vocabulary()
    pos_tags = processor.get_pos_tags()
    sentence_lengths = processor.get_sentence_length_distribution()
    
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Number of POS tags: {len(pos_tags)}")
    
    print(f"\nPOS tag distribution:")
    for tag, count in sorted(pos_tags.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag:8s}: {count:5d}")
    
    print(f"\nSentence length distribution (first 10):")
    for length, count in sorted(sentence_lengths.items())[:10]:
        print(f"  Length {length:2d}: {count:3d} sentences")

def prepare_data_for_neural_model():
    """Example of preparing data for neural PoS tagging model."""
    print("\n=== PREPARING DATA FOR NEURAL MODEL ===")
    
    # Load all datasets
    processors = load_ud_english_data(
        "data/ud_english_ewt/en_ewt-ud-train.conllu",
        "data/ud_english_ewt/en_ewt-ud-dev.conllu", 
        "data/ud_english_ewt/en_ewt-ud-test.conllu",
        max_sentence_length=128
    )
    
    # Extract data for neural model
    train_data = processors['train'].get_word_pos_pairs()
    dev_data = processors['dev'].get_word_pos_pairs()
    test_data = processors['test'].get_word_pos_pairs()
    
    # Example: Create word and POS vocabularies for model
    all_words = set()
    all_pos_tags = set()
    
    for dataset in [train_data, dev_data, test_data]:
        for sentence in dataset:
            for word, pos_tag in sentence:
                all_words.add(word)
                all_pos_tags.add(pos_tag)
    
    print(f"Total unique words across all datasets: {len(all_words)}")
    print(f"Total unique POS tags: {len(all_pos_tags)}")
    print(f"POS tags: {sorted(list(all_pos_tags))}")
    
    # Example: Show data format ready for neural model
    print(f"\nExample data format for neural model:")
    print(f"Each sentence is a list of (word, pos_tag) tuples:")
    print(f"  Sentence 1: {train_data[0]}")
    
    return {
        'train': train_data,
        'dev': dev_data, 
        'test': test_data,
        'word_vocab': sorted(list(all_words)),
        'pos_vocab': sorted(list(all_pos_tags))
    }

if __name__ == "__main__":
    print("CoNLL-U Processor Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        # Basic usage
        sentences = example_basic_usage()
        
        # Full dataset loading
        processors = example_full_dataset()
        
        # Data analysis
        example_data_analysis()
        
        # Prepare for neural model
        model_data = prepare_data_for_neural_model()
        
        print(f"\n" + "="*50)
        print("✓ All examples completed successfully!")
        print("✓ Your data is ready for neural PoS tagging model training.")
        print("="*50)
        
    except Exception as e:
        print(f"Error in examples: {e}")
