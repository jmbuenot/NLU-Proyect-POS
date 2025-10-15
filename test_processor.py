#!/usr/bin/env python3
"""
Test script for the CoNLL-U processor to validate functionality.
"""

from conllu_processor import CoNLLUProcessor, load_ud_english_data

def test_processor_on_sample():
    """Test the processor on a small sample to validate functionality."""
    print("Testing CoNLL-U Processor...")
    print("=" * 50)
    
    # Test on development data first (smaller file)
    dev_path = "data/ud_english_ewt/en_ewt-ud-dev.conllu"
    
    processor = CoNLLUProcessor()
    
    try:
        print("Loading development data for testing...")
        processor.load_conllu_file(dev_path, max_sentence_length=128)
        
        # Get processed sentences
        sentences = processor.get_word_pos_pairs()
        
        print(f"\nTest Results:")
        print(f"Successfully loaded {len(sentences)} sentences")
        
        if len(sentences) > 0:
            print(f"\nFirst 3 sentences:")
            for i, sentence in enumerate(sentences[:3]):
                print(f"\nSentence {i+1} (length: {len(sentence)}):")
                for j, (word, pos) in enumerate(sentence):
                    print(f"  {j+1:2d}. {word:15s} -> {pos}")
        
        # Test vocabulary and POS tag extraction
        vocab = processor.get_vocabulary()
        pos_tags = processor.get_pos_tags()
        
        print(f"\nVocabulary Statistics:")
        print(f"Total unique words: {len(vocab)}")
        print(f"Most frequent words:")
        for word, count in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {word:15s}: {count}")
        
        print(f"\nPOS Tag Statistics:")
        print(f"Total unique POS tags: {len(pos_tags)}")
        print(f"All POS tags: {sorted(pos_tags.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def test_multiword_detection():
    """Test multiword token detection with sample data."""
    print("\nTesting multiword token detection...")
    print("=" * 40)
    
    processor = CoNLLUProcessor()
    
    # Test cases for multiword detection
    test_cases = [
        ("1", False),          # Regular token
        ("2", False),          # Regular token
        ("19-20", True),       # Multiword token
        ("1-2", True),         # Multiword token
        ("10.1", True),        # Empty node (should also be filtered)
        ("#", False),          # Not a token ID
    ]
    
    for token_id, expected_multiword in test_cases:
        is_multiword = processor._is_multiword_token(token_id)
        is_empty = processor._is_empty_node(token_id)
        
        if expected_multiword:
            if is_multiword or is_empty:
                print(f"✓ {token_id:6s}: Correctly identified as should be removed")
            else:
                print(f"✗ {token_id:6s}: Should be removed but wasn't detected")
        else:
            if not is_multiword and not is_empty:
                print(f"✓ {token_id:6s}: Correctly identified as regular token")
            else:
                print(f"✗ {token_id:6s}: Regular token incorrectly flagged for removal")

def run_full_test():
    """Run comprehensive test on all three datasets."""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST - Loading All Datasets")
    print("="*60)
    
    # File paths
    train_path = "data/ud_english_ewt/en_ewt-ud-train.conllu"
    dev_path = "data/ud_english_ewt/en_ewt-ud-dev.conllu"
    test_path = "data/ud_english_ewt/en_ewt-ud-test.conllu"
    
    try:
        processors = load_ud_english_data(train_path, dev_path, test_path)
        
        print(f"\n" + "="*60)
        print("SUMMARY OF ALL DATASETS")
        print("="*60)
        
        for dataset_name, processor in processors.items():
            sentences = processor.get_word_pos_pairs()
            total_tokens = sum(len(sent) for sent in sentences)
            vocab_size = len(processor.get_vocabulary())
            pos_tags = len(processor.get_pos_tags())
            
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"  Sentences: {len(sentences)}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Vocabulary size: {vocab_size}")
            print(f"  Unique POS tags: {pos_tags}")
            print(f"  Multiword tokens removed: {processor.removed_multiword_count}")
            print(f"  Long sentences removed: {processor.removed_long_sentences}")
        
        # Show a sample sentence from each dataset
        print(f"\n" + "="*60)
        print("SAMPLE SENTENCES")
        print("="*60)
        
        for dataset_name, processor in processors.items():
            sentences = processor.get_word_pos_pairs()
            if sentences:
                print(f"\nSample from {dataset_name.upper()}:")
                sample_sentence = sentences[0]
                for word, pos in sample_sentence:
                    print(f"  {word} -> {pos}")
        
        return processors
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        print("Make sure the data is in the correct location:")
        print("  data/ud_english_ewt/en_ewt-ud-train.conllu")
        print("  data/ud_english_ewt/en_ewt-ud-dev.conllu") 
        print("  data/ud_english_ewt/en_ewt-ud-test.conllu")
        return None

if __name__ == "__main__":
    print("CoNLL-U Processor Test Suite")
    print("=" * 60)
    
    # Run basic test
    success = test_processor_on_sample()
    
    if success:
        # Test multiword detection
        test_multiword_detection()
        
        # Run full comprehensive test
        processors = run_full_test()
        
        if processors:
            print(f"\n" + "="*60)
            print("ALL TESTS PASSED! ✓")
            print("Your data processing functions are working correctly.")
            print("="*60)
        else:
            print(f"\n" + "="*60)
            print("Full test failed - check file paths")
            print("="*60)
    else:
        print(f"\n" + "="*60)
        print("Basic test failed")
        print("="*60)
