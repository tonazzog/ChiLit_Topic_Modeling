import nltk
from nltk.corpus.reader import PlaintextCorpusReader
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import os


class CorpusAnalyzer:
    def __init__(self, corpus):
        """
        Initialize the analyzer with an NLTK PlainTextCorpusReader corpus.
        
        Args:
            corpus: An NLTK PlainTextCorpusReader object
        """
        self.corpus = corpus
        
        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def analyze_document(self, fileid):
        """
        Analyze a single document in the corpus.
        
        Args:
            fileid: The file identifier for the document
            
        Returns:
            dict: Dictionary containing various statistics for the document
        """
        # Get raw text
        raw_text = self.corpus.raw(fileid)
        
        # Get words and sentences
        words = self.corpus.words(fileid)
        sentences = self.corpus.sents(fileid)
        
        # Basic counts
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(raw_text)
        char_count_no_spaces = len(raw_text.replace(' ', ''))
        
        # Filter out punctuation and convert to lowercase
        clean_words = [word.lower() for word in words if word.isalpha()]
        
        # Unique words
        unique_words = set(clean_words)
        unique_word_count = len(unique_words)
        
        # Content words (excluding stopwords)
        content_words = [word for word in clean_words if word not in self.stopwords]
        content_word_count = len(content_words)
        unique_content_words = set(content_words)
        unique_content_word_count = len(unique_content_words)
        
        # Calculate metrics
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # Lexical density (ratio of content words to total words)
        lexical_density = content_word_count / word_count if word_count > 0 else 0
        
        # Type-Token Ratio (TTR) - measure of lexical diversity
        ttr = unique_word_count / word_count if word_count > 0 else 0
        content_ttr = unique_content_word_count / content_word_count if content_word_count > 0 else 0
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in clean_words]) if clean_words else 0
        
        # Most common words
        word_freq = Counter(clean_words)
        most_common_words = word_freq.most_common(10)
        
        # Most common content words
        content_word_freq = Counter(content_words)
        most_common_content_words = content_word_freq.most_common(10)
        
        return {
            'file_id': fileid,
            'total_characters': char_count,
            'characters_no_spaces': char_count_no_spaces,
            'total_words': word_count,
            'clean_words': len(clean_words),
            'unique_words': unique_word_count,
            'content_words': content_word_count,
            'unique_content_words': unique_content_word_count,
            'sentences': sentence_count,
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'lexical_density': round(lexical_density, 3),
            'type_token_ratio': round(ttr, 3),
            'content_ttr': round(content_ttr, 3),
            'most_common_words': most_common_words,
            'most_common_content_words': most_common_content_words
        }
    
    def analyze_corpus(self):
        """
        Analyze all documents in the corpus.
        
        Returns:
            tuple: (individual_stats, corpus_summary)
        """
        individual_stats = []
        
        # Analyze each document
        for fileid in self.corpus.fileids():
            doc_stats = self.analyze_document(fileid)
            individual_stats.append(doc_stats)
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame(individual_stats)
        
        # Calculate corpus-wide statistics
        corpus_summary = {
            'total_documents': len(individual_stats),
            'total_words_corpus': df['total_words'].sum(),
            'total_sentences_corpus': df['sentences'].sum(),
            'avg_words_per_doc': round(df['total_words'].mean(), 2),
            'std_words_per_doc': round(df['total_words'].std(), 2),
            'min_words_per_doc': df['total_words'].min(),
            'max_words_per_doc': df['total_words'].max(),
            'avg_sentences_per_doc': round(df['sentences'].mean(), 2),
            'avg_lexical_density': round(df['lexical_density'].mean(), 3),
            'avg_ttr': round(df['type_token_ratio'].mean(), 3),
            'avg_word_length_corpus': round(df['avg_word_length'].mean(), 2)
        }
        
        return individual_stats, corpus_summary, df
    
    def print_summary(self, individual_stats, corpus_summary):
        """
        Print a formatted summary of the corpus analysis.
        """
        print("=" * 60)
        print("CORPUS ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nCorpus Overview:")
        print(f"  Total documents: {corpus_summary['total_documents']}")
        print(f"  Total words: {corpus_summary['total_words_corpus']:,}")
        print(f"  Total sentences: {corpus_summary['total_sentences_corpus']:,}")
        
        print(f"\nDocument Statistics:")
        print(f"  Average words per document: {corpus_summary['avg_words_per_doc']} (±{corpus_summary['std_words_per_doc']})")
        print(f"  Range: {corpus_summary['min_words_per_doc']} - {corpus_summary['max_words_per_doc']} words")
        print(f"  Average sentences per document: {corpus_summary['avg_sentences_per_doc']}")
        
        print(f"\nLinguistic Metrics:")
        print(f"  Average lexical density: {corpus_summary['avg_lexical_density']}")
        print(f"  Average type-token ratio: {corpus_summary['avg_ttr']}")
        print(f"  Average word length: {corpus_summary['avg_word_length_corpus']} characters")
        
        print(f"\n" + "=" * 60)
        print("INDIVIDUAL DOCUMENT DETAILS")
        print("=" * 60)
        
        for stats in individual_stats:
            print(f"\nDocument: {stats['file_id']}")
            print(f"  Words: {stats['total_words']:,} | Sentences: {stats['sentences']}")
            print(f"  Avg words/sentence: {stats['avg_words_per_sentence']}")
            print(f"  Lexical density: {stats['lexical_density']}")
            print(f"  Type-token ratio: {stats['type_token_ratio']}")
            print(f"  Top words: {', '.join([f'{word}({count})' for word, count in stats['most_common_words'][:5]])}")
            print(f"  Top content words: {', '.join([f'{word}({count})' for word, count in stats['most_common_content_words'][:5]])}")

    def analyze_document_lengths(self, individual_stats):
      """Analyze the distribution of document lengths."""
      
      doc_lengths = [file_id['total_words'] for file_id in individual_stats]
      
      stats = {
          'min': min(doc_lengths),
          'max': max(doc_lengths),
          'mean': np.mean(doc_lengths),
          'std': np.std(doc_lengths),
          'median': np.median(doc_lengths),
          'q25': np.percentile(doc_lengths, 25),
          'q75': np.percentile(doc_lengths, 75)
      }
      
      return doc_lengths, stats
    
    def plot_length_distribution(self, doc_lengths):
      """Plot the distribution of document lengths."""
      plt.figure(figsize=(12, 4))
      
      plt.subplot(1, 2, 1)
      plt.hist(doc_lengths, bins=30, alpha=0.7, edgecolor='black')
      plt.xlabel('Document Length (words)')
      plt.ylabel('Frequency')
      plt.title('Distribution of Document Lengths')
      
      plt.subplot(1, 2, 2)
      plt.boxplot(doc_lengths)
      plt.ylabel('Document Length (words)')
      plt.title('Document Length Box Plot')
      
      plt.tight_layout()
      plt.show()      