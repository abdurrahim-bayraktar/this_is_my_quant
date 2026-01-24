"""
Sentiment extraction using FinBERT.

This module provides:
- FinBERT-based sentiment analysis
- Batch processing for efficiency
- Caching of extracted embeddings
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

from config import sentiment_config, CACHE_DIR

logger = logging.getLogger(__name__)


class SentimentScore(NamedTuple):
    """Container for sentiment analysis results."""
    label: str           # "positive", "negative", "neutral"
    score: float         # Probability of the predicted label
    positive: float      # Probability of positive
    negative: float      # Probability of negative
    neutral: float       # Probability of neutral


class SentimentExtractor:
    """
    FinBERT-based sentiment extractor for financial texts.
    
    FinBERT is pre-trained on financial corpora and fine-tuned for
    sentiment analysis. It outputs probabilities for:
    - Positive (bullish)
    - Negative (bearish)
    - Neutral
    
    Attributes:
        model: The FinBERT model.
        tokenizer: The FinBERT tokenizer.
        device: Device for inference (cuda/cpu).
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = None,
        batch_size: int = None,
    ):
        """
        Initialize the sentiment extractor.
        
        Args:
            model_name: HuggingFace model name. Default: ProsusAI/finbert.
            device: Device for inference. Default: cuda if available.
            max_length: Maximum sequence length. Default: 512.
            batch_size: Batch size for inference. Default: 16.
        """
        self.model_name = model_name or sentiment_config.model_name
        
        # Dynamic device detection (override config if CUDA is available now but wasn't slightly earlier)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = sentiment_config.device
            
        self.max_length = max_length or sentiment_config.max_length
        self.batch_size = batch_size or sentiment_config.batch_size
        
        logger.info(f"Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # FinBERT label mapping
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
        
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()
    def extract(self, text: str) -> SentimentScore:
        """
        Extract sentiment from a single text.
        
        Args:
            text: Input text (tweet, news headline, etc.).
            
        Returns:
            SentimentScore with label and probabilities.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        
        # Get prediction
        pred_idx = probs.argmax().item()
        
        return SentimentScore(
            label=self.label_map[pred_idx],
            score=probs[pred_idx].item(),
            positive=probs[0].item(),
            negative=probs[1].item(),
            neutral=probs[2].item(),
        )
    
    @torch.no_grad()
    def batch_extract(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[SentimentScore]:
        """
        Extract sentiment from multiple texts efficiently.
        
        Uses batching for GPU efficiency.
        
        Args:
            texts: List of input texts.
            show_progress: Whether to show progress bar.
            
        Returns:
            List of SentimentScore objects.
        """
        results = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting sentiment")
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            # Process each result
            for j, prob in enumerate(probs):
                pred_idx = prob.argmax().item()
                results.append(SentimentScore(
                    label=self.label_map[pred_idx],
                    score=prob[pred_idx].item(),
                    positive=prob[0].item(),
                    negative=prob[1].item(),
                    neutral=prob[2].item(),
                ))
        
        return results
    
    def extract_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        cache_key: str = None,
    ) -> pd.DataFrame:
        """
        Extract sentiment for all texts in a DataFrame.
        
        Adds columns: sentiment_label, sentiment_score, 
        sentiment_positive, sentiment_negative, sentiment_neutral.
        
        Args:
            df: Input DataFrame.
            text_col: Column containing text.
            cache_key: Optional key for caching results.
            
        Returns:
            DataFrame with sentiment columns added.
        """
        # Check cache
        if cache_key and sentiment_config.cache_embeddings:
            cache_path = CACHE_DIR / f"sentiment_{cache_key}.pkl"
            if cache_path.exists():
                logger.info(f"Loading cached sentiment from {cache_path}")
                cached = pd.read_pickle(cache_path)
                return df.join(cached)
        
        logger.info(f"Extracting sentiment for {len(df)} texts...")
        
        # Extract
        texts = df[text_col].fillna("").tolist()
        results = self.batch_extract(texts)
        
        # Add to DataFrame
        df = df.copy()
        df["sentiment_label"] = [r.label for r in results]
        df["sentiment_score"] = [r.score for r in results]
        df["sentiment_positive"] = [r.positive for r in results]
        df["sentiment_negative"] = [r.negative for r in results]
        df["sentiment_neutral"] = [r.neutral for r in results]
        
        # Compute a single sentiment value: positive - negative
        df["sentiment_value"] = df["sentiment_positive"] - df["sentiment_negative"]
        
        # Cache results
        if cache_key and sentiment_config.cache_embeddings:
            sentiment_cols = [
                "sentiment_label", "sentiment_score", "sentiment_positive",
                "sentiment_negative", "sentiment_neutral", "sentiment_value"
            ]
            df[sentiment_cols].to_pickle(cache_path)
            logger.info(f"Cached sentiment to {cache_path}")
        
        return df
    
    def get_embeddings(
        self,
        texts: List[str],
        layer: int = -1,
    ) -> np.ndarray:
        """
        Get hidden state embeddings for texts.
        
        Useful for more sophisticated downstream processing.
        
        Args:
            texts: List of input texts.
            layer: Which layer to extract (-1 = last layer).
            
        Returns:
            NumPy array of shape (n_texts, hidden_size).
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.bert(**inputs, output_hidden_states=True)
                # Use CLS token embedding from specified layer
                hidden_states = outputs.hidden_states[layer]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)


def demo():
    """Demonstrate sentiment extraction."""
    logging.basicConfig(level=logging.INFO)
    
    extractor = SentimentExtractor()
    
    # Test texts
    test_texts = [
        "Apple beats earnings expectations, stock surges 5%",
        "Tesla recalls 50,000 vehicles due to safety concerns",
        "Federal Reserve keeps interest rates unchanged",
        "Amazon announces massive layoffs amid economic uncertainty",
        "Microsoft partners with OpenAI for enterprise AI solutions",
    ]
    
    print("\n" + "="*60)
    print("Sentiment Analysis Results")
    print("="*60)
    
    for text in test_texts:
        result = extractor.extract(text)
        print(f"\nText: {text}")
        print(f"  Label: {result.label} ({result.score:.2%})")
        print(f"  Positive: {result.positive:.2%}")
        print(f"  Negative: {result.negative:.2%}")
        print(f"  Neutral: {result.neutral:.2%}")


if __name__ == "__main__":
    demo()
