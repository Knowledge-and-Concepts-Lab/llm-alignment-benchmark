#!/usr/bin/env python3
"""
Hugging Face Model Information Extractor

This script takes a model identifier and extracts key architecture information
from the Hugging Face model repository, including:
- Number of layers
- Number of attention heads
- Attention dimensionality
- Number of tokens (vocabulary size)
- Embedding dimensions
"""

import requests
import json
import re
import sys
from typing import Dict, Any, Optional
from urllib.parse import urljoin

class HuggingFaceModelExtractor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.base_url = f"https://huggingface.co/{model_id}"
        self.api_url = f"https://huggingface.co/api/models/{model_id}"
        
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Fetch the model configuration from config.json"""
        config_url = f"{self.base_url}/raw/main/config.json"
        
        try:
            response = requests.get(config_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching config.json: {e}")
            return None
    
    def get_tokenizer_config(self) -> Optional[Dict[str, Any]]:
        """Fetch tokenizer configuration for vocabulary size"""
        tokenizer_url = f"{self.base_url}/raw/main/tokenizer_config.json"
        
        try:
            response = requests.get(tokenizer_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Could not fetch tokenizer_config.json: {e}")
            return None
    
    def get_vocab_size_from_tokenizer_json(self) -> Optional[int]:
        """Try to get vocab size from tokenizer.json"""
        tokenizer_json_url = f"{self.base_url}/raw/main/tokenizer.json"
        
        try:
            response = requests.get(tokenizer_json_url)
            response.raise_for_status()
            tokenizer_data = response.json()
            
            # Check different possible locations for vocab size
            if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                return len(tokenizer_data['model']['vocab'])
            elif 'vocab' in tokenizer_data:
                return len(tokenizer_data['vocab'])
                
        except requests.RequestException as e:
            print(f"Warning: Could not fetch tokenizer.json: {e}")
            return None
    
    def extract_model_info(self) -> Dict[str, Any]:
        """Extract all relevant model information"""
        config = self.get_config()
        if not config:
            return {"error": "Could not fetch model configuration"}
        
        # Initialize result dictionary
        result = {
            "model_id": self.model_id,
            "model_type": config.get("model_type", "Unknown"),
            "architecture": config.get("architectures", ["Unknown"])[0] if config.get("architectures") else "Unknown"
        }
        
        # Extract layer information
        layer_keys = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
        for key in layer_keys:
            if key in config:
                result["num_layers"] = config[key]
                break
        else:
            result["num_layers"] = "Not found"
        
        # Extract attention heads
        head_keys = ["num_attention_heads", "n_head", "num_heads", "attention_heads"]
        for key in head_keys:
            if key in config:
                result["num_attention_heads"] = config[key]
                break
        else:
            result["num_attention_heads"] = "Not found"
        
        # Extract key-value heads (for models with grouped-query attention)
        kv_head_keys = ["num_key_value_heads", "n_kv_head", "num_kv_heads"]
        for key in kv_head_keys:
            if key in config:
                result["num_key_value_heads"] = config[key]
                break
        
        # Extract hidden/embedding dimensions
        dim_keys = ["hidden_size", "n_embd", "d_model", "embedding_size"]
        for key in dim_keys:
            if key in config:
                result["embedding_dimensions"] = config[key]
                break
        else:
            result["embedding_dimensions"] = "Not found"
        
        # Calculate attention dimensionality
        if (result.get("embedding_dimensions", "Not found") != "Not found" and 
            result.get("num_attention_heads", "Not found") != "Not found"):
            result["attention_dimensionality"] = result["embedding_dimensions"] // result["num_attention_heads"]
        else:
            result["attention_dimensionality"] = "Not found"
        
        # Extract vocabulary size
        vocab_keys = ["vocab_size", "vocabulary_size", "n_vocab"]
        for key in vocab_keys:
            if key in config:
                result["vocabulary_size"] = config[key]
                break
        else:
            # Try to get from tokenizer files
            tokenizer_config = self.get_tokenizer_config()
            if tokenizer_config and "vocab_size" in tokenizer_config:
                result["vocabulary_size"] = tokenizer_config["vocab_size"]
            else:
                vocab_from_json = self.get_vocab_size_from_tokenizer_json()
                result["vocabulary_size"] = vocab_from_json if vocab_from_json else "Not found"
        
        # Extract other useful information
        result["max_position_embeddings"] = config.get("max_position_embeddings", 
                                                     config.get("n_positions", 
                                                              config.get("max_sequence_length", "Not found")))
        
        # Extract intermediate size (feed-forward dimension)
        ff_keys = ["intermediate_size", "n_inner", "ffn_dim", "feed_forward_size"]
        for key in ff_keys:
            if key in config:
                result["intermediate_size"] = config[key]
                break
        else:
            result["intermediate_size"] = "Not found"
        
        return result
    
    def print_model_info(self):
        """Print formatted model information"""
        info = self.extract_model_info()
        
        if "error" in info:
            print(f"Error: {info['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"MODEL INFORMATION: {info['model_id']}")
        print(f"{'='*60}")
        print(f"Model Type: {info['model_type']}")
        print(f"Architecture: {info['architecture']}")
        print(f"\nARCHITECTURE DETAILS:")
        print(f"  Number of Layers: {info['num_layers']}")
        print(f"  Number of Attention Heads: {info['num_attention_heads']}")
        if 'num_key_value_heads' in info:
            print(f"  Number of Key-Value Heads: {info['num_key_value_heads']}")
        print(f"  Attention Dimensionality: {info['attention_dimensionality']}")
        print(f"  Embedding Dimensions: {info['embedding_dimensions']}")
        print(f"  Vocabulary Size: {info['vocabulary_size']}")
        print(f"  Max Position Embeddings: {info['max_position_embeddings']}")
        print(f"  Intermediate Size (FFN): {info['intermediate_size']}")
        print(f"{'='*60}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python hf_model_extractor.py <model_id>")
        print("Example: python hf_model_extractor.py 01-ai/Yi-9B")
        sys.exit(1)
    
    model_id = sys.argv[1]
    
    # Handle different input formats
    if model_id.startswith("https://huggingface.co/"):
        # Extract model ID from URL
        model_id = model_id.replace("https://huggingface.co/", "").split("/")[0:2]
        model_id = "/".join(model_id)
    
    extractor = HuggingFaceModelExtractor(model_id)
    extractor.print_model_info()
    
    # Also return the raw data for programmatic use
    info = extractor.extract_model_info()
    return info

if __name__ == "__main__":
    main()
