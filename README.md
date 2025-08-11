# Stylized Retrieval-Augmented Generation (RAG) System

![Image](https://github.com/user-attachments/assets/1e8e8783-3515-475c-8003-765c82608a09)

This project implements a stylized Retrieval-Augmented Generation (RAG) system that performs text style transfer while preserving content meaning. The system combines keyword-based retrieval (BM25) with semantic search (Chroma embeddings) to retrieve relevant documents, then uses a large language model to rewrite text in specified styles.

## Key Features

- **Hybrid Retrieval System**: Combines BM25 (lexical search) and Chroma (semantic search)
- **Neural Style Transfer**: Rewrites text in different styles (formal, casual, Shakespearean, etc.)
- **Web Content Processing**: Fetches and processes web content for knowledge base
- **Ensemble Retriever**: Merges results from multiple retrieval methods
- **Hugging Face Integration**: Uses state-of-the-art language models

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Workflow Overview](#workflow-overview)
4. [Key Components](#key-components)
5. [Examples](#examples)
6. [Customization](#customization)
7. [Dependencies](#dependencies)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stylized-rag.git
cd stylized-rag
```

2. Set up Hugging Face API token:
```python
import os
from getpass import getpass

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("Enter your Hugging Face API token: ")
```

## Usage

Run the main script:
```python
python rag.py
```

The script will:
1. Scrape content from Wikipedia pages about AI and machine learning
2. Build the retrieval systems (BM25 + Chroma)
3. Prompt you for input text and desired style
4. Generate stylized output

Example interaction:
```python
Original Text: Explain machine learning.
Desired Style: as if it were a recipe for cooking

Styled Output:
"Machine learning is a method of data analysis that automates analytical model building... 
The following is a recipe for cooking machine learning:
Ingredients:
- Extremely large or complex datasets (Big Data)
- Artificial Neural Networks (Deep Learning)
..."
```

## Workflow Overview

1. **Content Acquisition**:
   - Fetch HTML content from specified URLs
   - Parse and clean text using BeautifulSoup
   - Split text into overlapping chunks

2. **Retrieval System Setup**:
   - Initialize BM25 retriever for keyword-based search
   - Build Chroma vector store with Hugging Face embeddings
   - Create ensemble retriever combining both approaches

3. **RAG Pipeline**:
   - Accept user query and style specification
   - Retrieve relevant documents using ensemble retriever
   - Format context for language model
   - Generate styled output using Mistral-7B model

4. **Output Generation**:
   - Parse and return styled text
   - Display results with source context

## Key Components

### 1. Retrievers
- **BM25Retriever**: Keyword-based retrieval using Okapi BM25 algorithm
- **Chroma Vector Store**: Semantic search using Hugging Face embeddings
- **EnsembleRetriever**: Combines results from both retrieval methods

### 2. Text Processing
- `fetch_and_parse()`: Fetches and cleans web content
- `split_text_into_documents()`: Chunks text into overlapping segments
- `format_docs()`: Prepares retrieved context for LLM input

### 3. Language Model Integration
- `setup_llm()`: Configures Hugging Face endpoint (default: Mistral-7B)
- Style transfer prompt template:
  ```python
  "Rewrite the given text in this {style} style.
   Use the context coming from \n{context}\n
   This is the original text: \n{original_text}\n"
  ```

### 4. RAG Pipeline
- `build_rag_chain()`: Integrates retrievers, formatters and LLM
- Hybrid retrieval with deduplication
- Context-aware style transfer

## Examples

### Shakespearean Style
**Input**: "Artificial intelligence is transforming industries"  
**Output**: 
"Verily, artificial intelligence doth work transformation upon industries diverse..."

### Formal Business Style
**Input**: "We're launching a new product next week"  
**Output**: 
"We are pleased to announce the forthcoming launch of a novel product in the next business week..."

### Recipe Style
**Input**: "Explain neural networks"  
**Output**:
"Neural Network Preparation Guide:
Ingredients:
- Input data layers (1 package)
- Hidden processing units (adjust to taste)
- Learning rate (1 tablespoon)
..."

## Customization

1. **Change Source Content**:
   ```python
   # In main.py
   example_urls = [
       "https://en.wikipedia.org/wiki/Your_Topic",
       "https://example.com/your_content"
   ]
   ```

2. **Adjust LLM Parameters**:
   ```python
   # In setup_llm()
   llm = HuggingFaceEndpoint(
       repo_id="google/gemma-7b",
       temperature=0.8,
       max_length=512
   )
   ```

3. **Modify Retrieval Settings**:
   ```python
   # In retrieve_and_format_context()
   context = retrieve_and_format_context(query, k=7)  # Change k value
   ```

4. **Add New Styles**:
   ```python
   # New style examples
   inputs = {
       "question": "your query",
       "style": "as a detective novel narrative",
       "original_text": "your text"
   }
   ```

## Dependencies

- Python 3.10+
- Required packages:
  ```requirements
  langchain
  langchain-community
  langchain-chroma
  langchain-huggingface
  beautifulsoup4
  rank_bm25
  requests
  huggingface_hub
  ```

## Acknowledgments
- Hugging Face for language models
- LangChain framework
- Wikipedia for sample content
