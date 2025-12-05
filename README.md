# Agentic AI Assignment - Kerala Ayurveda

## 📌 Overview

This repository contains my submission for the Agentic AI Internship assignment at Kerala Ayurveda. It demonstrates a practical approach to building a Safe & Precise RAG (Retrieval-Augmented Generation) system tailored for medical/Ayurvedic content.

## 📂 Repository Structure

simple_rag.py: The main Python script implementing the RAG engine.

products_catalog.csv: Structured data source (Products).

*.md: Unstructured data sources (Articles/FAQs).

🚀 Key Features

1. Content-Aware Chunking

Unlike standard fixed-size splitting, this project uses specific strategies for different data types to preserve medical safety context:

CSV Data: Row-based semantic chunking (keeps "Contraindications" linked to "Product Name").

Markdown Data: Recursive character splitting by Header (#, ##) to maintain section integrity.

2. Hybrid Retrieval Logic

The prototype implements a retrieval system using:

Semantic Search: Uses sentence-transformers (all-MiniLM-L6-v2) to understand intent (e.g., "stress" -> "Ashwagandha").

Cosine Similarity: To rank the most relevant documents.

## 🛠️ Setup & Installation

1. Prerequisites

Python 3.8 or higher

pip (Python Package Installer)

2. Install Dependencies

Run the following command to install the required "brain" libraries:

```bash
pip install pandas sentence-transformers scikit-learn numpy
```

(Note: On the first run, sentence-transformers will download the AI model, which is approximately 90MB.)

## 💻 How to Run

Clone this repository or ensure all files are in the same folder.

Open your terminal in the project folder.

Run the script:

```bash
python simple_rag.py
```


## Usage

The script runs in an interactive mode.

It will load and index the knowledge base (~54 items).

It will prompt you to enter a query.

Try asking:

"Is Ashwagandha safe for thyroid patients?"

"What are the benefits of Brahmi Tailam?"

The system will output the Retrieved Context from the CSV/Markdown files and the Simulated LLM Prompt.

