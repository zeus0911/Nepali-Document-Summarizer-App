
# Nepali Document Summarization App

This is a Streamlit web application for extractive document summarization of Nepali language. It allows users to upload PDF files, text, or images, and generates a summary of the content using natural language processing techniques.

## Approach

### Data Extraction

- **PDF Text Extraction**: The application uses PyMuPDF (`fitz`) to extract text from PDF documents. It iterates through each page of the PDF, renders it as an image, and then uses EasyOCR to perform optical character recognition (OCR) on the image to extract text.
  
- **Image Text Extraction**: For image inputs, the application uses EasyOCR directly to extract text from the image.

### Text Summarization

- **Sentence Embeddings**: Text summarization is performed using sentence embeddings. The application uses a pre-trained language model from the Hugging Face transformers library (`distilbert-base-nepali`) to generate embeddings for each sentence.

- **Clustering**: KMeans clustering algorithm is applied to group similar sentences together. The number of clusters is adjustable by the user. Each cluster represents a group of semantically similar sentences.

- **Summary Generation**: From each cluster, one representative sentence is chosen to form the summary. These representative sentences are concatenated to form the final summary.


- The application currently supports text summarization for Nepali language documents. However, it can be extended to support other languages by using appropriate pre-trained language models.



#### Interface
![Text Summarization](demo/interface.png)
*Interface for summarizing user-provided text.*


![PDF Summarization](demo/summarization.png)

*PDF summarization with num_clusters->2.*



## Installation

1. Clone the repository:

    ```bash
    git clone 
    cd Nepali-Document-Summarization-App
    ```

2. Install the required packages:

    ```bash
    pip install -r requirement.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run eng.py
    ```





