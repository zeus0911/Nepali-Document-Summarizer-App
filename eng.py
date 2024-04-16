import streamlit as st
import easyocr
import fitz
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import re
import io
import base64
from PIL import Image

tokenizer = AutoTokenizer.from_pretrained("Sakonii/distilbert-base-nepali")
nepali_model = AutoModel.from_pretrained("Sakonii/distilbert-base-nepali")

def extract_text_from_scanned_pdf(pdf_path, language):

    reader = easyocr.Reader([language])

    try:
        text = ''

        pdf_document = fitz.open(pdf_path)


        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Render the page as an image (Pillow image)
            pix = page.get_pixmap()
            img_bytes = pix.samples


            img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(pix.height, pix.width, -1)


            page_text = reader.readtext(img_array)


            text += ' '.join([item[1] for item in page_text]) + '\n'

        return text.strip() 
    except Exception as e:
        return str(e)
    
def read_text_from_pillow_image(image, language):

    reader = easyocr.Reader([language])

    try:

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()


        result = reader.readtext(img_bytes)


        text = '\n'.join([item[1] for item in result])
        return text.strip()  
    except Exception as e:
        return str(e)

    
def summarize_text(model, tokenizer, text, n_clusters):

    if not isinstance(text, str):
        text = str(text)


    sentences = re.split(r'[।?\.।\?।।]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]


    embeddings = []
    for sentence in sentences:

        inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs["input_ids"]


        with torch.no_grad():
            output = model(input_ids)


        embedding = output.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(embedding.tolist())


    if n_clusters > len(embeddings):
        n_clusters = len(embeddings)


    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(embeddings)


    summary_sentences = []
    for cluster_index in range(kmeans.n_clusters):
        sentence_indices = np.where(kmeans.labels_ == cluster_index)[0]
        closest_sentence_index = sentence_indices[0] if any(sentence_indices) else 0
        summary_sentences.append(sentences[closest_sentence_index])

    final_summary = " । ".join(summary_sentences)
    return final_summary


    
@st.cache_data
def displayPDF(file):

    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')


    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'


    st.markdown(pdf_display, unsafe_allow_html=True)
    

st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App")

    option = st.selectbox("Choose Option", ('Pdf', 'Text', 'Image'))

    if option == 'Pdf':
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

        language = st.text_input("Enter language code (e.g., 'en' for English):")
        n_clusters = st.slider("Number of clusters", 1, 10, 3)

        if uploaded_file is not None:
            if st.button("Summarize"):
                col1, col2 = st.columns(2)
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                
                with col1:
                    st.info("Uploaded File")
                    pdf_view = displayPDF(filepath)
                with col2:
                    text = extract_text_from_scanned_pdf(filepath, 'ne')

                    st.info("Extracted Text:")
                    st.success(text)
                    
                    final_summary = summarize_text(nepali_model, tokenizer, text, n_clusters)

                    st.info("Summarization Complete")
                    st.success(final_summary)

    elif option == 'Text':

        text = st.text_input("Enter text to summarize:")

        n_clusters = st.slider("Number of clusters", 1, 10, 3)

        if st.button("Summarize"):
            final_summary = summarize_text(nepali_model, tokenizer, text, n_clusters)

            st.info("Summarization Complete")
            st.success(final_summary)

    elif option == 'Image':
        uploaded_file = st.file_uploader("Upload your image file", type=['jpg', 'png'])

        language = st.text_input("Enter language code (e.g., 'en' for English):")
        n_clusters = st.slider("Number of clusters", 1, 10, 3)

        if uploaded_file is not None:
            if st.button("Summarize"):
                col1, col2 = st.columns(2)
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                
                with col1:
                    st.info("Uploaded File")
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image.', use_column_width=True)
                with col2:
                    text = read_text_from_pillow_image(image, language)

                    st.info("Extracted Text:")
                    st.success(text)
                    
                    final_summary = summarize_text(nepali_model, tokenizer, text, n_clusters)

                    st.info("Summarization Complete")
                    st.success(final_summary)

if __name__ == "__main__":
    main()
