import PyPDF2
# import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import re
import os
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load Spacy model
# nlp = spacy.load("en_core_web_sm")


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load external data
from model_names import model_names, domain_names, library_names



# # Load the tokenizer and model from the transformers library
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")


# Load Doc2Vec model
with open("doc2vec_model.pkl", "rb") as f:
    doc2vec_model = pickle.load(f)

# Initialize Sentence Transformer model
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Function to preprocess text
def preprocess_and_join(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens into a single string
    return ' '.join(tokens)

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_organizations(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    named_entities = nltk.ne_chunk(pos_tags, binary=False)
    
    organizations = []
    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree) and subtree.label() in ['ORGANIZATION', 'WORK_OF_ART','PRODUCT']:
            entity_name = " ".join([token for token, pos in subtree.leaves()])
            organizations.append(entity_name)
    
    return organizations





# # Create a NER pipeline
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
# def extract_organizations(text):
#     ner_results = ner_pipeline(text)
#     organizations = [result['word'] for result in ner_results if result['entity'] == 'B-ORG' or result['entity'] == 'I-ORG']
#     return organizations

# Function to extract organizations from text
# def extract_organizations(text):
#     doc = nlp(text)
#     organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
#     return organizations



# Function to compare organizations mentioned in papers with model names, library names, and domain names
def compare_entities(papers_entities):
    entities = []
    for orgs in papers_entities:
        matched_models = []
        matched_libraries = []
        matched_domains = []
        for org in orgs:
            for model in model_names:
                if model.lower() == org.lower():
                    matched_models.append(org)
                    break
            for library in library_names:
                if library.lower() == org.lower():
                    matched_libraries.append(org)
                    break
            for domain in domain_names:
                if domain.lower() == org.lower():
                    matched_domains.append(org)
                    break
        entity_dict = {
            "models": list(set(matched_models)),
            "libraries": list(set(matched_libraries)),
            "domains": list(set(matched_domains))
        }
        entities.append(entity_dict)
    return entities


# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    similarity = intersection_size / union_size if union_size > 0 else 0
    return similarity

# Function to calculate similarity based on entities mentioned in papers using Jaccard similarity
def calculate_entities_similarity(papers_entities):
    jaccard_similarities = []
    for i in range(len(papers_entities)):
        for j in range(i + 1, len(papers_entities)):
            set1 = set(papers_entities[i]["models"] + papers_entities[i]["libraries"] + papers_entities[i]["domains"])
            set2 = set(papers_entities[j]["models"] + papers_entities[j]["libraries"] + papers_entities[j]["domains"])
            similarity = calculate_jaccard_similarity(set1, set2)
            jaccard_similarities.append(similarity)
    mean_similarity_entities = np.mean(jaccard_similarities)
    return mean_similarity_entities

# Function to analyze similarity using TF-IDF
def analyze_similarity_tfidf(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix)
    return similarity_scores

# Function to analyze similarity using LSA
def analyze_similarity_lsa(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=100, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    similarity_scores = cosine_similarity(lsa_matrix)
    return similarity_scores

# Function to compute document embeddings using trained Doc2Vec model
def compute_doc2vec_embeddings(documents):
    embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
    return embeddings

# Function to calculate similarity using document embeddings and cosine similarity
def calculate_similarity_doc2vec(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# Function to calculate similarity using sentence-transformers
def calculate_similarity_transformer(documents):
    embeddings = sentence_model.encode(documents)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


# Function to plot Hierarchical Clustering Dendrogram
def plot_dendrogram(similarity_matrix, method='ward'):
    plt.figure(figsize=(10, 7))
    dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Papers')
    plt.ylabel('Normalized Distance')
    plt.ylim((0, 1))
    plt.axhline(y=1, color='r', linestyle='--')
    st.pyplot(plt)


# Function to extract abstract from text
def extract_abstract(text, max_word_count=250):
    abstract_start_keywords = ['abstract', 'Abstract']
    abstract_end_keywords = [
        'introduction', ' Introduction', 'index terms', 'Index Terms',
        'Index terms', 'Keywords', 'keywords'
    ]
    abstract_end_sequence = '    '  # Four spaces

    abstract_start = None
    abstract_end = None

    # Find the start of the abstract
    for keyword in abstract_start_keywords:
        start = text.lower().find(keyword.lower())
        if start != -1:
            abstract_start = start + len(keyword)
            break

    if abstract_start is not None:
        # Find the end of the abstract using keywords
        for keyword in abstract_end_keywords:
            end = text.lower().find(keyword.lower(), abstract_start)
            if end != -1 and (abstract_end is None or end < abstract_end):
                abstract_end = end

        # Find the end of the abstract using
        end_by_space = text.find(abstract_end_sequence, abstract_start)
                # Find the end of the abstract using space sequence
        end_by_space = text.find(abstract_end_sequence, abstract_start)
        if end_by_space != -1 and (abstract_end is None or end_by_space < abstract_end):
            abstract_end = end_by_space

        # Extract the abstract
        if abstract_end is not None:
            abstract = text[abstract_start:abstract_end].strip()
        else:
            abstract = text[abstract_start:].strip()

        # Limit by max word count
        words = abstract.split()
        if len(words) > max_word_count:
            abstract = ' '.join(words[:max_word_count])

        # Ensure it ends at a sentence boundary
        last_sentence_end = max(abstract.rfind('.'), abstract.rfind('!'), abstract.rfind('?'))
        if last_sentence_end != -1 and last_sentence_end + 1 < len(abstract):
            abstract = abstract[:last_sentence_end + 1]

        return abstract
    else:
        return None


# Function to compute similarity between user abstract and papers' abstracts
# def compute_similarity(user_abstract, papers_abstracts):
#     documents = [user_abstract] + papers_abstracts
#     vectorizer = TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()

#     cosine_matrix = cosine_similarity(vectors)
#     user_similarity_scores = cosine_matrix[0][1:]

#     return user_similarity_scores


def compute_similarity(user_abstract, papers_abstracts):
    documents = list(papers_abstracts.values())  # Extracting only the abstracts
    documents.append(user_abstract)  # Adding the user's abstract to the list
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity(vectors)
    user_similarity_scores = cosine_matrix[-1][:-1]  # Removing the similarity score of the user's abstract with itself

    return user_similarity_scores


def extract_abstracts_from_directory(directory):
    abstracts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            abstract = extract_abstract(text)
            if abstract:
                abstracts[filename] = abstract
    return abstracts


# def extract_abstracts_from_directory(directory):
#     abstracts = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(directory, filename)
#             text = extract_text_from_pdf(pdf_path)
#             abstract = extract_abstract(text)
#             if abstract:
#                 abstracts.append((filename, abstract))
#     return abstracts


# def extract_abstracts_from_directory(directory):
#     abstracts = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(directory, filename)
#             text = extract_text_from_pdf(pdf_path)
#             abstract = extract_abstract(text)
#             if abstract:
#                 abstracts.append(abstract)
#     return abstracts


def generate_summary(text):
    # Load BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=350, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


# def generate_summary(text, max_length=1024):
#     # Load pre-trained model and tokenizer

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     # Split input text into chunks of appropriate length
#     input_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

#     # Generate summary for each chunk
#     summaries = []

#     for chunk in input_chunks:
#         # Tokenize input text chunk
#         inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", truncation=True)

#         # Generate summary for chunk
#         summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

#         # Decode summary for chunk
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)

#     # Combine summaries of all chunks
#     combined_summary = " ".join(summaries)
    
#     return combined_summary


def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return ' '.join(sentences)


# Main function
def main():
    st.title("Research Paper Similarity Analyzer")
   
    st.sidebar.header("Upload Papers")
    uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)
    st.sidebar.header("Find Relevant Papers")
    user_abstract = st.sidebar.text_area("Enter your abstract:", "")

    # User input for directory containing PDF files
    pdf_directory = "data"

    # Button to trigger analysis
    if st.sidebar.button("Analyze"):
        if not user_abstract:
            st.error("Please enter your abstract.")
        elif not pdf_directory:
            st.error("Please enter the path to the directory containing PDF files.")
        else:
            # Extract abstracts from PDF files in the directory
            papers_abstracts_raw = extract_abstracts_from_directory(pdf_directory)
            papers_abstracts = {filename: preprocess_text(abstract) for filename, abstract in papers_abstracts_raw.items()}
            
            # Compute similarity between user abstract and papers' abstracts
            similarity_scores = compute_similarity(user_abstract, papers_abstracts)

            # Sort papers by similarity score and select top 5
            top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]
            top_papers = [(list(papers_abstracts.keys())[i], similarity_scores[i]) for i in top_indices]

            top_paper_names = []
            # Display top 5 papers
            st.subheader("Top 5 Papers:")
            
            for i, (paper, score) in enumerate(top_papers):
                abstract = papers_abstracts.get(paper, "Abstract not available")
                st.write(f"**Paper {i+1}:** {paper}")
                st.write(f"**Similarity Score:** {score:.2f}")
                st.write(f"**Abstract:** {abstract}")

                if st.button(f"Show Models and Libraries for Paper {i+1}"):
                    text = extract_text_from_pdf(os.path.join(pdf_directory, paper))
                    entities = extract_organizations(text)
                    matched_entities = compare_entities([entities])[0]
                    st.write(f"**Models:** {', '.join(matched_entities['models']) if matched_entities['models'] else 'None'}")
                    st.write(f"**Libraries:** {', '.join(matched_entities['libraries']) if matched_entities['libraries'] else 'None'}")
                    st.write(f"**Domains:** {', '.join(matched_entities['domains']) if matched_entities['domains'] else 'None'}")
                    
                st.write("\n")
                top_paper_names.append(paper)
                
                with st.expander(f"Entities for Paper {i+1}"):
                    st.write(f"Selected Paper: {paper}")
                    # st.write(f"pdf_directory: {pdf_directory}")  # Debugging
                    # st.write(f"papers_abstracts_raw: {papers_abstracts_raw}")  # Debugging
                    # st.write(f"papers_abstracts: {papers_abstracts}")  # Debugging
        
                    try:
                        # Extract text from the PDF file based on its name
                        pdf_path = os.path.join(pdf_directory, paper)
                        # st.write(f"PDF Path: {pdf_path}")  # Debugging
            
                        paper_text = extract_text_from_pdf(pdf_path)
                        # st.write(f"Paper Text: {paper_text}")  # Debugging
            
                        organizations = extract_organizations(paper_text)  # Extract organizations from the text
                        # st.write(f"Organizations: {organizations}")  # Debugging
            
                        entities = compare_entities([organizations])[0]  # Compare entities
                        st.write(f"**Entities**: {entities}")  # Debugging
        
                        st.write(f"**Models**: {entities['models']}")
                        st.write(f"**Libraries**: {entities['libraries']}")
                        st.write(f"**Domains**: {entities['domains']}")
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                
                with st.expander(f"Summary Of Paper {i+1}"):
                    st.write(f"Selected Paper: {paper}")
                    pdf_path = os.path.join(pdf_directory, paper)
                    paper_text = extract_text_from_pdf(pdf_path)
                    summary = generate_summary(paper_text)
                    st.write(f"**Summary**:{summary}")


    if uploaded_files:
        if len(uploaded_files) == 1:
            st.sidebar.warning("Upload at least two papers to perform comparison.")
        else:
            # Extract text and organizations from uploaded files
            papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            papers_entities = [extract_organizations(text) for text in papers_text]
            
            # Print entities extracted from each paper under the ORG label
            st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
            for i, entities in enumerate(papers_entities):
                st.write(f"Paper {i+1}: {', '.join(entities)}")
                
            # Compare organizations mentioned in papers with model names, library names, and domain names
            papers_entities_matched = compare_entities(papers_entities)

            # Display matched entities under their respective labels
            st.subheader("Entities Categorized:")
            for i, entities in enumerate(papers_entities_matched):
                st.write(f"Paper {i+1}:")
                if entities["models"]:
                    st.write(f"Model Names: {', '.join(entities['models'])}")
                if entities["libraries"]:
                    st.write(f"Library Names: {', '.join(entities['libraries'])}")
                if entities["domains"]:
                    st.write(f"Domain Names: {', '.join(entities['domains'])}")  
           
            # Calculate model-based similarity using Jaccard similarity
            mean_similarity_entities = calculate_entities_similarity(papers_entities_matched)
            
            # Perform similarity analysis based on text using TF-IDF
            similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
            mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
            # Perform similarity analysis based on text using LSA
            similarity_scores_lsa = analyze_similarity_lsa(papers_text)
            mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
            # Compute document embeddings using Doc2Vec
            embeddings = compute_doc2vec_embeddings(papers_text)
           
            # Perform similarity analysis based on document embeddings using Doc2Vec
            similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
            mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
            # Calculate similarity using sentence-transformers
            similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
            mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
            # Display results
            st.subheader("Analysis Results")
            st.write(f"Mean Similarity (models, libraries, domains): {mean_similarity_entities:.2f}")
            st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
            st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
            st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
            st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

            # Plot Hierarchical Clustering Dendrogram
            st.subheader("Hierarchical Clustering Dendrogram")
            plot_dendrogram(similarity_matrix_transformer)

if __name__ == "__main__":
    main()










#works
# import PyPDF2
# # import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from sentence_transformers import SentenceTransformer, util
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc
# import re
# import os
# # from transformers import AutoTokenizer, AutoModelForTokenClassification
# # from transformers import pipeline


# # Load Spacy model
# # nlp = spacy.load("en_core_web_sm")

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')

# # Load external data
# from model_names import model_names, domain_names, library_names

# # # Load the tokenizer and model from the transformers library
# # tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# # model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# # Load Doc2Vec model
# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# # Initialize Sentence Transformer model
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF file
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text


# def extract_organizations(text):
#     words = nltk.word_tokenize(text)
#     pos_tags = nltk.pos_tag(words)
#     named_entities = nltk.ne_chunk(pos_tags, binary=False)
    
#     organizations = []
#     for subtree in named_entities:
#         if isinstance(subtree, nltk.Tree) and subtree.label() in ['ORGANIZATION', 'WORK_OF_ART','PRODUCT']:
#             entity_name = " ".join([token for token, pos in subtree.leaves()])
#             organizations.append(entity_name)
    
#     return organizations





# # # Create a NER pipeline
# # ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
# # def extract_organizations(text):
# #     ner_results = ner_pipeline(text)
# #     organizations = [result['word'] for result in ner_results if result['entity'] == 'B-ORG' or result['entity'] == 'I-ORG']
# #     return organizations

# # Function to extract organizations from text
# # def extract_organizations(text):
# #     doc = nlp(text)
# #     organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
# #     return organizations

# # Function to compare organizations mentioned in papers with model names, library names, and domain names
# def compare_entities(papers_entities):
#     entities = []
#     for orgs in papers_entities:
#         matched_models = []
#         matched_libraries = []
#         matched_domains = []
#         for org in orgs:
#             for model in model_names:
#                 if model.lower() == org.lower():
#                     matched_models.append(org)
#                     break
#             for library in library_names:
#                 if library.lower() == org.lower():
#                     matched_libraries.append(org)
#                     break
#             for domain in domain_names:
#                 if domain.lower() == org.lower():
#                     matched_domains.append(org)
#                     break
#         entity_dict = {
#             "models": list(set(matched_models)),
#             "libraries": list(set(matched_libraries)),
#             "domains": list(set(matched_domains))
#         }
#         entities.append(entity_dict)
#     return entities

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to calculate similarity based on entities mentioned in papers using Jaccard similarity
# def calculate_entities_similarity(papers_entities):
#     jaccard_similarities = []
#     for i in range(len(papers_entities)):
#         for j in range(i + 1, len(papers_entities)):
#             set1 = set(papers_entities[i]["models"] + papers_entities[i]["libraries"] + papers_entities[i]["domains"])
#             set2 = set(papers_entities[j]["models"] + papers_entities[j]["libraries"] + papers_entities[j]["domains"])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_entities = np.mean(jaccard_similarities)
#     return mean_similarity_entities

# # Function to analyze similarity using TF-IDF
# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to analyze similarity using LSA
# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to plot Hierarchical Clustering Dendrogram
# def plot_dendrogram(similarity_matrix, method='ward'):
#     plt.figure(figsize=(10, 7))
#     dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Papers')
#     plt.ylabel('Normalized Distance')
#     plt.ylim((0, 1))
#     plt.axhline(y=1, color='r', linestyle='--')
#     st.pyplot(plt)

# # Function to extract abstract from text
# def extract_abstract(text, max_word_count=250):
#     abstract_start_keywords = ['abstract', 'Abstract']
#     abstract_end_keywords = [
#         'introduction', ' Introduction', 'index terms', 'Index Terms',
#         'Index terms', 'Keywords', 'keywords'
#     ]
#     abstract_end_sequence = '    '  # Four spaces

#     abstract_start = None
#     abstract_end = None

#     # Find the start of the abstract
#     for keyword in abstract_start_keywords:
#         start = text.lower().find(keyword.lower())
#         if start != -1:
#             abstract_start = start + len(keyword)
#             break

#     if abstract_start is not None:
#         # Find the end of the abstract using keywords
#         for keyword in abstract_end_keywords:
#             end = text.lower().find(keyword.lower(), abstract_start)
#             if end != -1 and (abstract_end is None or end < abstract_end):
#                 abstract_end = end

#         # Find the end of the abstract using
#         end_by_space = text.find(abstract_end_sequence, abstract_start)
#                 # Find the end of the abstract using space sequence
#         end_by_space = text.find(abstract_end_sequence, abstract_start)
#         if end_by_space != -1 and (abstract_end is None or end_by_space < abstract_end):
#             abstract_end = end_by_space

#         # Extract the abstract
#         if abstract_end is not None:
#             abstract = text[abstract_start:abstract_end].strip()
#         else:
#             abstract = text[abstract_start:].strip()

#         # Limit by max word count
#         words = abstract.split()
#         if len(words) > max_word_count:
#             abstract = ' '.join(words[:max_word_count])

#         # Ensure it ends at a sentence boundary
#         last_sentence_end = max(abstract.rfind('.'), abstract.rfind('!'), abstract.rfind('?'))
#         if last_sentence_end != -1 and last_sentence_end + 1 < len(abstract):
#             abstract = abstract[:last_sentence_end + 1]

#         return abstract
#     else:
#         return None

# # Function to compute similarity between user abstract and papers' abstracts
# # def compute_similarity(user_abstract, papers_abstracts):
# #     documents = [user_abstract] + papers_abstracts
# #     vectorizer = TfidfVectorizer().fit_transform(documents)
# #     vectors = vectorizer.toarray()

# #     cosine_matrix = cosine_similarity(vectors)
# #     user_similarity_scores = cosine_matrix[0][1:]

# #     return user_similarity_scores

# def compute_similarity(user_abstract, papers_abstracts):
#     documents = list(papers_abstracts.values())  # Extracting only the abstracts
#     documents.append(user_abstract)  # Adding the user's abstract to the list
#     vectorizer = TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()

#     cosine_matrix = cosine_similarity(vectors)
#     user_similarity_scores = cosine_matrix[-1][:-1]  # Removing the similarity score of the user's abstract with itself

#     return user_similarity_scores


# def extract_abstracts_from_directory(directory):
#     abstracts = {}
#     for filename in os.listdir(directory):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(directory, filename)
#             text = extract_text_from_pdf(pdf_path)
#             abstract = extract_abstract(text)
#             if abstract:
#                 abstracts[filename] = abstract
#     return abstracts


# # def extract_abstracts_from_directory(directory):
# #     abstracts = []
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.pdf'):
# #             pdf_path = os.path.join(directory, filename)
# #             text = extract_text_from_pdf(pdf_path)
# #             abstract = extract_abstract(text)
# #             if abstract:
# #                 abstracts.append((filename, abstract))
# #     return abstracts


# # def extract_abstracts_from_directory(directory):
# #     abstracts = []
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.pdf'):
# #             pdf_path = os.path.join(directory, filename)
# #             text = extract_text_from_pdf(pdf_path)
# #             abstract = extract_abstract(text)
# #             if abstract:
# #                 abstracts.append(abstract)
# #     return abstracts

# def preprocess_text(text):
#     sentences = nltk.sent_tokenize(text)
#     return ' '.join(sentences)

# # Main function
# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)
#     user_abstract = st.sidebar.text_area("Enter your abstract:", "")

#     # User input for directory containing PDF files
#     pdf_directory = "data"

#     # Button to trigger analysis
#     if st.sidebar.button("Analyze"):
#         if not user_abstract:
#             st.error("Please enter your abstract.")
#         elif not pdf_directory:
#             st.error("Please enter the path to the directory containing PDF files.")
#         else:
#             # Extract abstracts from PDF files in the directory
#             papers_abstracts_raw = extract_abstracts_from_directory(pdf_directory)
#             papers_abstracts = {filename: preprocess_text(abstract) for filename, abstract in papers_abstracts_raw.items()}
            
#             # Compute similarity between user abstract and papers' abstracts
#             similarity_scores = compute_similarity(user_abstract, papers_abstracts)

#             # Sort papers by similarity score and select top 5
#             top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]
#             top_papers = [(list(papers_abstracts.keys())[i], similarity_scores[i]) for i in top_indices]

#             # Display top 5 papers
#             st.subheader("Top 5 Papers:")
#             for i, (paper, score) in enumerate(top_papers):
#                 abstract = papers_abstracts.get(paper, "Abstract not available")
#                 st.write(f"**Paper {i+1}:** {paper}")
#                 st.write(f"**Similarity Score:** {score:.2f}")
#                 st.write(f"**Abstract:** {abstract}")
#                 st.write("\n")

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and organizations from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             papers_entities = [extract_organizations(text) for text in papers_text]
            
#             # Print entities extracted from each paper under the ORG label
#             st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
#             for i, entities in enumerate(papers_entities):
#                 st.write(f"Paper {i+1}: {', '.join(entities)}")
                
#             # Compare organizations mentioned in papers with model names, library names, and domain names
#             papers_entities_matched = compare_entities(papers_entities)

#             # Display matched entities under their respective labels
#             st.subheader("Entities Categorized:")
#             for i, entities in enumerate(papers_entities_matched):
#                 st.write(f"Paper {i+1}:")
#                 if entities["models"]:
#                     st.write(f"Model Names: {', '.join(entities['models'])}")
#                 if entities["libraries"]:
#                     st.write(f"Library Names: {', '.join(entities['libraries'])}")
#                 if entities["domains"]:
#                     st.write(f"Domain Names: {', '.join(entities['domains'])}")  
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_entities = calculate_entities_similarity(papers_entities_matched)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (models, libraries, domains): {mean_similarity_entities:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             # Plot Hierarchical Clustering Dendrogram
#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# if __name__ == "__main__":
#     main()









# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)
#     user_abstract = st.sidebar.text_area("Enter your abstract:", "")

#     # User input for directory containing PDF files
#     pdf_directory = "data"

#     # Button to trigger analysis
#     if st.sidebar.button("Analyze"):
#         if not user_abstract:
#             st.error("Please enter your abstract.")
#         elif not pdf_directory:
#             st.error("Please enter the path to the directory containing PDF files.")
#         else:
#             # Extract abstracts from PDF files in the directory
#             papers_abstracts_raw = extract_abstracts_from_directory(pdf_directory)
#             papers_abstracts = [preprocess_text(abstract) for abstract in papers_abstracts_raw] 
            
#             # Compute similarity between user abstract and papers' abstracts
#             similarity_scores = compute_similarity(user_abstract, papers_abstracts)

#             # Sort papers by similarity score and select top 5
#             top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]
#             top_papers = [(os.listdir(pdf_directory)[i], similarity_scores[i]) for i in top_indices]

#             # Display top 5 papers
#             # st.subheader("Top 5 Papers:")
#             # for paper, score in top_papers:
#             #     st.write(f"Paper: {paper}, Similarity Score: {score:.2f}")
#             st.subheader("Top 5 Papers:")
#             for index, (paper, score) in enumerate(top_papers):
#                 abstract = papers_abstracts[index] if index < len(papers_abstracts) else "Abstract not available"
#                 st.write(f"**Paper {index+1}:** {paper}")
#                 st.write(f"**Similarity Score:** {score:.2f}")
#                 st.write(f"**Abstract:** {abstract}")
#                 st.write("\n")
                
#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and organizations from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             papers_entities = [extract_organizations(text) for text in papers_text]
            
#             # Print entities extracted from each paper under the ORG label
#             st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
#             for i, entities in enumerate(papers_entities):
#                 st.write(f"Paper {i+1}: {', '.join(entities)}")
                
#             # Compare organizations mentioned in papers with model names, library names, and domain names
#             papers_entities_matched = compare_entities(papers_entities)

#             # Display matched entities under their respective labels
#             st.subheader("Entities Categorized:")
#             for i, entities in enumerate(papers_entities_matched):
#                 st.write(f"Paper {i+1}:")
#                 if entities["models"]:
#                     st.write(f"Model Names: {', '.join(entities['models'])}")
#                 if entities["libraries"]:
#                     st.write(f"Library Names: {', '.join(entities['libraries'])}")
#                 if entities["domains"]:
#                     st.write(f"Domain Names: {', '.join(entities['domains'])}")  
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_entities = calculate_entities_similarity(papers_entities_matched)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            

                
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (models, libraries, domains): {mean_similarity_entities:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             # Plot Hierarchical Clustering Dendrogram
#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# # Check if the script is being run directly
# if __name__ == "__main__":
#     main()














# import PyPDF2
# import spacy
# import os
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from sentence_transformers import SentenceTransformer, util
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc

# nlp = spacy.load("en_core_web_sm")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# from model_names import model_names, domain_names, library_names

# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# def preprocess_and_join(text):
#     tokens = word_tokenize(text)
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)

# def extract_text_from_pdf(file):
#     reader = PyPDF2.PdfReader(file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# def extract_abstract(text):
#     abstract_start_keywords = ['abstract', 'Abstract']
#     abstract_end_keywords = ['introduction', 'Introduction', 'index terms', 'Index Terms', 'Index terms']

#     abstract_start = None
#     abstract_end = None

#     for keyword in abstract_start_keywords:
#         start = text.lower().find(keyword.lower())
#         if start != -1:
#             abstract_start = start
#             break

#     if abstract_start is not None:
#         for keyword in abstract_end_keywords:
#             end = text.lower().find(keyword.lower(), abstract_start)
#             if end != -1:
#                 abstract_end = end
#                 break

#     if abstract_start is not None and abstract_end is not None:
#         return text[abstract_start:abstract_end].strip()
#     else:
#         return None

# def preprocess_text(text):
#     sentences = nltk.sent_tokenize(text)
#     return ' '.join(sentences)

# def compute_similarity(user_abstract, papers_abstracts):
#     documents = [user_abstract] + papers_abstracts
#     vectorizer = TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()
#     cosine_matrix = cosine_similarity(vectors)
#     user_similarity_scores = cosine_matrix[0][1:]
#     return user_similarity_scores

# def extract_organizations(text):
#     doc = nlp(text)
#     organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
#     return organizations

# def compare_entities(papers_entities):
#     entities = []
#     for orgs in papers_entities:
#         matched_models = []
#         matched_libraries = []
#         matched_domains = []
#         for org in orgs:
#             for model in model_names:
#                 if model.lower() == org.lower():
#                     matched_models.append(org)
#                     break
#             for library in library_names:
#                 if library.lower() == org.lower():
#                     matched_libraries.append(org)
#                     break
#             for domain in domain_names:
#                 if domain.lower() == org.lower():
#                     matched_domains.append(org)
#                     break
#         entity_dict = {
#             "models": list(set(matched_models)),
#             "libraries": list(set(matched_libraries)),
#             "domains": list(set(matched_domains))
#         }
#         entities.append(entity_dict)
#     return entities

# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# def calculate_entities_similarity(papers_entities):
#     jaccard_similarities = []
#     for i in range(len(papers_entities)):
#         for j in range(i + 1, len(papers_entities)):
#             set1 = set(papers_entities[i]["models"] + papers_entities[i]["libraries"] + papers_entities[i]["domains"])
#             set2 = set(papers_entities[j]["models"] + papers_entities[j]["libraries"] + papers_entities[j]["domains"])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_entities = np.mean(jaccard_similarities)
#     return mean_similarity_entities

# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def plot_dendrogram(similarity_matrix, method='ward'):
#     plt.figure(figsize=(10, 7))
#     dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Papers')
#     plt.ylabel('Normalized Distance')
#     plt.ylim((0, 1))
#     plt.axhline(y=1, color='r', linestyle='--')
#     st.pyplot(plt)

# def main():
#     st.title("Research Paper Similarity Analyzer")

#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)
#     user_abstract = st.sidebar.text_area("Enter your abstract")

#     if uploaded_files:
#         if len(uploaded_files) < 2:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]

#             if user_abstract:
#                 papers_abstracts = [extract_abstract(text) for text in papers_text]
#                 papers_abstracts = [preprocess_text(abstract) for abstract in papers_abstracts if abstract is not None]

#                 user_abstract_preprocessed = preprocess_text(user_abstract)
#                 similarity_scores = compute_similarity(user_abstract_preprocessed, papers_abstracts)

#                 paper_names = [file.name for file in uploaded_files]
#                 top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]
#                 top_papers = [(paper_names[i], similarity_scores[i]) for i in top_indices]

#                 st.subheader("Top Similar Papers:")
#                 for paper, score in top_papers:
#                     st.write(f"Paper: {paper}, Similarity Score: {score:.2f}")

#             papers_entities = [extract_organizations(text) for text in papers_text]
#             st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
#             for i, entities in enumerate(papers_entities):
#                 st.write(f"Paper {i+1}: {', '.join(entities)}")

#             papers_entities_matched = compare_entities(papers_entities)

#             st.subheader("Entities Categorized:")
#             for i, entities in enumerate(papers_entities_matched):
#                 st.write(f"Paper {i+1}:")
#                 if entities["models"]:
#                     st.write(f"Model Names: {', '.join(entities['models'])}")
#                 if entities["libraries"]:
#                     st.write(f"Library Names: {', '.join(entities['libraries'])}")
#                 if entities["domains"]:
#                     st.write(f"Domain Names: {', '.join(entities['domains'])}")

#             mean_similarity_entities = calculate_entities_similarity(papers_entities_matched)
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = np.mean(similarity_scores_tfidf)

#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = np.mean(similarity_scores_lsa)

#             embeddings = compute_doc2vec_embeddings(papers_text)
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = np.mean(similarity_matrix_doc2vec)

#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = np.mean(similarity_matrix_transformer)

#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (models, libraries, domains): {mean_similarity_entities:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# if __name__ == "__main__":
#     main()



# import PyPDF2
# import spacy
# import os
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from sentence_transformers import SentenceTransformer, util
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc

# nlp = spacy.load("en_core_web_sm")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load model names, library names, and domain names
# # with open("model_names.py", "rb") as f:
# #     model_names = pickle.load(f)
# # with open("library_names.py", "rb") as f:
# #     library_names = pickle.load(f)
# # with open("domain_names.py", "rb") as f:
# #     domain_names = pickle.load(f)

# from model_names import model_names,domain_names,library_names
# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# # def extract_text_from_pdf(pdf_file):
# #     reader = PyPDF2.PdfReader(pdf_file)
# #     text = ''
# #     for page in reader.pages:
# #         text += page.extract_text()
# #     return text

# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text

# # for abstract extraction
# def extract_abstract(text):
#     abstract_start_keywords = ['abstract', 'Abstract']
#     abstract_end_keywords = ['introduction', 'Introduction', 'index terms', 'Index Terms', 'Index terms']

#     abstract_start = None
#     abstract_end = None

#     for keyword in abstract_start_keywords:
#         start = text.lower().find(keyword.lower())
#         if start != -1:
#             abstract_start = start
#             break

#     if abstract_start is not None:
#         for keyword in abstract_end_keywords:
#             end = text.lower().find(keyword.lower(), abstract_start)
#             if end != -1:
#                 abstract_end = end
#                 break

#     if abstract_start is not None and abstract_end is not None:
#         return text[abstract_start:abstract_end].strip()
#     else:
#         return None

# #short preprocessing code for abstract
# def preprocess_text(text):
#     sentences = nltk.sent_tokenize(text)
#     return ' '.join(sentences)

# def compute_similarity(user_abstract, papers_abstracts):
#     documents = [user_abstract] + papers_abstracts
#     vectorizer = TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()

#     cosine_matrix = cosine_similarity(vectors)
#     user_similarity_scores = cosine_matrix[0][1:]

#     return user_similarity_scores

# # Function to extract organizations from text
# def extract_organizations(text):
#     doc = nlp(text)
#     # organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
#     organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG","PRODUCT","WORK_OF_ART"]]
#     return organizations

# # Function to compare organizations mentioned in papers with model names, library names, and domain names
# def compare_entities(papers_entities):
#     entities = []
#     for orgs in papers_entities:
#         matched_models = []
#         matched_libraries = []
#         matched_domains = []
#         for org in orgs:
#             for model in model_names:
#                 if model.lower() == org.lower():
#                     matched_models.append(org)
#                     break
#             for library in library_names:
#                 if library.lower() == org.lower():
#                     matched_libraries.append(org)
#                     break
#             for domain in domain_names:
#                 if domain.lower() == org.lower():
#                     matched_domains.append(org)
#                     break
#         entity_dict = {
#             "models": list(set(matched_models)),
#             "libraries": list(set(matched_libraries)),
#             "domains": list(set(matched_domains))
#         }
#         entities.append(entity_dict)
#     return entities

# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to calculate similarity based on entities mentioned in papers using Jaccard similarity
# def calculate_entities_similarity(papers_entities):
#     jaccard_similarities = []
#     for i in range(len(papers_entities)):
#         for j in range(i + 1, len(papers_entities)):
#             set1 = set(papers_entities[i]["models"] + papers_entities[i]["libraries"] + papers_entities[i]["domains"])
#             set2 = set(papers_entities[j]["models"] + papers_entities[j]["libraries"] + papers_entities[j]["domains"])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_entities = np.mean(jaccard_similarities)
#     return mean_similarity_entities


# # def calculate_entities_similarity(papers_entities):
# #     jaccard_similarities = []
# #     for i in range(len(papers_entities)):
# #         for j in range(i + 1, len(papers_entities)):
# #             set1 = set(papers_entities[i])
# #             set2 = set(papers_entities[j])
# #             similarity = calculate_jaccard_similarity(set1, set2)
# #             jaccard_similarities.append(similarity)
# #     mean_similarity_entities = np.mean(jaccard_similarities)
# #     return mean_similarity_entities

# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def plot_dendrogram(similarity_matrix, method='ward'):
#     plt.figure(figsize=(10, 7))
#     dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Papers')
#     plt.ylabel('Normalized Distance')
#     plt.ylim((0, 1))  
#     plt.axhline(y=1, color='r', linestyle='--')  
#     st.pyplot(plt)

# def main():
#     st.title("Research Paper Similarity Analyzer")

#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)
#     user_abstract = st.sidebar.text_area("Enter your abstract")

#     if uploaded_files:
#         if len(uploaded_files) < 2:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             if user_abstract:
#                 # Extract abstracts from uploaded files
#                 papers_abstracts = [extract_abstract(text) for text in papers_text]
#                 papers_abstracts = [preprocess_text(abstract) for abstract in papers_abstracts if abstract is not None]

#                 # Compute similarity between user abstract and paper abstracts
#                 user_abstract_preprocessed = preprocess_text(user_abstract)
#                 similarity_scores = compute_similarity(user_abstract_preprocessed, papers_abstracts)

#                 # Find top 5 similar papers
#                 paper_names = [file.name for file in uploaded_files]
#                 top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]
#                 top_papers = [(paper_names[i], similarity_scores[i]) for i in top_indices]

#                 # Display top similar papers
#                 st.subheader("Top Similar Papers:")
#                 for paper, score in top_papers:
#                     st.write(f"Paper: {paper}, Similarity Score: {score:.2f}")

#             # Extract and display entities from the papers
#             papers_entities = [extract_organizations(text) for text in papers_text]
#             st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
#             for i, entities in enumerate(papers_entities):
#                 st.write(f"Paper {i+1}: {', '.join(entities)}")

#             # Compare organizations mentioned in papers with model names, library names, and domain names
#             papers_entities_matched = compare_entities(papers_entities)

#             # Display matched entities under their respective labels
#             st.subheader("Entities Categorized:")
#             for i, entities in enumerate(papers_entities_matched):
#                 st.write(f"Paper {i+1}:")
#                 if entities["models"]:
#                     st.write(f"Model Names: {', '.join(entities['models'])}")
#                 if entities["libraries"]:
#                     st.write(f"Library Names: {', '.join(entities['libraries'])}")
#                 if entities["domains"]:
#                     st.write(f"Domain Names: {', '.join(entities['domains'])}")

#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_entities = calculate_entities_similarity(papers_entities_matched)

#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]

#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]

#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)

#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]

#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (models, libraries, domains): {mean_similarity_entities:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             # Plot Hierarchical Clustering Dendrogram
#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# if __name__ == "__main__":
#     main()







#works correctly with old dataset
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc

# nlp = spacy.load("en_core_web_sm")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)




# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract organizations from text
# def extract_organizations(text):
#     doc = nlp(text)
#     # organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
#     organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG","PRODUCT","WORK_OF_ART"]]
#     return organizations

# # Function to compare organizations mentioned in papers with model names
# # def compare_entities(papers_entities):
# #     entities_models = [[model for model in model_names if any(org.lower() == model.lower() for model in model_names)] for orgs in papers_entities]
# #     return entities_models


# # def compare_entities(papers_entities):
# #     entities_models = [[org for org in orgs if any(org.lower() == model.lower() for model in model_names)] for orgs in papers_entities]
# #     return entities_models

# def compare_entities(papers_entities):
#     entities_models = []
#     for orgs in papers_entities:
#         matched_models = []
#         for org in orgs:
#             for model in model_names:
#                 if model.lower() == org.lower():
#                     matched_models.append(model.lower())
#                     break  # Break once a match is found to avoid duplicates
#         # entities_models.append(matched_models)
#         entities_models.append(list(set(matched_models)))
#     return entities_models




# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to calculate similarity based on entities mentioned in papers using Jaccard similarity
# def calculate_entities_similarity(papers_entities):
#     jaccard_similarities = []
#     for i in range(len(papers_entities)):
#         for j in range(i + 1, len(papers_entities)):
#             set1 = set(papers_entities[i])
#             set2 = set(papers_entities[j])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_entities = np.mean(jaccard_similarities)
#     return mean_similarity_entities

# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def plot_dendrogram(similarity_matrix, method='ward'):
#     plt.figure(figsize=(10, 7))
#     dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Papers')
#     plt.ylabel('Normalized Distance')
#     plt.ylim((0, 1))  
#     plt.axhline(y=1, color='r', linestyle='--')  
#     st.pyplot(plt)

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and organizations from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             papers_entities = [extract_organizations(text) for text in papers_text]
            
#             # Print entities extracted from each paper under the ORG label
#             st.subheader("Entities Extracted from Each Paper (ORG label, Product, Work of art):")
#             for i, entities in enumerate(papers_entities):
#                 st.write(f"Paper {i+1}: {', '.join(entities)}")
                
#             # Compare organizations mentioned in papers with model names
#             papers_models = compare_entities(papers_entities)


#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_entities = calculate_entities_similarity(papers_models)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (models): {mean_similarity_entities:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             # Plot Hierarchical Clustering Dendrogram
#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# if __name__ == "__main__":
#     main()



#works
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc

# nlp = spacy.load("en_core_web_sm")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = [[model for model in model_names if model.lower() in text.lower()] for text in papers_text]
#     return papers_models

# # Function to calculate similarity based on models mentioned in papers using Jaccard similarity
# def calculate_model_similarity(papers_models):
#     jaccard_similarities = []
#     for i in range(len(papers_models)):
#         for j in range(i + 1, len(papers_models)):
#             set1 = set(papers_models[i])
#             set2 = set(papers_models[j])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_models = np.mean(jaccard_similarities)
#     return mean_similarity_models


# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def plot_dendrogram(similarity_matrix, method='ward'):
#     plt.figure(figsize=(10, 7))
#     dend = shc.dendrogram(shc.linkage(similarity_matrix, method=method), labels=range(len(similarity_matrix)), leaf_font_size=10)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Papers')
#     plt.ylabel('Normalized Distance')
#     plt.ylim((0, 1))  
#     plt.axhline(y=1, color='r', linestyle='--')  
#     st.pyplot(plt)

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
#             print(model_names_new)
            
#             # Compare models mentioned in papers
#             papers_models = compare_models(papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_models = calculate_model_similarity(papers_models)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

#             # Plot Hierarchical Clustering Dendrogram
#             st.subheader("Hierarchical Clustering Dendrogram")
#             plot_dendrogram(similarity_matrix_transformer)

# if __name__ == "__main__":
#     main()






#works for all 5 ways
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util


# nlp = spacy.load("en_core_web_sm")


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)


# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = [[model for model in model_names if model.lower() in text.lower()] for text in papers_text]
#     return papers_models

# # Function to calculate similarity based on models mentioned in papers using Jaccard similarity
# def calculate_model_similarity(papers_models):
#     jaccard_similarities = []
#     for i in range(len(papers_models)):
#         for j in range(i + 1, len(papers_models)):
#             set1 = set(papers_models[i])
#             set2 = set(papers_models[j])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_models = np.mean(jaccard_similarities)
#     return mean_similarity_models


# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
#             print(model_names_new)
            
#             # Compare models mentioned in papers
#             papers_models = compare_models(papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_models = calculate_model_similarity(papers_models)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

# if __name__ == "__main__":
#     main()








# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util


# nlp = spacy.load("en_core_web_sm")


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)


# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# # Function to extract text from PDF and extract keywords using NER
# def extract_text_and_keywords_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     keywords = []
#     for page in reader.pages:
#         page_text = page.extract_text()
#         text += page_text
#         doc = nlp(page_text)
#         for ent in doc.ents:
#             if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
#                 keywords.append(ent.text)
#     return text, keywords

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = [[model for model in model_names if model.lower() in text.lower()] for text in papers_text]
#     return papers_models

# # Function to calculate similarity based on models mentioned in papers using Jaccard similarity
# def calculate_model_similarity(papers_models):
#     jaccard_similarities = []
#     for i in range(len(papers_models)):
#         for j in range(i + 1, len(papers_models)):
#             set1 = set(papers_models[i])
#             set2 = set(papers_models[j])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_models = np.mean(jaccard_similarities)
#     return mean_similarity_models


# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and keywords from uploaded files
#             papers_data = [extract_text_and_keywords_from_pdf(file) for file in uploaded_files]
#             papers_text = [data[0] for data in papers_data]
#             all_keywords = [data[1] for data in papers_data]
            
#             # Display keywords
#             st.subheader("Keywords Extracted from Papers:")
#             for i, keywords in enumerate(all_keywords):
#                 st.write(f"Paper {i+1}:")
#                 for label in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
#                     label_keywords = [keyword for keyword in keywords if nlp(keyword).ents and nlp(keyword).ents[0].label_ == label]
#                     if label_keywords:
#                         st.write(f"{label}: {', '.join(label_keywords)}")
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
            
#             # Compare models mentioned in papers
#             papers_models = compare_models(papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_models = calculate_model_similarity(papers_models)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

# if __name__ == "__main__":
#     main()





#works for all 5 ways correctly
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util


# nlp = spacy.load("en_core_web_sm")


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)


# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)


# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = [[model for model in model_names if model.lower() in text.lower()] for text in papers_text]
#     return papers_models

# # Function to calculate similarity based on models mentioned in papers using Jaccard similarity
# def calculate_model_similarity(papers_models):
#     jaccard_similarities = []
#     for i in range(len(papers_models)):
#         for j in range(i + 1, len(papers_models)):
#             set1 = set(papers_models[i])
#             set2 = set(papers_models[j])
#             similarity = calculate_jaccard_similarity(set1, set2)
#             jaccard_similarities.append(similarity)
#     mean_similarity_models = np.mean(jaccard_similarities)
#     return mean_similarity_models


# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores


# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to calculate similarity using sentence-transformers
# def calculate_similarity_transformer(documents):
#     embeddings = sentence_model.encode(documents)
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
#             print(model_names_new)
            
#             # Compare models mentioned in papers
#             papers_models = compare_models(papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity using Jaccard similarity
#             mean_similarity_models = calculate_model_similarity(papers_models)
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = similarity_scores_tfidf[0, 1]
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = similarity_scores_lsa[0, 1]
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = similarity_matrix_doc2vec[0, 1]
           
#             # Calculate similarity using sentence-transformers
#             similarity_matrix_transformer = calculate_similarity_transformer(papers_text)
#             mean_similarity_transformer = similarity_matrix_transformer[0, 1]
            
#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_transformer:.2f}")

# if __name__ == "__main__":
#     main()







#works for 0 to 1 and all 5 ways 
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load trained Doc2Vec model
# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# # Load SentenceTransformer model
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to perform similarity analysis using LSA and cosine similarity
# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = []
#     for text in papers_text:
#         models = [model for model in model_names if model.lower() in text.lower()]
#         papers_models.append(models)
#     return papers_models


# def calculate_similarity_statistics(similarity_matrix):
#     return similarity_matrix[0, 1]



# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
#             print(model_names_new)
            
#             # Compare models mentioned in papers
#             papers_models = compare_models(papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
            
            
            
#             # Print models extracted from papers in the terminal
#             print("Models extracted from each paper:")
#             for i, models in enumerate(papers_models):
#                 print(f"Paper {i+1}: {', '.join(models)}")
            
#             # Calculate model-based similarity
#             mean_similarity_models = 0
#             if len(papers_models) == 2:
#                 set1 = set(papers_models[0])
#                 set2 = set(papers_models[1])
#                 intersection_size = len(set1.intersection(set2))
#                 union_size = len(set1.union(set2))
#                 mean_similarity_models = intersection_size / union_size if union_size > 0 else 0
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(papers_text)
#             mean_similarity_text_tfidf = calculate_similarity_statistics(similarity_scores_tfidf)
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(papers_text)
#             mean_similarity_text_lsa = calculate_similarity_statistics(similarity_scores_lsa)
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec = calculate_similarity_statistics(similarity_matrix_doc2vec)
           
#             # Calculate similarity using sentence-transformers
#             embeddings_sentence_transformers = sentence_model.encode(papers_text)
#             similarity_matrix_sentence_transformers = cosine_similarity(embeddings_sentence_transformers)
#             mean_similarity_sentence_transformers = calculate_similarity_statistics(similarity_matrix_sentence_transformers)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_sentence_transformers:.2f}")

# if __name__ == "__main__":
#     main()



#works with doc2vec, tfidf, lsa,models and transformers
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names
# from sentence_transformers import SentenceTransformer, util

# nlp = spacy.load("en_core_web_sm")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load trained Doc2Vec model
# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)

# # Load SentenceTransformer model
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to perform similarity analysis using LSA and cosine similarity
# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = []
#     for text in papers_text:
#         models = [model for model in model_names if model.lower() in text.lower()]
#         papers_models.append(models)
#     return papers_models

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
            
#             # Preprocess papers' text
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
           
#             # Compare models mentioned in papers
#             papers_models = compare_models(preprocessed_papers_text, model_names)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
            
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
            
#             # Print models extracted from papers in the terminal
#             print("Models extracted from each paper:")
#             for i, models in enumerate(papers_models):
#                 print(f"Paper {i+1}: {', '.join(models)}")
            
#             # Calculate model-based similarity
#             mean_similarity_models = 0
#             if len(papers_models) == 2:
#                 set1 = set(papers_models[0])
#                 set2 = set(papers_models[1])
#                 intersection_size = len(set1.intersection(set2))
#                 union_size = len(set1.union(set2))
#                 mean_similarity_models = intersection_size / union_size if union_size > 0 else 0
            
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(preprocessed_papers_text)
#             mean_similarity_text_tfidf, _, _ = calculate_similarity_statistics(similarity_scores_tfidf)
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(preprocessed_papers_text)
#             mean_similarity_text_lsa, _, _ = calculate_similarity_statistics(similarity_scores_lsa)
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(preprocessed_papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec, _, _ = calculate_similarity_statistics(similarity_matrix_doc2vec)
           
#             # Calculate similarity using sentence-transformers
#             embeddings_sentence_transformers = sentence_model.encode(preprocessed_papers_text)
#             similarity_matrix_sentence_transformers = cosine_similarity(embeddings_sentence_transformers)
#             mean_similarity_sentence_transformers, _, _ = calculate_similarity_statistics(similarity_matrix_sentence_transformers)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Sentence Transformers): {mean_similarity_sentence_transformers:.2f}")

# if __name__ == "__main__":
#     main()



#Works with tfidf lsa doc2vec and models
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st
# import pickle
# from model_names import model_names

# nlp = spacy.load("en_core_web_sm")


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load trained Doc2Vec model
# with open("doc2vec_model.pkl", "rb") as f:
#     doc2vec_model = pickle.load(f)


# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to perform similarity analysis using LSA and cosine similarity
# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to compute document embeddings using trained Doc2Vec model
# def compute_doc2vec_embeddings(documents):
#     embeddings = [doc2vec_model.infer_vector(word_tokenize(doc)) for doc in documents]
#     return embeddings

# # Function to calculate similarity using document embeddings and cosine similarity
# def calculate_similarity_doc2vec(embeddings):
#     similarity_matrix = cosine_similarity(embeddings)
#     return similarity_matrix

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to compare models mentioned in papers
# def compare_models(papers_text, model_names):
#     papers_models = [[model for model in model_names if model in text] for text in papers_text]
#     return papers_models

# # Function to calculate similarity based on models mentioned in papers
# def calculate_model_similarity(papers_models):
#     jaccard_similarities = []
#     for i in range(len(papers_models)):
#         for j in range(i + 1, len(papers_models)):
#             similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#             jaccard_similarities.append(similarity)
#     mean_similarity_models = np.mean(jaccard_similarities)
#     return mean_similarity_models

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity


# def main():
#     st.title("Research Paper Similarity Analyzer")
   
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
           
           
#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(preprocessed_papers_text)
#             mean_similarity_text_tfidf, _, _ = calculate_similarity_statistics(similarity_scores_tfidf)
           
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(preprocessed_papers_text)
#             mean_similarity_text_lsa, _, _ = calculate_similarity_statistics(similarity_scores_lsa)
           
#             # Compute document embeddings using Doc2Vec
#             embeddings = compute_doc2vec_embeddings(preprocessed_papers_text)
           
#             # Perform similarity analysis based on document embeddings using Doc2Vec
#             similarity_matrix_doc2vec = calculate_similarity_doc2vec(embeddings)
#             mean_similarity_doc2vec, _, _ = calculate_similarity_statistics(similarity_matrix_doc2vec)
           
#             # Predefined list of models to compare
#             # model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "generative adversarial network", "clothes warping module, pix2pixHD"]
           
#             # Convert all items in model_names to lowercase and only keep unique values
#             model_names_new = list(set([model.lower() for model in model_names]))
#             print(model_names_new)
           
#             # Compare models mentioned in papers
#             papers_models = compare_models(preprocessed_papers_text, model_names_new)
            
#             # Print model names for each paper
#             st.subheader("Models Extracted from Each Paper:")
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}") 
           
#             # Calculate model-based similarity
#             mean_similarity_models = calculate_model_similarity(papers_models)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, Doc2Vec): {mean_similarity_doc2vec:.2f}")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()


# #LSA (Works)
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using LSA and cosine similarity
# def analyze_similarity_lsa(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     svd = TruncatedSVD(n_components=100, random_state=42)
#     lsa_matrix = svd.fit_transform(tfidf_matrix)
#     similarity_scores = cosine_similarity(lsa_matrix)
#     return similarity_scores

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity_tfidf(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     # Predefined list of models to compare
#     model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "generative adversarial network", "clothes warping module, pix2pixHD"]

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
            
#             # Perform similarity analysis based on text using LSA
#             similarity_scores_lsa = analyze_similarity_lsa(preprocessed_papers_text)
#             mean_similarity_text_lsa, _, _ = calculate_similarity_statistics(similarity_scores_lsa)

#             # Perform similarity analysis based on text using TF-IDF
#             similarity_scores_tfidf = analyze_similarity_tfidf(preprocessed_papers_text)
#             mean_similarity_text_tfidf, _, _ = calculate_similarity_statistics(similarity_scores_tfidf)

#             # Extract models mentioned in each paper
#             papers_models = [[model for model in model_names if model in text] for text in preprocessed_papers_text]
            
#             # Debugging: Print out extracted models
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}")

#             # Calculate Jaccard similarity between models mentioned in each pair of papers
#             jaccard_similarities = []
#             for i in range(len(uploaded_files)):
#                 for j in range(i + 1, len(uploaded_files)):
#                     similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                     jaccard_similarities.append(similarity)

#             # Calculate mean similarity based on models
#             mean_similarity_models = np.mean(jaccard_similarities)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Text, LSA): {mean_similarity_text_lsa:.2f}")
#             st.write(f"Mean Similarity (Text, TF-IDF): {mean_similarity_text_tfidf:.2f}")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()






# #Works
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     # Predefined list of models to compare
#     model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "generative adversarial network", "clothes warping module, pix2pixHD"]

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
            
#             # Perform similarity analysis based on text
#             similarity_scores = analyze_similarity(preprocessed_papers_text)
#             mean_similarity_text, _, _ = calculate_similarity_statistics(similarity_scores)

#             # Extract models mentioned in each paper
#             papers_models = [[model for model in model_names if model in text] for text in preprocessed_papers_text]
            
#             # Debugging: Print out extracted models
#             for i, models in enumerate(papers_models):
#                 st.write(f"Paper {i+1}:")
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}")
#                 # st.write(f"Preprocessed text: {preprocessed_papers_text}")

#             # Calculate Jaccard similarity between models mentioned in each pair of papers
#             jaccard_similarities = []
#             for i in range(len(uploaded_files)):
#                 for j in range(i + 1, len(uploaded_files)):
#                     similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                     jaccard_similarities.append(similarity)

#             # Calculate mean similarity based on models
#             mean_similarity_models = np.mean(jaccard_similarities)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Text): {mean_similarity_text:.2f}")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()






# works with preprocessed text display
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     processed_text = ' '.join(tokens)
#     return processed_text, tokens

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     # Predefined list of models to compare
#     model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "generative adversarial network", "clothes warping module, pix2pixHD"]

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
            
#             # Perform similarity analysis based on text
#             similarity_scores = analyze_similarity([t[0] for t in preprocessed_papers_text])
#             mean_similarity_text, _, _ = calculate_similarity_statistics(similarity_scores)

#             # Extract models mentioned in each paper
#             papers_models = [[model for model in model_names if model in text[0]] for text in preprocessed_papers_text]
            
#             # Debugging: Print out extracted models
#             for i, models in enumerate(papers_models):
#                 st.write(f"Models extracted from Paper {i+1}: {', '.join(models)}")
#                 st.write(f"Preprocessed text from Paper {i+1}: {preprocessed_papers_text[i][0]}")  # Display preprocessed text

#             # Calculate Jaccard similarity between models mentioned in each pair of papers
#             jaccard_similarities = []
#             for i in range(len(uploaded_files)):
#                 for j in range(i + 1, len(uploaded_files)):
#                     similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                     jaccard_similarities.append(similarity)

#             # Calculate mean similarity based on models
#             mean_similarity_models = np.mean(jaccard_similarities)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Text): {mean_similarity_text:.2f}")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()







# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     # Predefined list of models to compare
#     model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "Generative Adversarial Networks", "clothes warping module, pix2pixHD"]

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
            
#             # Perform similarity analysis based on text
#             similarity_scores = analyze_similarity(preprocessed_papers_text)
#             mean_similarity_text, _, _ = calculate_similarity_statistics(similarity_scores)

#             # Extract models mentioned in each paper
#             papers_models = [[model for model in model_names if model in text] for text in preprocessed_papers_text]

#             # Calculate Jaccard similarity between models mentioned in each pair of papers
#             jaccard_similarities = []
#             for i in range(len(uploaded_files)):
#                 for j in range(i + 1, len(uploaded_files)):
#                     similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                     jaccard_similarities.append(similarity)

#             # Calculate mean similarity based on models
#             mean_similarity_models = np.mean(jaccard_similarities)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Text): {mean_similarity_text:.2f}")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()



# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract models mentioned in the text
# def extract_models(text):
#     doc = nlp(text)
#     models = [ent.text for ent in doc.ents if ent.label_ == "MODEL"]
#     return models

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         if len(uploaded_files) == 1:
#             st.sidebar.warning("Upload at least two papers to perform comparison.")
#         else:
#             # Extract text and preprocess from uploaded files
#             papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#             preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
            
#             # Extract models mentioned in each paper
#             papers_models = [extract_models(text) for text in preprocessed_papers_text]

#             # Calculate Jaccard similarity between models mentioned in each pair of papers
#             jaccard_similarities = []
#             for i in range(len(uploaded_files)):
#                 for j in range(i + 1, len(uploaded_files)):
#                     similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                     jaccard_similarities.append(similarity)

#             # Calculate mean similarity based on models
#             mean_similarity_models = np.mean(jaccard_similarities)

#             # Perform similarity analysis based on text content
#             similarity_scores = analyze_similarity(preprocessed_papers_text)
#             mean_similarity_text, _, _ = calculate_similarity_statistics(similarity_scores)

#             # Display results
#             st.subheader("Analysis Results")
#             st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")
#             st.write(f"Mean Similarity (Text): {mean_similarity_text:.2f}")

# if __name__ == "__main__":
#     main()





#model comparison works
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract models mentioned in the text
# def extract_models(text):
#     doc = nlp(text)
#     models = [ent.text for ent in doc.ents if ent.label_ == "MODEL"]
#     return models

# # Function to calculate Jaccard similarity
# def calculate_jaccard_similarity(set1, set2):
#     intersection_size = len(set1.intersection(set2))
#     union_size = len(set1.union(set2))
#     similarity = intersection_size / union_size if union_size > 0 else 0
#     return similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files and len(uploaded_files) > 1:
#         # Extract text and preprocess from uploaded files
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         # Extract models mentioned in each paper
#         papers_models = [extract_models(text) for text in preprocessed_papers_text]

#         # Calculate Jaccard similarity between models mentioned in each pair of papers
#         jaccard_similarities = []
#         for i in range(len(uploaded_files)):
#             for j in range(i + 1, len(uploaded_files)):
#                 similarity = calculate_jaccard_similarity(set(papers_models[i]), set(papers_models[j]))
#                 jaccard_similarities.append(similarity)

#         # Calculate mean similarity based on models
#         mean_similarity_models = np.mean(jaccard_similarities)

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

# if __name__ == "__main__":
#     main()


# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
# import numpy as np
# import streamlit as st


# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# def compare_models(model_names_1, model_names_2):
#     # Find common models mentioned in both papers
#     common_models = set(model_names_1).intersection(set(model_names_2))
    
#     # Calculate Jaccard similarity
#     intersection_size = len(common_models)
#     union_size = len(set(model_names_1).union(set(model_names_2)))
#     similarity = intersection_size / union_size if union_size > 0 else 0
    
#     return similarity

# # def compare_models(model_names):
# #     similarity_matrix = []
# #     for model_name1 in model_names:
# #         similarities = []
# #         for model_name2 in model_names:
# #             # Calculate Jaccard similarity between model names
# #             intersection = len(set(model_name1.split()) & set(model_name2.split()))
# #             union = len(set(model_name1.split()) | set(model_name2.split()))
# #             similarity = intersection / union
# #             similarities.append(similarity)
# #         similarity_matrix.append(similarities)
# #     return similarity_matrix



# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# # Function to calculate similarity statistics
# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         # Extract text and preprocess from uploaded files
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         # Perform similarity analysis
#         similarity_scores = analyze_similarity(preprocessed_papers_text)
#         mean_similarity, _, _ = calculate_similarity_statistics(similarity_scores)

#         # Extract domains from uploaded papers
#         extracted_domains = [extract_domains(text) for text in papers_text]

#         # Compare models used in papers
#         model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "Generative Adversarial Networks", "semantic generation module", "clothes warping module, pix2pixHD"]
#         similarity_matrix = compare_models(model_names, model_names)
#         mean_similarity_models, _, _ = calculate_similarity_statistics(similarity_matrix)

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity (Text): {mean_similarity:.2f}")
#         st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

#         # Display extracted domains
#         st.subheader("Extracted Domains:")
#         for i, domains in enumerate(extracted_domains):
#             st.write(f"Domains in Paper {i+1}: {', '.join(domains)}")

# if __name__ == "__main__":
#     main()




# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import streamlit as st

# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         # Extract text and preprocess from uploaded files
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         # Perform similarity analysis
#         similarity_scores = analyze_similarity(preprocessed_papers_text)
#         mean_similarity, _, _ = calculate_similarity_statistics(similarity_scores)

#         # Extract domains from uploaded papers
#         extracted_domains = [extract_domains(text) for text in papers_text]

#         # Compare models used in papers
#         model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "deep learning", "augmented reality", "generative adversarial networks", "semantic generation module", "clothes warping module, pix2pixHD"]
#         st.subheader("Mean Similarity Based on Models")
#         for i, model_name in enumerate(model_names):
#             st.write(f"Model: {model_name}")
#             similarity_scores = analyze_similarity([preprocess_and_join(model_name)] + preprocessed_papers_text)
#             mean_similarity_models, _, _ = calculate_similarity_statistics(similarity_scores[1:])
#             st.write(f"Mean Similarity: {mean_similarity_models:.2f}")

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity (Text): {mean_similarity:.2f}")

#         # Display extracted domains
#         st.subheader("Extracted Domains:")
#         for i, domains in enumerate(extracted_domains):
#             st.write(f"Domains in Paper {i+1}: {', '.join(domains)}")

# if __name__ == "__main__":
#     main()


#works with 0.13
# import PyPDF2
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
# import numpy as np
# import streamlit as st

# # Load English language model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Function to preprocess text
# def preprocess_and_join(text):
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and lowercase
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if not word in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     # Join tokens into a single string
#     return ' '.join(tokens)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to extract domains using spaCy
# def extract_domains(text):
#     doc = nlp(text)
#     domains = [ent.text for ent in doc.ents if ent.label_ == "DOMAIN"]
#     return domains

# # Function to calculate similarity between models
# def compare_models(model_names):
#     model = Word2Vec(sentences=[model_names], vector_size=100, window=5, min_count=1, workers=4)
#     similarity_matrix = []
#     for model_name1 in model_names:
#         similarities = []
#         for model_name2 in model_names:
#             similarity = model.wv.similarity(model_name1, model_name2)
#             similarities.append(similarity)
#         similarity_matrix.append(similarities)
#     return similarity_matrix



# # Function to perform similarity analysis using TF-IDF and cosine similarity
# def analyze_similarity(documents):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#     similarity_scores = cosine_similarity(tfidf_matrix)
#     return similarity_scores

# def calculate_similarity_statistics(similarity_matrix):
#     similarity_values = np.array(similarity_matrix).flatten()
#     mean_similarity = np.mean(similarity_values)
#     max_similarity = np.max(similarity_values)
#     min_similarity = np.min(similarity_values)
#     return mean_similarity, max_similarity, min_similarity

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         # Extract text and preprocess from uploaded files
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         # Perform similarity analysis
#         similarity_scores = analyze_similarity(preprocessed_papers_text)
#         mean_similarity, _, _ = calculate_similarity_statistics(similarity_scores)

#         # Extract domains from uploaded papers
#         extracted_domains = [extract_domains(text) for text in papers_text]

#         # Compare models used in papers
#         model_names = ["spacy", "part-of-speech tagging", "stemming", "lemmatization", "spacy", "deep learning", "augmented reality", "Generative Adversarial Networks", "semantic generation module", "clothes warping module, pix2pixHD"]
#         similarity_matrix = compare_models(model_names)
#         mean_similarity_models, _, _ = calculate_similarity_statistics(similarity_matrix)

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity (Text): {mean_similarity:.2f}")
#         st.write(f"Mean Similarity (Models): {mean_similarity_models:.2f}")

#         # Display extracted domains
#         st.subheader("Extracted Domains:")
#         for i, domains in enumerate(extracted_domains):
#             st.write(f"Domains in Paper {i+1}: {', '.join(domains)}")

# if __name__ == "__main__":
#     main()







# import streamlit as st
# from file1 import calculate_similarity_statistics, analyze_similarity, extract_domains, extract_text_from_pdf, preprocess_and_join

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         # Extract text and preprocess from uploaded files
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         # Perform similarity analysis
#         similarity_scores = analyze_similarity(preprocessed_papers_text)
#         mean_similarity, _, _ = calculate_similarity_statistics(similarity_scores)

#         # Extract domains from uploaded papers
#         extracted_domains = [extract_domains(text) for text in papers_text]

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity: {mean_similarity:.2f}")

#         # Display extracted domains
#         st.subheader("Extracted Domains:")
#         for i, domains in enumerate(extracted_domains):
#             st.write(f"Domains in Paper {i+1}: {', '.join(domains)}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from file1 import calculate_similarity_statistics, analyze_similarity, extract_domains, extract_text_from_pdf, preprocess_and_join

# # Define Streamlit app content
# def main():
#     st.title("Research Paper Similarity Analyzer")
    
#     st.sidebar.header("Upload Papers")
#     uploaded_files = st.sidebar.file_uploader("Upload your research papers (PDF format)", accept_multiple_files=True)

#     if uploaded_files:
#         papers_text = [extract_text_from_pdf(file) for file in uploaded_files]
#         preprocessed_papers_text = [preprocess_and_join(text) for text in papers_text]
        
#         similarity_scores = analyze_similarity(preprocessed_papers_text)
#         mean_similarity, _, _ = calculate_similarity_statistics(similarity_scores)

#         extracted_domains = [extract_domains(text) for text in papers_text]

#         # Display results
#         st.subheader("Analysis Results")
#         st.write(f"Mean Similarity: {mean_similarity:.2f}")

#         # Display extracted domains
#         st.subheader("Extracted Domains:")
#         for i, domains in enumerate(extracted_domains):
#             st.write(f"Domains in Paper {i+1}: {', '.join(domains)}")

# if __name__ == "__main__":
#     main()
