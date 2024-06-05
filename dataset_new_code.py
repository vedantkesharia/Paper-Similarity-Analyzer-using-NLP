import pandas as pd
import random
from tqdm import tqdm


# models = [
#     "GAN", "CNN", "RNN", "SVM", "Random Forest", "K-means", "PCA", "LSTM", "Transformer",
#     "Decision Tree", "Gradient Boosting", "Autoencoder", "Bayesian Network", "Markov Chain",
#     "Deep Belief Network", "Self-Organizing Map", "Hidden Markov Model", "Reinforcement Learning",
#     "Transfer Learning", "Meta-Learning", "Few-Shot Learning", "Zero-Shot Learning",
#     "Semi-Supervised Learning", "Active Learning", "Multi-Task Learning", "Ensemble Learning"
# ]

# domains = [
#     "Machine Learning", "NLP", "Computer Vision", "Sequence Prediction", "Classification", "Regression",
#     "Clustering", "Dimensionality Reduction", "Supervised Learning", "Unsupervised Learning",
#     "Probabilistic Graphical Models", "Stochastic Processes", "Generative Models", "Visualization",
#     "Temporal Pattern Recognition", "Decision Making", "Transfer Learning", "Meta-Learning",
#     "Few-Shot Learning", "Zero-Shot Learning", "Semi-Supervised Learning", "Active Learning",
#     "Multi-Task Learning", "Ensemble Learning", "Feature Selection", "Hyperparameter Tuning"
# ]

# libraries = [
#     "spacy", "nltk", "tensorflow", "pytorch", "scikit-learn", "keras", "gensim", "transformers",
#     "xgboost", "lightgbm", "catboost", "fastai", "mxnet", "chainer", "theano", "cntk", "caffe",
#     "torch", "dlib", "opencv", "nltk", "spacy", "stanfordnlp", "flair", "allennlp", "fairseq"
# ]


models = [
    "GAN", "CNN", "RNN", "SVM", "Random Forest", "K-means", "PCA", "LSTM", "Transformer",
    "Decision Tree", "Gradient Boosting", "Autoencoder", "Bayesian Network", "Markov Chain",
    "Deep Belief Network", "Self-Organizing Map", "Hidden Markov Model", "Reinforcement Learning",
    "Transfer Learning", "Meta-Learning", "Few-Shot Learning", "Zero-Shot Learning",
    "Semi-Supervised Learning", "Active Learning", "Multi-Task Learning", "Ensemble Learning",
    "Linear Regression", "Logistic Regression", "Naive Bayes", "K-Nearest Neighbors (KNN)",
    "Gradient Boosted Trees", "AdaBoost", "Neural Networks", "Extreme Learning Machines (ELM)",
    "Variational Autoencoder (VAE)", "Capsule Networks", "Graph Neural Networks (GNN)",
    "Conditional Random Fields (CRF)", "Restricted Boltzmann Machines (RBM)",
    "Stochastic Gradient Descent (SGD)", "Evolutionary Algorithms", "Genetic Algorithms",
    "Self-Attention Networks", "Temporal Convolutional Networks (TCN)", "Siamese Networks",
    "Neural Architecture Search (NAS)", "Deep Q-Networks (DQN)", "Monte Carlo Methods",
    "Bayesian Optimization", "Swarm Intelligence", "Decision Forests", "Echo State Networks",
    "Modular Neural Networks", "Growing Neural Gas (GNG)", "Hierarchical Temporal Memory (HTM)",
    "Wavelet Neural Networks", "Radial Basis Function Networks (RBFN)", "Extreme Gradient Boosting (XGBoost)",
    "Q-Learning", "Actor-Critic Methods", "Value Iteration Networks (VIN)", "Generative Moment Matching Networks (GMMN)",
    "Generative Latent Optimization (GLO)", "Generative Stochastic Networks (GSN)"
]


domains = [
    "Machine Learning", "NLP", "Computer Vision", "Sequence Prediction", "Classification", "Regression",
    "Clustering", "Dimensionality Reduction", "Supervised Learning", "Unsupervised Learning",
    "Probabilistic Graphical Models", "Stochastic Processes", "Generative Models", "Visualization",
    "Temporal Pattern Recognition", "Decision Making", "Transfer Learning", "Meta-Learning",
    "Few-Shot Learning", "Zero-Shot Learning", "Semi-Supervised Learning", "Active Learning",
    "Multi-Task Learning", "Ensemble Learning", "Feature Selection", "Hyperparameter Tuning",
    "Time Series Analysis", "Anomaly Detection", "Reinforcement Learning", "Speech Recognition",
    "Image Segmentation", "Object Detection", "Text Generation", "Machine Translation",
    "Sentiment Analysis", "Topic Modeling", "Information Retrieval", "Recommender Systems",
    "Robotics", "Bioinformatics", "Fraud Detection", "Autonomous Systems", "Smart Grids",
    "Internet of Things (IoT)", "Edge Computing", "Cloud Computing", "Quantum Machine Learning",
    "Ethical AI", "Explainable AI (XAI)", "Fairness in AI", "Healthcare Informatics",
    "Financial Analytics", "Predictive Maintenance", "Supply Chain Optimization",
    "Customer Relationship Management (CRM)", "Cybersecurity", "Remote Sensing",
    "Geographical Information Systems (GIS)", "Social Network Analysis", "E-commerce",
    "Education Technology (EdTech)", "Agricultural Technology (AgriTech)", "Media & Entertainment",
    "Retail Analytics", "Energy Forecasting", "Urban Planning", "Climate Modeling",
    "Disaster Response & Recovery"
]


libraries = [
    "spacy", "nltk", "tensorflow", "pytorch", "scikit-learn", "keras", "gensim", "transformers",
    "xgboost", "lightgbm", "catboost", "fastai", "mxnet", "chainer", "theano", "cntk", "caffe",
    "torch", "dlib", "opencv", "stanfordnlp", "flair", "allennlp", "fairseq", "turi create",
    "pycaret", "ludwig", "tpot", "h2o.ai", "dask-ml", "rapids.ai", "bigdl", "deeplearning4j",
    "horovod", "opennmt", "scipy", "numpy", "pandas", "matplotlib", "seaborn", "plotly",
    "bokeh", "dash", "streamlit", "scrapy", "beautifulsoup", "selenium", "pattern",
    "gensim", "keras-tuner", "optuna", "ray", "dask", "joblib", "mlflow", "wandb",
    "tensorboard", "pytorch-lightning", "fasttext", "word2vec", "glove", "bert", "gpt",
    "xlnet", "roberta", "albert", "electra", "bart", "t5", "gpt-3", "imageio", "PIL",
    "scikit-image", "imgaug", "tensorflow-io", "pydicom", "vtk", "mayavi", "plotly",
    "folium", "geopandas", "shapely", "cartopy", "osmnx", "networkx", "igraph", "graph-tool",
    "geopy", "h3", "datetime", "arrow", "pytz", "dateutil", "numpy", "scipy", "pandas", "statsmodels",
    "pyflux", "prophet", "gluonts", "tsfresh", "tslearn", "sktime", "fbprophet", "pmdarima",
    "arviz", "deeppavlov", "rasa", "stanza", "gensim", "textblob", "vaderSentiment",
    "transformers", "spacy-transformers", "huggingface", "rasa-nlu", "rasa-core",
    "rasa-sdk", "rasa-x", "openai-gpt", "openai-gpt-2", "openai-gpt-3"
]


# Function to generate a random sentence with annotations
def generate_sentence(model, domain, library):
    return 'The ' + model + ' model is used in the ' + domain + ' domain with the ' + library + ' library.'

# Generate the dataset
num_sentences = 3000
sentences = set()  # Use a set to ensure uniqueness

while len(sentences) < num_sentences:
    model = random.choice(models)
    domain = random.choice(domains)
    library = random.choice(libraries)
    sentence = generate_sentence(model, domain, library)
    sentences.add((sentence, model, domain, library))

# Convert the set to a list of dictionaries
sentences_list = [{'sentence': s[0], 'model': s[1], 'domain': s[2], 'library': s[3]} for s in sentences]

# Create a DataFrame
sentences_df = pd.DataFrame(sentences_list)

# Display the head of the DataFrame
print(sentences_df.head())

# Save the DataFrame to a CSV file
sentences_df.to_csv('unique_annotated_sentences.csv', index=False)