#main_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class QuestionEmbeddingLayer(nn.Module):
    """
    Layer to convert question text into embeddings
    Can use either TF-IDF + linear projection or BERT embeddings
    """
    def __init__(self, num_features, embedding_dim=128, use_bert=False):
        super(QuestionEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_bert = use_bert
        
        if use_bert:
            # Use pre-trained BERT for embeddings
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # Projection from BERT's hidden size to our embedding size
            self.projection = nn.Linear(768, embedding_dim)
        else:
            # Simple linear projection from TF-IDF features
            self.projection = nn.Linear(num_features, embedding_dim)
    
    def forward(self, questions, tfidf_matrix=None):
        if self.use_bert:
            # Process with BERT
            encoded_input = self.tokenizer(questions, padding=True, truncation=True, 
                                          max_length=128, return_tensors='pt')
            outputs = self.bert(**encoded_input)
            # Use [CLS] token representation (first token)
            cls_outputs = outputs.last_hidden_state[:, 0, :]
            return self.projection(cls_outputs)
        else:
            # Project TF-IDF features
            return self.projection(tfidf_matrix)


class MultiLayerAttention(nn.Module):
    """
    Multiple stacked self-attention layers for question-question interaction
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Apply each attention layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x


# class TopicAwareLayer(nn.Module):
#     """
#     Layer that learns to identify and weight topics within questions
#     """
#     def __init__(self, embed_dim, num_topics=7):
#         super(TopicAwareLayer, self).__init__()
#         self.num_topics = num_topics
        
#         # Topic attention mechanism
#         self.topic_query = nn.Parameter(torch.randn(num_topics, embed_dim))
#         self.topic_attention = nn.Linear(embed_dim, num_topics)
        
#         # Topic importance weighting
#         self.topic_importance = nn.Parameter(torch.ones(num_topics, 1))
        
#         # Output projection
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
    
#     def forward(self, x):
#         # Calculate attention scores for each topic
#         topic_scores = self.topic_attention(x)  # [batch, seq_len, num_topics]
#         topic_weights = F.softmax(topic_scores, dim=-1)  # [batch, seq_len, num_topics]
        
#         # Weight embeddings by topic importance
#         weighted_importance = topic_weights @ self.topic_importance  # [batch, seq_len, 1]
        
#         # Apply topic-aware weighting
#         topic_enhanced = x * weighted_importance
        
#         # Project back to embedding space
#         return self.output_proj(topic_enhanced)


class QuestionQuestionTransformer(nn.Module):
    """
    Transformer architecture to model relationships between questions
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super(QuestionQuestionTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
    
    def forward(self, x, mask=None):
        return self.transformer_encoder(x, mask=mask)


class EnhancedProbabilityModel(nn.Module):
    """
    Complete model with all enhanced neural network components
    """
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super(EnhancedProbabilityModel, self).__init__()
        self.embed_dim = embed_dim
        
        # Positional encoding for questions
        self.register_buffer('positional_encoding', self._create_positional_encoding(1000, embed_dim))
        
        # Embedding layer projects to embedding dimension
        self.embedding_projection = nn.Linear(1, embed_dim)
        
        # Multi-layer attention mechanism
        self.multi_attention = MultiLayerAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Remove the Topic-aware layer
        # self.topic_layer = TopicAwareLayer(
        #     embed_dim=embed_dim,
        #     num_topics=num_topics
        # )
        
        # Question-question transformer
        self.transformer = QuestionQuestionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Final prediction layers
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()  # For 0-1 output (will scale to 0-100%)
        )
    def _create_positional_encoding(self, max_len, embed_dim):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, base_scores, similarity_matrix, question_embeddings=None):
        batch_size, seq_len = base_scores.shape
        
        # Project base scores to embedding dimension
        x = self.embedding_projection(base_scores.unsqueeze(-1))
        
        # Add positional encoding
        positions = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + positions
        
        # Apply multi-layer attention
        x = self.multi_attention(x)
        
        # Remove topic-aware layer application
        # x = self.topic_layer(x)
        
        # Apply question-question transformer
        if question_embeddings is not None:
            # If we have question text embeddings, add them here
            x = x + question_embeddings
        x = self.transformer(x)
        
        # Final prediction
        scores = self.final_projection(x).squeeze(-1)
        
        return scores * 100  # Scale to 0-100%

class EnhancedQuestionProbabilityModel:
    def __init__(self, historical_questions, appearance_counts, recency_weights=None):
        """
        Initialize the enhanced question probability prediction model.
        
        Parameters:
        - historical_questions: List of question texts
        - appearance_counts: Number of times each question appeared in past papers
        - recency_weights: List of weights based on how recently questions appeared (optional)
        """
        self.questions = historical_questions
        self.appearance_counts = np.array(appearance_counts)
        self.num_questions = len(historical_questions)
        
        # Default recency weights if not provided
        if recency_weights is None:
            self.recency_weights = np.ones_like(self.appearance_counts)
        else:
            self.recency_weights = np.array(recency_weights)
            
        # Remove topic_weights attribute
        # self.topic_weights = topic_weights
        
        # Text processing
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        self.similarity_matrix = cosine_similarity(self.question_vectors)
        
        # Get the actual number of features from the vectorizer
        num_features = self.question_vectors.shape[1]
        
        # Create embedding layer with correct number of features
        self.embedding_layer = QuestionEmbeddingLayer(num_features, embedding_dim=128, use_bert=False)
        
        # Create enhanced neural model
        self.nn_model = EnhancedProbabilityModel(
            embed_dim=128,
            # Remove num_topics parameter
            # num_topics=len(topic_weights) if topic_weights else 7,
            num_heads=4,
            num_layers=2
        )
        
        # Maximum possible score to scale probabilities against
        self.max_appearance_count = max(self.appearance_counts) if len(self.appearance_counts) > 0 else 1
        self.max_recency_weight = max(self.recency_weights) if len(self.recency_weights) > 0 else 1
        # Remove topic weight max
        # self.max_topic_weight = max(topic_weights.values()) if topic_weights else 1
        
        # Set model to evaluation mode by default
        self.nn_model.eval()
        
    def calculate_base_scores(self):
        """Calculate initial scores based on appearance counts"""
        # Scale by max appearance to get a 0-1 range
        return self.appearance_counts / self.max_appearance_count
    
    def apply_recency_adjustment(self, base_scores):
        """Apply recency weights to base scores"""
        # Apply recency and scale against maximum possible recency impact
        return base_scores * (self.recency_weights / self.max_recency_weight)
    
    # def apply_topic_adjustment(self, scores, question_topics):
    #     """Apply topic importance weights"""
    #     if self.topic_weights is None:
    #         return scores
            
    #     topic_multipliers = np.array([self.topic_weights.get(topic, 1.0) for topic in question_topics])
    #     # Scale by maximum topic weight
    #     return scores * (topic_multipliers / self.max_topic_weight)
    
    def process_through_nn(self, scores):
        """Process scores through the neural network model"""
        # Convert to torch tensors
        scores_tensor = torch.FloatTensor(scores).unsqueeze(0)  # Add batch dimension
        sim_matrix_tensor = torch.FloatTensor(self.similarity_matrix)
        
        # Create TF-IDF embeddings projection
        tfidf_dense = torch.FloatTensor(self.question_vectors.toarray()).unsqueeze(0)
        embeddings = self.embedding_layer(None, tfidf_dense)
        
        # Process through neural network
        with torch.no_grad():  # No gradient tracking needed for inference
            enhanced_scores = self.nn_model(scores_tensor, sim_matrix_tensor, embeddings)
        
        # Convert back to numpy
        return enhanced_scores.squeeze(0).numpy()
    
    def predict_probabilities(self, new_question=None):
        """
        Predict the probability of each question appearing in the next exam.
        Each probability is independent and ranges from 0% to 100%.
        
        Parameters:
        - new_question: Text of a new question to evaluate (optional)
        
        Returns:
        - Dictionary mapping questions to their independent probabilities
        """
        # Calculate base scores from historical data
        base_scores = self.calculate_base_scores()
        
        # Apply recency adjustment
        recency_adjusted = self.apply_recency_adjustment(base_scores)
        
        # Remove topic adjustment
        # if question_topics:
        #     topic_adjusted = self.apply_topic_adjustment(recency_adjusted, question_topics)
        # else:
        #     topic_adjusted = recency_adjusted
        
        # Use recency_adjusted directly instead of topic_adjusted
        final_scores = self.process_through_nn(recency_adjusted)
        
        # If a new question is provided, calculate its probability
        if new_question:
            new_vector = self.vectorizer.transform([new_question])
            similarities = cosine_similarity(new_vector, self.question_vectors)[0]
            
            # Calculate probability based on similarities to existing questions
            similarity_score = np.max(similarities)  # Use max similarity
            
            # Find the most similar question
            most_similar_idx = np.argmax(similarities)
            similar_question_score = final_scores[most_similar_idx]
            
            # Combine similarity and the score of the most similar question
            new_score = similarity_score * similar_question_score / 100
            
            # Return percentage (capped at 100%)
            return min(new_score * 100, 100)
        
        # Cap scores at 100%
        percentages = np.minimum(final_scores, 100)
        result = {q: f"{p:.2f}%" for q, p in zip(self.questions, percentages)}
        
        return result
        
    def train(self, training_data, epochs=10, learning_rate=0.001):
        """
        Train the neural network model with labeled data.
        
        Parameters:
        - training_data: list of (question_idx, label) pairs, where label is 1 if 
                         the question appeared in an exam, 0 otherwise
        - epochs: Number of training epochs
        - learning_rate: Learning rate for optimizer
        """
        # Prepare optimizer
        optimizer = Adam(self.nn_model.parameters(), lr=learning_rate)
        
        # Set model to training mode
        self.nn_model.train()
        
        # Calculate base features
        base_scores = self.calculate_base_scores()
        recency_adjusted = self.apply_recency_adjustment(base_scores)
        
        # Prepare input tensors
        scores_tensor = torch.FloatTensor(recency_adjusted).unsqueeze(0)
        sim_matrix_tensor = torch.FloatTensor(self.similarity_matrix)
        tfidf_dense = torch.FloatTensor(self.question_vectors.toarray()).unsqueeze(0)
        embeddings = self.embedding_layer(None, tfidf_dense)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            # Process batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.nn_model(scores_tensor, sim_matrix_tensor, embeddings)
            
            # Calculate loss
            targets = torch.zeros_like(outputs)
            for idx, label in training_data:
                targets[0, idx] = label * 100  # Scale to 0-100%
            
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass and optimize
            loss.backward(retain_graph=True if epoch < epochs-1 else False)
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Set model back to evaluation mode
        self.nn_model.eval()
        print("Training completed")
    def evaluate_model(self, test_data, threshold=50.0):
            """
            Evaluate the model's accuracy on test data
            
            Parameters:
            - test_data: List of (question_idx, actual_label) pairs
            - threshold: Probability threshold above which we predict a question will appear
            
            Returns:
            - Dictionary with various accuracy metrics
            """
            # Get predictions for all questions
            probabilities = self.predict_probabilities()
            
            # Convert probabilities to binary predictions
            predictions = []
            actual_labels_binary = []
            
            for question_idx, actual_label in test_data:
                if question_idx < len(self.questions):
                    question = self.questions[question_idx]
                    prob_str = probabilities[question]
                    prob_value = float(prob_str.strip('%'))
                    
                    # Convert probability to binary prediction using threshold
                    predicted_label = 1 if prob_value >= threshold else 0
                    
                    # Convert actual label to binary (1 if question appeared, 0 if not)
                    # Any positive value means the question appeared
                    actual_binary = 1 if actual_label > 0 else 0
                    
                    predictions.append(predicted_label)
                    actual_labels_binary.append(actual_binary)
            
            # Ensure we have at least some positive and negative examples
            if len(set(actual_labels_binary)) < 2:
                print("Warning: All test samples have the same label. Metrics may not be meaningful.")
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'confusion_matrix': np.array([[0, 0], [0, 0]]),
                    'predictions': predictions,
                    'actual_labels': actual_labels_binary
                }
            
            # Calculate metrics with proper averaging for binary classification
            accuracy = accuracy_score(actual_labels_binary, predictions)
            precision = precision_score(actual_labels_binary, predictions, average='binary', zero_division=0)
            recall = recall_score(actual_labels_binary, predictions, average='binary', zero_division=0)
            f1 = f1_score(actual_labels_binary, predictions, average='binary', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(actual_labels_binary, predictions)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': predictions,
                'actual_labels': actual_labels_binary
            }
        
    def cross_validate(self, k_folds=5):
        """
        Perform k-fold cross validation
        
        Parameters:
        - k_folds: Number of folds for cross validation
        
        Returns:
        - Dictionary with cross validation results
        """
        # Prepare all training data
        dataset = prepare_enhanced_dataset()
        all_data = dataset['training_data']
        
        # Shuffle the data
        np.random.shuffle(all_data)
        
        fold_size = len(all_data) // k_folds
        cv_results = {
            'accuracy_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': []
        }
        
        for fold in range(k_folds):
            # Split data into train and validation
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(all_data)
            
            val_data = all_data[start_idx:end_idx]
            train_data = all_data[:start_idx] + all_data[end_idx:]
            
            # Create a new model instance for this fold
            fold_model = EnhancedQuestionProbabilityModel(
                self.questions,
                self.appearance_counts,
                self.recency_weights
            )
            
            # Train on the training portion
            fold_model.train(train_data, epochs=10, learning_rate=0.001)
            
            # Evaluate on validation portion
            metrics = fold_model.evaluate_model(val_data)
            
            cv_results['accuracy_scores'].append(metrics['accuracy'])
            cv_results['precision_scores'].append(metrics['precision'])
            cv_results['recall_scores'].append(metrics['recall'])
            cv_results['f1_scores'].append(metrics['f1_score'])
        
        # Calculate mean and std for each metric
        cv_results['mean_accuracy'] = np.mean(cv_results['accuracy_scores'])
        cv_results['std_accuracy'] = np.std(cv_results['accuracy_scores'])
        cv_results['mean_precision'] = np.mean(cv_results['precision_scores'])
        cv_results['std_precision'] = np.std(cv_results['precision_scores'])
        cv_results['mean_recall'] = np.mean(cv_results['recall_scores'])
        cv_results['std_recall'] = np.std(cv_results['recall_scores'])
        cv_results['mean_f1'] = np.mean(cv_results['f1_scores'])
        cv_results['std_f1'] = np.std(cv_results['f1_scores'])
        
        return cv_results
    
    def plot_confusion_matrix(self, confusion_matrix, title="Confusion Matrix"):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0, 1], ['Not Appear', 'Appear'])
        plt.yticks([0, 1], ['Not Appear', 'Appear'])
        plt.tight_layout()
        plt.show()
    
    def analyze_threshold_performance(self, test_data, thresholds=None):
        """
        Analyze model performance across different probability thresholds
        
        Parameters:
        - test_data: Test dataset
        - thresholds: List of thresholds to test
        
        Returns:
        - Dictionary with performance metrics for each threshold
        """
        if thresholds is None:
            thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        threshold_results = []
        
        for threshold in thresholds:
            metrics = self.evaluate_model(test_data, threshold=threshold)
            threshold_results.append({
                'threshold': threshold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
        
        return threshold_results
    
    def plot_threshold_analysis(self, threshold_results):
        """
        Plot performance metrics across different thresholds
        """
        thresholds = [r['threshold'] for r in threshold_results]
        accuracies = [r['accuracy'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1_scores = [r['f1_score'] for r in threshold_results]
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2)
        
        plt.xlabel('Probability Threshold (%)')
        plt.ylabel('Score')
        plt.title('Model Performance vs Probability Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(min(thresholds)-5, max(thresholds)+5)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()


def prepare_enhanced_dataset():
    """
    Prepares a comprehensive dataset from all available question papers
    """
    # All questions from the papers (simplified version shown here, actual implementation would parse PDFs)
    historical_questions = [
        # Original questions from model
        "Explain the usage any 5 list operating methods with examples.",
        "Implement a telephone directory using Dictionaries.",
        "Develop a python program to find the sum of even numbers and odd numbers in the given list.",
        "How do you create and access dictionaries in Python? List and describe any 5 methods on dictionaries.",
        "Design a simple calculator with different mathematical operations using python script.",
        "Develop a python program to find whether the given string is palindrome or not.",
        "Let a be the list of values produced by range(1,50). Using the function filter and a lambda argument, write the expression that will produce each of the following: (i) A list of odd numbers in a (ii) A list of even numbers in a (iii) A list of values in a divisible by 3 and not divisible by 7.",
        "Develop a recursive function to generate prime numbers in a given range.",
        "Explain list comprehension with example.",
        "Define a function that takes a positive integer n, and then produces n lines of output in the pattern '+', '+ +', '+ + +', and so on. Is it possible to get the same output using a single loop? Justify.",
        "Illustrate the following with example: (i) DOC strings (ii) local and global variables (iii) pass by reference and pass by value in python.",
        "What is the difference between a method and a function? Give an example each.",
        "Design a Python class called account and implement the functions deposit, withdraw and display balance.",
        "Explain different ways of accessing attributes in a class.",
        "Describe the following along with example with respect to python: (i) Constructor (ii) destructor (iii) self keyword (iv) del keyword (v) static members.",
        "Explain the inheritance in python.",
        "List any 6 regular expression patterns in python along with their meaning and example.",
        "What are tkinter widgets? What geometry manager classes does tkinter expose? Write a program that has a button on a canvas which when clicked the message 'Hello World' has to be displayed.",
        "What is an event in python? Exemplify how for each widget, you can bind Python functions and methods to events.",
        "Discover what exception is produced by each of the following points. Then, for each, write a small example program that illustrates catching the exception using a try statement and continuing with exception after the interrupt: (i) Division by zero (ii) Opening a file that does not exist (iii) Indexing a list with an illegal value (iv) Using an improper key with a dictionary (v) Passing an improperly formatted expression to the function expr().",
        "Write a program that will prompt the user for a file name, read all the lines from the file into a list, sort the list, and then print the lines in sorted order.",
        "Explain MVT architecture of Django framework.",
        "Create an HTML form to read bio data of a candidate with fields First name, Last name, Age, Address, Hobbies (checkboxes), Gender (Radio buttons), and submit button to submit form data using GET method. On form submission the data should be displayed in proper format.",
        "Show the necessary steps and code to create web page and submit form data using POST method.",
        "Explain the functionalities of models, views, templates of Django with an example.",
        
        # Additional questions from MCA11_Supp_Nov_2022
        "Explain the significance of break, continue and pass with suitable example.",
        "Develop a python program that reads two integer values n and m from the user, then produces a box that is n wide and m deep.",
        "List and describe any five methods on tuples.",
        "Demonstrate slicing on strings. Also explain the use of join() and split() string methods with examples.",
        "Illustrate the different types of iterative statements available in Python.",
        "Develop a recursive Python function that recursively computes sum of elements in a list of lists.",
        "What is lambda function? What are the characteristics of a lambda function? Give an example.",
        "Develop a Python program that prints the intersection of two lists.",
        "Discus the following ways of passing parameters to functions: i. Keyword only parameters ii. Variable length arguments iii. pass by reference and pass by value.",
        "Develop Python function to calculate sum and product of two arguments, return them.",
        "Create a list of even numbers from 1 to 10 using the loop and filter method.",
        "Explain the following concepts in python with example: i) Data hiding ii) Inheritance.",
        "Explain the basic structure of class in python. Also explain the difference between data attributes and class attributes with example.",
        
        # Additional questions from SEE_June 2022
        "List the operators supported in Python? Describe specifically about identity and membership operator with a suitable example?",
        "Differentiate between lists and tuples in Python. How to create nested lists? Demonstrate how to create and print a 3-dimensional matrix with lists.",
        "Develop a Python program that counts the number of occurrences of a letter in a string, using dictionaries.",
        "Develop a Python code to extract the Even elements indices from the given list.",
        "Develop Python script that takes a list of words and returns the length of the longest one using tuples.",
        "Explain the following arguments to functions in python with examples: (i) Keyword arguments (ii) Default arguments (iii) Variable length arguments.",
        "Explain the scope of local and global variables.",
        "Write short notes on anonymous functions in python.",
        "Write a function to display the Fibonacci sequence up to nth term where n is provided by the user.",
        "Write short notes on the following: (i) Mapping (ii) Filtering (iii) List comprehension.",
        
        # Additional questions from Supp_2021
        "Explain the usage of the following methods with examples: i) Extend() ii) pop() iii) sort() iv) split() v) join()",
        "Explain what ord() and chr() function is used for. Using the same, Write a function that takes a string input and converts all uppercase letters to lowercase and vice versa",
        "Develop a python program to count the frequency of words in a string using dictionary.",
        "Develop a python program to print unique elements in a list.",
        "Write a python function to check whether the given string is palindrome or not Function should take a string as argument and return Boolean value.",
        "Implement anonymous(lambda) functions for: i) Filter out only even numbers from the given list. ii) Reduce the given list of numbers to its sum.",
        "Differentiate between keyword arguments, required arguments and variable length arguments with suitable example.",
        "Discuss list comprehension with example.",
        "What is recursion? Find the factorial of a number using recursive function.",
        
        # Additional questions from Supp_2020
        "Describe Arithmetic Operators, Assignment Operators, Comparison Operators, Logical Operators and Bitwise Operators in detail with examples.",
        "Implement a Python Program to reverse a number and also find the number of digits and Sum of digits in the reversed number.",
        "Illustrate break, continue and pass statements in Python.",
        "Demonstrate the creation and operation of dictionaries in Python.",
        "Write a python function that accepts a sentence containing alpha numeric characters and calculates the number of digits, uppercase and lowercase letters.",
        "Illustrate *args and **kwargs parameters in Python programming language with an example.",
        "Explain keyword, required and default function parameters with examples.",
        "Write a function to find the factorial of a number using functional programming."
    ]
    
    # Track appearance counts across all papers
    # Initialize as 1 for each newly added question, higher for questions appearing in multiple papers
    appearance_counts = [
        # Original 25 questions (keeping original counts)
        4, 3, 3, 4, 3, 2, 4, 2, 3, 2, 2, 3, 4, 3, 3, 5, 1, 2, 2, 3, 2, 2, 1, 1, 1,
        
        # Additional questions from MCA11_Supp_Nov_2022 (assign counts based on appearance)
        2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1,
        
        # Additional questions from SEE_June 2022
        1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
        
        # Additional questions from Supp_2021
        1, 1, 1, 1, 2, 1, 1, 2, 2,
        
        # Additional questions from Supp_2020
        1, 1, 2, 2, 1, 1, 2, 1
    ]
    
    
    recency_weights = [
        # Original questions (keeping original weights)
        0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5,
        
        # MCA11_Supp_Nov_2022 - most recent
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        
        # SEE_June 2022
        1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
        
        # Supp_2021
        1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        
        # Supp_2020
        1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2
    ]
    
    # Enhanced topic weights with more granular topics
    # topic_weights = {
    #     "data_structures": 1.5,      # Lists, tuples, dictionaries
    #     "functions": 1.3,            # Functions, parameters, recursion
    #     "oop": 1.3,                  # Classes, inheritance, attributes
    #     "regular_expressions": 0.0,  # Regex patterns
    #     "gui": 0.0,                  # Tkinter, widgets, events
    #     "exceptions_io": 0.9,        # Exception handling, file operations
    #     "web_development": 0.0,      # Django, HTML, web forms
    #     "operators": 1.0,            # Python operators
    #     "control_flow": 1.1,         # Loops, conditionals, break/continue
    #     "strings": 1.2,              # String operations, methods
    #     "functional_programming": 1.0 # Lambda, map, filter, reduce
    # }
    
    # Assign topics to each question
    # question_topics = [
    #     # Original 25 questions
    #     "data_structures", "data_structures", "data_structures", "data_structures", 
    #     "functions", "functions", "functions", "functions", "data_structures", 
    #     "functions", "functions", "oop", "oop", "oop", "oop", "oop", 
    #     "regular_expressions", "gui", "gui", "exceptions_io", "exceptions_io", 
    #     "web_development", "web_development", "web_development", "web_development",
        
    #     # Additional questions from MCA11_Supp_Nov_2022
    #     "control_flow", "control_flow", "data_structures", "strings", "control_flow",
    #     "functions", "functional_programming", "data_structures", "functions", 
    #     "functions", "functional_programming", "oop", "oop",
        
    #     # Additional questions from SEE_June 2022
    #     "operators", "data_structures", "data_structures", "data_structures", 
    #     "data_structures", "functions", "functions", "functional_programming", 
    #     "functions", "functional_programming",
        
    #     # Additional questions from Supp_2021
    #     "data_structures", "strings", "data_structures", "data_structures", 
    #     "strings", "functional_programming", "functions", "data_structures", "functions",
        
    #     # Additional questions from Supp_2020
    #     "operators", "functions", "control_flow", "data_structures", "strings", 
    #     "functions", "functions", "functional_programming"
    # ]
    
    # Create training data for the model
    # Format: [(question_index, appeared_or_not), ...]
    # This is just an example - ideally you would use actual data on which questions appeared in exams
    training_data = [
    # Original 25 questions
    (0, 1),  # "Explain the usage any 5 list operating methods with examples." (appeared 4 times)
    (1, 1),  # "Implement a telephone directory using Dictionaries." (appeared 3 times)
    (2, 1),  # "Develop a python program to find the sum of even numbers and odd numbers in the given list." (appeared 3 times)
    (3, 1),  # "How do you create and access dictionaries in Python? List and describe any 5 methods on dictionaries." (appeared 4 times)
    (4, 1),  # "Design a simple calculator with different mathematical operations using python script." (appeared 3 times)
    (5, 0),  # "Develop a python program to find whether the given string is palindrome or not." (appeared 2 times)
    (6, 5),  # "Let a be the list of values produced by range(1,50)..." (appeared 4 times)
    (7, 0),  # "Develop a recursive function to generate prime numbers in a given range." (appeared 2 times)
    (8, 1),  # "Explain list comprehension with example." (appeared 3 times)
    (9, 0),  # "Define a function that takes a positive integer n..." (appeared 2 times)
    (10, 0), # "Illustrate the following with example: (i) DOC strings..." (appeared 2 times)
    (11, 1), # "What is the difference between a method and a function? Give an example each." (appeared 3 times)
    (12, 1), # "Design a Python class called account and implement the functions deposit, withdraw and display balance." (appeared 4 times)
    (13, 1), # "Explain different ways of accessing attributes in a class." (appeared 3 times)
    (14, 1), # "Describe the following along with example with respect to python: (i) Constructor..." (appeared 3 times)
    (15, 1), # "Explain the inheritance in python." (appeared 5 times)
    (16, 0), # "List any 6 regular expression patterns in python along with their meaning and example." (appeared 1 time)
    (17, 0), # "What are tkinter widgets? What geometry manager classes does tkinter expose?..." (appeared 2 times)
    (18, 0), # "What is an event in python? Exemplify how for each widget..." (appeared 2 times)
    (19, 1), # "Discover what exception is produced by each of the following points..." (appeared 3 times)
    (20, 0), # "Write a program that will prompt the user for a file name..." (appeared 2 times)
    (21, 0), # "Explain MVT architecture of Django framework." (appeared 2 times)
    (22, 0), # "Create an HTML form to read bio data of a candidate..." (appeared 1 time)
    (23, 0), # "Show the necessary steps and code to create web page and submit form data using POST method." (appeared 1 time)
    (24, 0), # "Explain the functionalities of models, views, templates of Django with an example." (appeared 1 time)
    
    # Additional questions from MCA11_Supp_Nov_2022
    (25, 2), # "Explain the significance of break, continue and pass with suitable example." (appeared 2 times)
    (26, 0), # "Develop a python program that reads two integer values n and m from the user..." (appeared 1 time)
    (27, 0), # "List and describe any five methods on tuples." (appeared 2 times)
    (28, 0), # "Demonstrate slicing on strings. Also explain the use of join() and split() string methods with examples." (appeared 2 times)
    (29, 0), # "Illustrate the different types of iterative statements available in Python." (appeared 1 time)
    (30, 0), # "Develop a recursive Python function that recursively computes sum of elements in a list of lists." (appeared 1 time)
    (31, 0), # "What is lambda function? What are the characteristics of a lambda function? Give an example." (appeared 2 times)
    (32, 0), # "Develop a Python program that prints the intersection of two lists." (appeared 1 time)
    (33, 0), # "Discus the following ways of passing parameters to functions..." (appeared 2 times)
    (34, 0), # "Develop Python function to calculate sum and product of two arguments, return them." (appeared 1 time)
    (35, 0), # "Create a list of even numbers from 1 to 10 using the loop and filter method." (appeared 1 time)
    (36, 0), # "Explain the following concepts in python with example: i) Data hiding ii) Inheritance." (appeared 2 times)
    (37, 0), # "Explain the basic structure of class in python. Also explain the difference between data attributes and class attributes with example." (appeared 1 time)
    
    # Additional questions from SEE_June 2022
    (38, 0), # "List the operators supported in Python? Describe specifically about identity and membership operator with a suitable example?" (appeared 1 time)
    (39, 0), # "Differentiate between lists and tuples in Python. How to create nested lists?..." (appeared 1 time)
    (40, 0), # "Develop a Python program that counts the number of occurrences of a letter in a string, using dictionaries." (appeared 1 time)
    (41, 1), # "Develop a Python code to extract the Even elements indices from the given list." (appeared 1 time)
    (42, 1), # "Develop Python script that takes a list of words and returns the length of the longest one using tuples." (appeared 1 time)
    (43, 0), # "Explain the following arguments to functions in python with examples..." (appeared 2 times)
    (44, 0), # "Explain the scope of local and global variables." (appeared 2 times)
    (45, 0), # "Write short notes on anonymous functions in python." (appeared 1 time)
    (46, 0), # "Write a function to display the Fibonacci sequence up to nth term where n is provided by the user." (appeared 1 time)
    (47, 0), # "Write short notes on the following: (i) Mapping (ii) Filtering (iii) List comprehension." (appeared 1 time)
    
    # Additional questions from Supp_2021
    (48, 1), # "Explain the usage of the following methods with examples: i) Extend() ii) pop() iii) sort() iv) split() v) join()" (appeared 1 time)
    (49, 0), # "Explain what ord() and chr() function is used for. Using the same, Write a function that takes a string input and converts all uppercase letters to lowercase and vice versa" (appeared 1 time)
    (50, 0), # "Develop a python program to count the frequency of words in a string using dictionary." (appeared 1 time)
    (51, 0), # "Develop a python program to print unique elements in a list." (appeared 1 time)
    (52, 0), # "Write a python function to check whether the given string is palindrome or not Function should take a string as argument and return Boolean value." (appeared 2 times)
    (53, 0), # "Implement anonymous(lambda) functions for: i) Filter out only even numbers from the given list. ii) Reduce the given list of numbers to its sum." (appeared 1 time)
    (54, 0), # "Differentiate between keyword arguments, required arguments and variable length arguments with suitable example." (appeared 1 time)
    (55, 0), # "Discuss list comprehension with example." (appeared 2 times)
    (56, 1), # "What is recursion? Find the factorial of a number using recursive function." (appeared 2 times)
    
    # Additional questions from Supp_2020
    (57, 0), # "Describe Arithmetic Operators, Assignment Operators, Comparison Operators, Logical Operators and Bitwise Operators in detail with examples." (appeared 1 time)
    (58, 1), # "Implement a Python Program to reverse a number and also find the number of digits and Sum of digits in the reversed number." (appeared 1 time)
    (59, 0), # "Illustrate break, continue and pass statements in Python." (appeared 2 times)
    (60, 0), # "Demonstrate the creation and operation of dictionaries in Python." (appeared 2 times)
    (61, 0), # "Write a python function that accepts a sentence containing alpha numeric characters and calculates the number of digits, uppercase and lowercase letters." (appeared 1 time)
    (62, 0), # "Illustrate *args and **kwargs parameters in Python programming language with an example." (appeared 1 time)
    (63, 0), # "Explain keyword, required and default function parameters with examples." (appeared 2 times)
    (64, 0), # "Write a function to find the factorial of a number using functional programming." (appeared 1 time)
]
    
    # For this example, we'll assume questions with higher appearance counts are more likely to appear
    # in future exams, and will create synthetic training data accordingly
    for i, count in enumerate(appearance_counts):
        # If a question appeared 3 or more times, mark it as likely to appear in next exam (1)
        # Otherwise, mark it as less likely (0)
        appeared = 1 if count >= 3 else 0
        training_data.append((i, appeared))
    
    return {
        'historical_questions': historical_questions,
        'appearance_counts': appearance_counts,
        'recency_weights': recency_weights,
        # 'topic_weights': topic_weights,
        # 'question_topics': question_topics,
        'training_data': training_data
    }

# Function to train and demonstrate the enhanced model
def train_with_extended_dataset():
    # Get the prepared dataset
    dataset = prepare_enhanced_dataset()
    
    # Split data into train and test (80-20 split)
    training_data = dataset['training_data']
    split_point = int(0.8 * len(training_data))
    train_data = training_data[:split_point]
    test_data = training_data[split_point:]
    
    # Initialize the model with the extended dataset
    model = EnhancedQuestionProbabilityModel(
        dataset['historical_questions'],
        dataset['appearance_counts'],
        dataset['recency_weights']
    )
    
    # Train the model with the training data
    model.train(train_data, epochs=20, learning_rate=0.001)
    

    # Get the prepared dataset
    dataset = prepare_enhanced_dataset()
    
    # Initialize the model with the extended dataset
    model = EnhancedQuestionProbabilityModel(
        dataset['historical_questions'],
        dataset['appearance_counts'],
        dataset['recency_weights']
        # dataset['topic_weights']
    )
    
    # Train the model with the prepared training data
    model.train(dataset['training_data'], epochs=20, learning_rate=0.001)
    
    # Predict probabilities for all questions
    probabilities = model.predict_probabilities()
    
    # Print top 10 most likely questions
    print("Top 10 Most Likely Questions to Appear in Next Exam:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
    for i, (question, prob) in enumerate(sorted_probs[:15]):
        print(f"{i+1}. {question} : {prob}")
    
    # Test with some new questions
    new_questions = [
        "Explain object-oriented programming concepts in Python.",
        "What are classes in python.",
        "how does inheritance work in python.",
        "Discuss list comprehension with example."
    ]
    
    print("\nPredictions for New Questions:")
    for q in new_questions:
        prob = model.predict_probabilities(q)
        print(f"- {q} : {prob:.2f}%")
        
def train_and_evaluate_model():
    """
    Train the model and evaluate its accuracy
    """
    # Get the prepared dataset
    dataset = prepare_enhanced_dataset()
    
    # Split data into train and test (80-20 split)
    training_data = dataset['training_data']
    np.random.shuffle(training_data)  # Shuffle for better split
    
    split_point = int(0.8 * len(training_data))
    train_data = training_data[:split_point]
    test_data = training_data[split_point:]
    
    print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Initialize the model
    model = EnhancedQuestionProbabilityModel(
        dataset['historical_questions'],
        dataset['appearance_counts'],
        dataset['recency_weights']
    )
    
    # Train the model
    print("Training the model...")
    model.train(train_data, epochs=20, learning_rate=0.001)
    
    # Evaluate on test data
    print("\nEvaluating model performance...")
    
    # Test different thresholds
    threshold_results = model.analyze_threshold_performance(test_data)
    
    print("\nPerformance across different thresholds:")
    print("Threshold | Accuracy | Precision | Recall | F1-Score")
    print("-" * 55)
    for result in threshold_results:
        print(f"{result['threshold']:8.0f}% | {result['accuracy']:8.3f} | {result['precision']:9.3f} | {result['recall']:6.3f} | {result['f1_score']:8.3f}")
    
    # Find best threshold based on F1 score
    best_result = max(threshold_results, key=lambda x: x['f1_score'])
    best_threshold = best_result['threshold']
    
    print(f"\nBest threshold: {best_threshold}% (F1-Score: {best_result['f1_score']:.3f})")
    
    # Detailed evaluation with best threshold
    detailed_metrics = model.evaluate_model(test_data, threshold=best_threshold)
    
    print(f"\nDetailed Metrics (Threshold: {best_threshold}%):")
    print(f"Accuracy:  {detailed_metrics['accuracy']:.3f}")
    print(f"Precision: {detailed_metrics['precision']:.3f}")
    print(f"Recall:    {detailed_metrics['recall']:.3f}")
    print(f"F1-Score:  {detailed_metrics['f1_score']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(detailed_metrics['confusion_matrix'])
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = model.cross_validate(k_folds=5)
    
    print(f"Cross-Validation Results:")
    print(f"Accuracy:  {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"Precision: {cv_results['mean_precision']:.3f} ± {cv_results['std_precision']:.3f}")
    print(f"Recall:    {cv_results['mean_recall']:.3f} ± {cv_results['std_recall']:.3f}")
    print(f"F1-Score:  {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
    
    # Plot results
    model.plot_threshold_analysis(threshold_results)
    model.plot_confusion_matrix(detailed_metrics['confusion_matrix'])
    
    # Show top predictions
    probabilities = model.predict_probabilities()
    print("\nTop 10 Most Likely Questions to Appear:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
    for i, (question, prob) in enumerate(sorted_probs[:10]):
        print(f"{i+1}. {prob} - {question[:80]}...")
    
    return model, detailed_metrics, cv_results

# Modified main function
if __name__ == "__main__":
    # train_with_extended_dataset()
    model, metrics, cv_results = train_and_evaluate_model()