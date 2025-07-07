import json
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
# Use scikit-learn for text vectorization instead of sentence transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class FinancialDataEmbeddingsGenerator:
    """
    Generate embeddings for financial data products, datasets, and physical schema
    using TF-IDF vectorization and dimensionality reduction.
    """
    
    def __init__(self, max_features=512, n_components=128):
        """
        Initialize the embeddings generator with TF-IDF parameters.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            n_components (int): Number of components for dimensionality reduction
        """
        self.stop_words = set(stopwords.words('english'))
        # Add financial and technical stop words
        self.stop_words.update([
            'data', 'product', 'dataset', 'financial', 'services', 'management',
            'system', 'information', 'process', 'processing', 'service', 'operations',
            'banking', 'analysis', 'analytics', 'calculation', 'calculations',
            'monitoring', 'reporting', 'support', 'supporting', 'comprehensive',
            'various', 'multiple', 'enable', 'enabling', 'provide', 'providing'
        ])
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize the TF-IDF vectorizer and SVD for dimensionality reduction
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            ngram_range=(1, 2),
            stop_words='english'
        )
        """
        We use singular value decomposition (SVD). Contrary to PCA, this estimator does not center 
        the data before computing the singular value decomposition. 
        This means it can work with sparse matrices efficiently.
        """
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        print(f"Initialized TF-IDF vectorizer with {max_features} features and SVD with {n_components} components")
        
    def load_data_products(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data products from JSON file
        
        Args:
            filename (str): Path to the JSON file containing data products
            
        Returns:
            List[Dict[str, Any]]: List of data products
        """
        with open(filename, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis by removing special characters,
        converting to lowercase, tokenizing, removing stop words and lemmatizing.
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: The preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Replace underscores with spaces
        text = text.replace('_', ' ')
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_domain(self, product_name: str) -> str:
        """
        Extract domain from product name
        
        Args:
            product_name (str): Name of the data product
            
        Returns:
            str: Domain name
        """
        if 'investment_banking' in product_name:
            return 'Investment Banking'
        elif 'wealth_management' in product_name:
            return 'Wealth Management'
        elif 'risk_management' in product_name:
            return 'Risk Management'
        elif 'corporate_banking' in product_name:
            return 'Corporate Banking'
        elif 'retail_banking' in product_name:
            return 'Retail Banking'
        elif 'asset_management' in product_name:
            return 'Asset Management'
        else:
            return 'Other'
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using TF-IDF and SVD
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            np.ndarray: Matrix of embeddings with shape (len(texts), n_components)
        """
        if not texts:
            return np.array([])
            
        # Fit or transform based on whether this is the first call
        try:
            X = self.vectorizer.transform(texts)
        except ValueError:
            # If vocabulary hasn't been built yet
            X = self.vectorizer.fit_transform(texts)
            self.svd.fit(X)
            
        # Apply dimensionality reduction
        X_reduced = self.svd.transform(X)
        
        return X_reduced
    
    def extract_data_product_embeddings(self, data_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract embeddings for data product names and descriptions
        
        Args:
            data_products (List[Dict[str, Any]]): List of data products
            
        Returns:
            pd.DataFrame: DataFrame with product information and embeddings
        """
        records = []
        
        texts_for_embedding = []
        for product in data_products:
            # Extract product information
            product_name = product['info']['name']
            product_desc = product['info']['description']
            
            # Combine name and description for embedding
            combined_text = f"{product_name} {product_desc}"
            processed_text = self.preprocess_text(combined_text)
            
            # Add to list for batch embedding generation
            texts_for_embedding.append(processed_text)
            
            # Determine domain
            domain = self.extract_domain(product_name)
            
            records.append({
                'product_name': product_name,
                'product_description': product_desc,
                'processed_text': processed_text,
                'domain': domain
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Generate embeddings for all products at once
        if texts_for_embedding:
            embeddings = self.generate_embeddings(texts_for_embedding)
            
            # Store embeddings as separate columns instead of as a single object
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'product_emb_{i}' for i in range(embeddings.shape[1])]
            )
            
            # Concatenate the original dataframe with the embedding dataframe
            df = pd.concat([df, embedding_df], axis=1)
        
        return df
    
    def extract_dataset_embeddings(self, data_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract embeddings for dataset names and descriptions
        
        Args:
            data_products (List[Dict[str, Any]]): List of data products
            
        Returns:
            pd.DataFrame: DataFrame with dataset information and embeddings
        """
        records = []
        
        texts_for_embedding = []
        for product in data_products:
            # Extract product information
            product_name = product['info']['name']
            product_desc = product['info']['description']
            domain = self.extract_domain(product_name)
            
            # Extract dataset information
            for dataset in product['datasets']:
                dataset_name = dataset['name']
                dataset_desc = dataset['description']
                
                # Combine name and description for embedding
                combined_text = f"{dataset_name} {dataset_desc}"
                processed_text = self.preprocess_text(combined_text)
                
                # Add to list for batch embedding generation
                texts_for_embedding.append(processed_text)
                
                records.append({
                    'product_name': product_name,
                    'dataset_name': dataset_name,
                    'dataset_description': dataset_desc,
                    'processed_text': processed_text,
                    'domain': domain
                })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Generate embeddings for all datasets at once
        if texts_for_embedding:
            embeddings = self.generate_embeddings(texts_for_embedding)
            
            # Store embeddings as separate columns instead of as a single object
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'dataset_emb_{i}' for i in range(embeddings.shape[1])]
            )
            
            # Concatenate the original dataframe with the embedding dataframe
            df = pd.concat([df, embedding_df], axis=1)
        
        return df
    
    def extract_schema_embeddings(self, data_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract embeddings for physical schema fields and descriptions
        
        Args:
            data_products (List[Dict[str, Any]]): List of data products
            
        Returns:
            pd.DataFrame: DataFrame with schema information and embeddings
        """
        records = []
        
        texts_for_embedding = []
        for product in data_products:
            # Extract product information
            product_name = product['info']['name']
            domain = self.extract_domain(product_name)
            
            # Extract dataset information
            for dataset in product['datasets']:
                dataset_name = dataset['name']
                
                # Extract schema information
                if 'physicalSchema' in dataset:
                    for field in dataset['physicalSchema']:
                        field_name = field['name']
                        field_type = field['type']
                        field_desc = field['description']
                        
                        # Combine field information for embedding
                        combined_text = f"{field_name} {field_type} {field_desc}"
                        processed_text = self.preprocess_text(combined_text)
                        
                        # Add to list for batch embedding generation
                        texts_for_embedding.append(processed_text)
                        
                        records.append({
                            'product_name': product_name,
                            'dataset_name': dataset_name,
                            'field_name': field_name,
                            'field_type': field_type,
                            'field_description': field_desc,
                            'processed_text': processed_text,
                            'domain': domain
                        })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Generate embeddings for all schema fields at once
        if texts_for_embedding:
            embeddings = self.generate_embeddings(texts_for_embedding)
            
            # Store embeddings as separate columns instead of as a single object
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'schema_emb_{i}' for i in range(embeddings.shape[1])]
            )
            
            # Concatenate the original dataframe with the embedding dataframe
            df = pd.concat([df, embedding_df], axis=1)
            
        return df
    
    def visualize_embeddings(self, df: pd.DataFrame, embedding_prefix: str, 
                           color_column: str = 'domain', method: str = 'tsne',
                           title: str = 'Embedding Visualization',
                           save_path: str = None) -> None:
        """
        Visualize embeddings using dimensionality reduction (t-SNE or PCA)
        
        Args:
            df (pd.DataFrame): DataFrame containing embeddings
            embedding_column (str): Name of the column containing embeddings
            color_column (str): Name of the column to use for coloring points
            method (str): Dimensionality reduction method ('tsne' or 'pca')
            title (str): Title for the plot
            save_path (str, optional): Path to save the visualization image
        """
        # Extract embeddings from columns with the given prefix
        embedding_cols = [col for col in df.columns if col.startswith(embedding_prefix)]
        if not embedding_cols:
            print(f"No embedding columns found with prefix {embedding_prefix}")
            return
            
        # Extract the embedding matrix
        embeddings = df[embedding_cols].values
        
        # Perform dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(embeddings)
            method_name = 't-SNE'
        else:
            reducer = PCA(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(embeddings)
            method_name = 'PCA'
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'color': df[color_column]
        })
        
        # Plot
        plt.figure(figsize=(12, 10))
        ax = sns.scatterplot(data=plot_df, x='x', y='y', hue='color', palette='viridis')
        plt.title(f'{title} ({method_name})', fontsize=16)
        plt.xlabel(f'{method_name} Dimension 1', fontsize=14)
        plt.ylabel(f'{method_name} Dimension 2', fontsize=14)
        plt.legend(title=color_column, fontsize=12, title_fontsize=14)
        
        # Add annotations for some points (limit to avoid clutter)
        if len(df) <= 50:  # Only annotate if not too many points
            for i, row in df.head(30).iterrows():
                if 'product_name' in df.columns:
                    label = row['product_name'].split('_')[-1]  # Just the last part of the name
                elif 'dataset_name' in df.columns:
                    label = row['dataset_name'].split('_')[-1]  # Just the last part of the name
                elif 'field_name' in df.columns:
                    label = row['field_name']
                else:
                    label = str(i)
                    
                plt.annotate(label, (plot_df.iloc[i]['x'], plot_df.iloc[i]['y']), 
                             fontsize=8, alpha=0.75)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        plt.show()
        
    def save_embeddings(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save embeddings to a pickle file
        
        Args:
            df (pd.DataFrame): DataFrame with embeddings
            filename (str): Path to save the embeddings
        """
        df.to_pickle(filename)
        print(f"Embeddings saved to {filename}")
    
    def load_embeddings(self, filename: str) -> pd.DataFrame:
        """
        Load embeddings from a pickle file
        
        Args:
            filename (str): Path to the embeddings file
            
        Returns:
            pd.DataFrame: DataFrame with loaded embeddings
        """
        df = pd.read_pickle(filename)
        print(f"Loaded embeddings from {filename} with shape {df.shape}")
        return df
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score (between -1 and 1)
        """
        # Reshape for cosine_similarity function which expects 2D arrays
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity using sklearn
        return cosine_similarity(e1, e2)[0][0]
    
    def find_similar_items(self, df: pd.DataFrame, query_text: str, 
                         embedding_prefix: str, n: int = 5) -> pd.DataFrame:
        """
        Find items similar to a query text based on embedding similarity
        
        Args:
            df (pd.DataFrame): DataFrame with embeddings
            query_text (str): Query text to find similar items for
            embedding_prefix (str): Prefix for embedding column names (e.g. 'product_emb')
            n (int): Number of similar items to return
            
        Returns:
            pd.DataFrame: DataFrame with similar items and similarity scores
        """
        # Process the query text and generate embedding
        processed_query = self.preprocess_text(query_text)
        # We need to pass it as a list to generate_embeddings
        query_embedding = self.generate_embeddings([processed_query])[0]
        
        # Get the embedding columns
        embedding_cols = [col for col in df.columns if col.startswith(embedding_prefix)]
        if not embedding_cols:
            print(f"No embedding columns found with prefix {embedding_prefix}")
            return pd.DataFrame()
        
        # Calculate similarity scores
        similarities = []
        for i, row in df.iterrows():
            # Extract the embedding vector from the columns
            item_embedding = row[embedding_cols].values
            similarity = self.calculate_similarity(query_embedding, item_embedding)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity and get top n
        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:n]
        
        # Create a DataFrame with the results
        result_indices = [item['index'] for item in similarities]
        result_df = df.iloc[result_indices].copy()
        
        # Add similarity scores
        result_df['similarity_score'] = [item['similarity'] for item in similarities]
        
        return result_df.sort_values('similarity_score', ascending=False)
    
    def run_complete_embedding_analysis(self, filename: str):
        """
        Run the complete embedding analysis pipeline
        
        Args:
            filename (str): Path to the JSON file containing data products
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for products, datasets, and schema embeddings
        """
        print("Loading data products...")
        data_products = self.load_data_products(filename)
        
        print(f"Loaded {len(data_products)} data products")
        
        print("\nGenerating product embeddings...")
        product_df = self.extract_data_product_embeddings(data_products)
        print(f"Generated embeddings for {len(product_df)} products")
        
        print("\nGenerating dataset embeddings...")
        dataset_df = self.extract_dataset_embeddings(data_products)
        print(f"Generated embeddings for {len(dataset_df)} datasets")
        
        print("\nGenerating schema embeddings...")
        schema_df = self.extract_schema_embeddings(data_products)
        print(f"Generated embeddings for {len(schema_df)} schema fields")
        
        # Save embeddings
        os.makedirs('embeddings', exist_ok=True)
        self.save_embeddings(product_df, 'embeddings/product_embeddings.pkl')
        self.save_embeddings(dataset_df, 'embeddings/dataset_embeddings.pkl')
        self.save_embeddings(schema_df, 'embeddings/schema_embeddings.pkl')
        
        # Visualize embeddings
        print("\nVisualizing product embeddings...")
        self.visualize_embeddings(product_df, 'product_embedding', 'domain', 
                                title='Data Product Embeddings', 
                                save_path='embeddings/product_embeddings_visualization.png')
        
        print("\nVisualizing dataset embeddings...")
        self.visualize_embeddings(dataset_df, 'dataset_embedding', 'domain', 
                                title='Dataset Embeddings',
                                save_path='embeddings/dataset_embeddings_visualization.png')
        
        # Schema embeddings might be too many to visualize well
        if len(schema_df) <= 100:
            print("\nVisualizing schema embeddings...")
            self.visualize_embeddings(schema_df, 'schema_embedding', 'domain', 
                                    title='Schema Field Embeddings',
                                    save_path='embeddings/schema_embeddings_visualization.png')
        else:
            print(f"\nSkipping schema embeddings visualization (too many points: {len(schema_df)})")
            
        return product_df, dataset_df, schema_df


def main():
    """Main function to run the embedding analysis"""
    print("Initializing Financial Data Embeddings Generator...")
    embedding_generator = FinancialDataEmbeddingsGenerator()
    
    try:
        print("Running embedding analysis on financial data products...")
        product_df, dataset_df, schema_df = embedding_generator.run_complete_embedding_analysis(
            'financial_data_products.json'
        )
        
        print("\nEmbedding analysis complete!")
        print("Results saved to:")
        print("- embeddings/product_embeddings.pkl")
        print("- embeddings/dataset_embeddings.pkl")
        print("- embeddings/schema_embeddings.pkl")
        print("- embeddings visualizations")
        
        # Example of similarity search
        print("\nExample similarity search:")
        query = "investment risk assessment"
        print(f"Query: '{query}'")
        
        similar_products = embedding_generator.find_similar_items(
            product_df, query, 'product_emb', n=3
        )
        print("\nTop 3 similar products:")
        for i, row in similar_products.iterrows():
            print(f"  {row['product_name']} (Similarity: {row['similarity_score']:.4f})")
            print(f"  Description: {row['product_description']}")
            print()
        
    except FileNotFoundError:
        print("Error: financial_data_products.json not found.")
        print("Please ensure the data product JSON file exists.")
    except Exception as e:
        print(f"Error during embedding analysis: {str(e)}")


if __name__ == "__main__":
    main()
