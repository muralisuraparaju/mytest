import json
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any, Tuple
import seaborn as sns

# Download required NLTK data if needed
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


class EmbeddingWordCloudGenerator:
    """
    Generate word clouds and cluster analysis based on text embeddings
    from financial data products, datasets, and physical schema.
    """
    
    def __init__(self, max_features=512, n_components=128):
        """
        Initialize the generator with TF-IDF parameters.
        
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
        
        # Initialize the TF-IDF vectorizer for embeddings
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # For dimensionality reduction
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        print(f"Initialized with {max_features} TF-IDF features and {n_components} SVD components")
    
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
        Preprocess text for analysis
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
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
    
    def extract_text_features(self, data_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract text features from data products
        
        Args:
            data_products (List[Dict[str, Any]]): List of data products
            
        Returns:
            pd.DataFrame: DataFrame with text features
        """
        records = []
        
        for product in data_products:
            # Extract product information
            product_name = product['info']['name']
            product_desc = product['info']['description']
            domain = self.extract_domain(product_name)
            
            # Process product text
            product_text = f"{product_name} {product_desc}"
            processed_product_text = self.preprocess_text(product_text)
            
            # Add product level record
            records.append({
                'name': product_name,
                'description': product_desc,
                'processed_text': processed_product_text,
                'domain': domain,
                'level': 'product',
                'full_text': product_text
            })
            
            # Process datasets
            for dataset in product['datasets']:
                dataset_name = dataset['name']
                dataset_desc = dataset['description']
                
                dataset_text = f"{dataset_name} {dataset_desc}"
                processed_dataset_text = self.preprocess_text(dataset_text)
                
                # Add dataset level record
                records.append({
                    'name': dataset_name,
                    'description': dataset_desc,
                    'processed_text': processed_dataset_text,
                    'domain': domain,
                    'level': 'dataset',
                    'full_text': dataset_text,
                    'product_name': product_name
                })
                
                # Process schema fields if available
                if 'physicalSchema' in dataset:
                    # Combine all field information for this dataset's schema
                    schema_texts = []
                    
                    for field in dataset['physicalSchema']:
                        field_name = field['name']
                        field_type = field['type']
                        field_desc = field['description']
                        
                        field_text = f"{field_name} {field_type} {field_desc}"
                        processed_field_text = self.preprocess_text(field_text)
                        schema_texts.append(processed_field_text)
                    
                    # Add schema level record with combined fields
                    if schema_texts:
                        combined_schema_text = " ".join(schema_texts)
                        records.append({
                            'name': f"{dataset_name}_schema",
                            'description': f"Schema fields for {dataset_name}",
                            'processed_text': combined_schema_text,
                            'domain': domain,
                            'level': 'schema',
                            'full_text': combined_schema_text,
                            'product_name': product_name,
                            'dataset_name': dataset_name
                        })
        
        return pd.DataFrame(records)
    
    def generate_wordcloud(self, text_data: str, title: str, save_path: str = None, 
                         figsize=(10, 6), max_words=100, mask=None) -> None:
        """
        Generate and display word cloud
        
        Args:
            text_data (str): Text to generate word cloud from
            title (str): Title for the word cloud
            save_path (str, optional): Path to save the word cloud image
            figsize (tuple): Figure size for the plot
            max_words (int): Maximum number of words to include
            mask (np.ndarray, optional): Mask for word cloud shape
        """
        if not text_data or text_data.strip() == '':
            print(f"No text data available for '{title}'")
            return
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            collocations=False,
            mask=mask
        ).generate(text_data)
        
        # Display
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wordcloud saved to {save_path}")
        
        plt.show()
    
    def create_domain_wordclouds(self, df: pd.DataFrame, level: str = None) -> None:
        """
        Create word clouds for each domain
        
        Args:
            df (pd.DataFrame): DataFrame with text features
            level (str, optional): Filter by level ('product', 'dataset', 'schema')
        """
        # Filter by level if specified
        if level:
            filtered_df = df[df['level'] == level].copy()
        else:
            filtered_df = df.copy()
        
        if len(filtered_df) == 0:
            print(f"No data available for level '{level}'")
            return
        
        # Create output directory
        os.makedirs('wordclouds', exist_ok=True)
        
        # Generate word cloud for each domain
        for domain in filtered_df['domain'].unique():
            domain_text = ' '.join(filtered_df[filtered_df['domain'] == domain]['processed_text'])
            
            level_str = f"_{level}" if level else ""
            title = f"{domain}{level_str} Word Cloud"
            save_path = f"wordclouds/{domain.lower().replace(' ', '_')}{level_str}_wordcloud.png"
            
            self.generate_wordcloud(domain_text, title, save_path)
    
    def create_embedding_based_wordclouds(self, df: pd.DataFrame) -> None:
        """
        Create word clouds based on embedding clusters
        
        Args:
            df (pd.DataFrame): DataFrame with text features
        """
        # Ensure we have processed text
        if 'processed_text' not in df.columns or len(df) == 0:
            print("No processed text available for embedding-based word clouds")
            return
            
        # Transform text to TF-IDF features
        print("Generating TF-IDF features...")
        X = self.vectorizer.fit_transform(df['processed_text'])
        
        # Get feature names for later use
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply dimensionality reduction
        print("Applying dimensionality reduction...")
        X_reduced = self.svd.fit_transform(X)
        
        # Determine optimal number of clusters
        max_clusters = min(8, len(df) - 1)  # Don't try more clusters than data points
        if max_clusters < 2:
            print("Not enough data for clustering")
            return
            
        print("Finding optimal number of clusters...")
        scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_reduced)
            silhouette = silhouette_score(X_reduced, cluster_labels)
            scores.append((n_clusters, silhouette))
            print(f"  {n_clusters} clusters: silhouette = {silhouette:.3f}")
            
        # Select best number of clusters
        optimal_clusters = max(scores, key=lambda x: x[1])[0]
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        # Perform K-means clustering with optimal clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_reduced)
        
        # Analyze cluster characteristics
        centroids = kmeans.cluster_centers_
        
        # Create directory for cluster wordclouds
        os.makedirs('wordclouds/clusters', exist_ok=True)
        
        # For each cluster
        for cluster_id in range(optimal_clusters):
            # Get items in this cluster
            cluster_items = df[df['cluster'] == cluster_id]
            
            # Extract top terms from cluster centroid
            centroid = centroids[cluster_id]
            original_space_centroid = self.svd.inverse_transform(centroid.reshape(1, -1))
            sorted_indices = original_space_centroid[0].argsort()[::-1]
            top_feature_indices = sorted_indices[:30]  # Get top 30 features
            top_features = [feature_names[i] for i in top_feature_indices]
            
            # Create frequency dictionary for wordcloud
            freq_dict = {}
            for feature, value in zip(top_features, original_space_centroid[0][top_feature_indices]):
                if value > 0:
                    freq_dict[feature] = value
            
            # Determine cluster theme
            top_terms = top_features[:5]
            theme = " ".join(top_terms)
            
            # Create title with cluster info
            title = f"Cluster {cluster_id}: {theme}"
            save_path = f"wordclouds/clusters/cluster_{cluster_id}_wordcloud.png"
            
            # Create word cloud from frequency dictionary
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                collocations=False
            ).generate_from_frequencies(freq_dict)
            
            # Display
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.tight_layout()
            
            # Save
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster {cluster_id} wordcloud saved to {save_path}")
            
            plt.show()
            
            # Print cluster summary
            print(f"\nCluster {cluster_id} Summary:")
            print(f"Theme: {theme}")
            print(f"Size: {len(cluster_items)} items")
            print(f"Top terms: {', '.join(top_terms)}")
            
            # Domain distribution
            domain_counts = cluster_items['domain'].value_counts()
            print("Domain distribution:")
            for domain, count in domain_counts.items():
                print(f"  {domain}: {count} items ({count/len(cluster_items):.1%})")
            
            # Level distribution
            level_counts = cluster_items['level'].value_counts()
            print("Level distribution:")
            for level, count in level_counts.items():
                print(f"  {level}: {count} items ({count/len(cluster_items):.1%})")
                
            # Sample items
            print("Sample items:")
            for _, item in cluster_items.sample(min(3, len(cluster_items))).iterrows():
                print(f"  - {item['name']}: {item['description'][:100]}...")
                
            print("-" * 80)
            
        # Visualize clusters in 2D
        self.visualize_clusters(df, X_reduced, optimal_clusters)
            
    def visualize_clusters(self, df: pd.DataFrame, X_reduced: np.ndarray, n_clusters: int) -> None:
        """
        Visualize clusters in 2D using PCA
        
        Args:
            df (pd.DataFrame): DataFrame with cluster assignments
            X_reduced (np.ndarray): Reduced feature matrix
            n_clusters (int): Number of clusters
        """
        # Further reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_reduced)
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': df['cluster'],
            'domain': df['domain'],
            'level': df['level'],
            'name': df['name']
        })
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', palette='viridis', 
                       style='level', s=100, alpha=0.7)
        
        plt.title('Clusters Visualization (PCA)', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(title='Cluster', fontsize=12, title_fontsize=14)
        
        # Add annotations for some points (limit to avoid clutter)
        for i, row in plot_df.sample(min(20, len(plot_df))).iterrows():
            label = row['name'].split('_')[-1]  # Just the last part of the name
            plt.annotate(label, (row['x'], row['y']), fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('wordclouds/clusters/cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also create domain-colored plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=plot_df, x='x', y='y', hue='domain', palette='Set2', 
                       style='level', s=100, alpha=0.7)
        
        plt.title('Domain Visualization (PCA)', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(title='Domain', fontsize=12, title_fontsize=14)
        
        plt.tight_layout()
        plt.savefig('wordclouds/clusters/domain_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, filename: str):
        """
        Run complete embedding-based word cloud analysis
        
        Args:
            filename (str): Path to JSON file with data products
        """
        print("Loading data products...")
        data_products = self.load_data_products(filename)
        print(f"Loaded {len(data_products)} data products")
        
        print("\nExtracting text features...")
        df = self.extract_text_features(data_products)
        print(f"Extracted features for {len(df)} items:")
        print(df['level'].value_counts())
        
        print("\nGenerating domain-based word clouds...")
        self.create_domain_wordclouds(df)
        
        print("\nGenerating level-specific word clouds...")
        for level in ['product', 'dataset', 'schema']:
            self.create_domain_wordclouds(df, level)
            
        print("\nGenerating embedding-based cluster word clouds...")
        self.create_embedding_based_wordclouds(df)
        
        return df


def main():
    """Main function to run the embedding word cloud generator"""
    print("Initializing Embedding Word Cloud Generator...")
    generator = EmbeddingWordCloudGenerator(max_features=512, n_components=128)
    
    try:
        print("\nRunning embedding-based word cloud analysis...")
        df = generator.run_complete_analysis('financial_data_products.json')
        
        print("\nAnalysis complete!")
        print("Word cloud images saved to 'wordclouds' directory")
        
    except FileNotFoundError:
        print("Error: financial_data_products.json not found.")
        print("Please make sure the JSON file exists in the current directory.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
