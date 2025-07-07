import json
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from typing import List, Dict, Tuple, Any

# Download required NLTK data (run once)
try:
    print("Downloading NLTK data...")
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    print("Downloading NLTK data...")
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    print("Downloading NLTK data...")
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FinancialDataAnalyzer:
    def __init__(self):
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
        
    def load_data_products(self, filename: str) -> List[Dict[str, Any]]:
        """Load data products from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
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
    
    def extract_text_features(self, data_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract text features from data products"""
        records = []
        
        for product in data_products:
            # Extract product information
            product_name = product['info']['name']
            product_desc = product['info']['description']
            
            # Extract dataset information
            for dataset in product['datasets']:
                dataset_name = dataset['name']
                dataset_desc = dataset['description']
                
                # Combine all text
                combined_text = f"{product_name} {product_desc} {dataset_name} {dataset_desc}"
                processed_text = self.preprocess_text(combined_text)
                
                # Determine domain
                domain = self.extract_domain(product_name)
                
                records.append({
                    'product_name': product_name,
                    'product_description': product_desc,
                    'dataset_name': dataset_name,
                    'dataset_description': dataset_desc,
                    'combined_text': processed_text,
                    'domain': domain,
                    'original_combined': combined_text
                })
        
        return pd.DataFrame(records)
    
    def extract_domain(self, product_name: str) -> str:
        """Extract domain from product name"""
        if 'investment_banking' in product_name:
            return 'Investment Banking'
        elif 'wealth_management' in product_name:
            return 'Wealth Management'
        elif 'risk_management' in product_name:
            return 'Risk Management'
        else:
            return 'Other'
    
    def generate_wordcloud(self, text_data: str, title: str, save_path: str = None):
        """Generate and display word cloud"""
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text_data)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_multiple_wordclouds(self, df: pd.DataFrame):
        """Create word clouds for different categories"""
        
        # Overall word cloud
        all_text = ' '.join(df['combined_text'])
        self.generate_wordcloud(all_text, 'Overall Financial Data Products Word Cloud', 
                              'overall_wordcloud.png')
        
        # Domain-specific word clouds
        for domain in df['domain'].unique():
            domain_text = ' '.join(df[df['domain'] == domain]['combined_text'])
            self.generate_wordcloud(domain_text, f'{domain} Word Cloud', 
                                  f'{domain.lower().replace(" ", "_")}_wordcloud.png')
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Perform K-means clustering on text data"""
        
        # Vectorize text data
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(df['combined_text'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Get cluster centers and top terms
        cluster_info = self.analyze_clusters(kmeans, vectorizer, feature_names, df_clustered)
        
        return clusters, df_clustered, cluster_info
    
    def find_optimal_clusters(self, X, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using silhouette score"""
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, X.shape[0]))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find the number of clusters with the highest silhouette score
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_clusters
    
    def analyze_clusters(self, kmeans, vectorizer, feature_names, df_clustered) -> pd.DataFrame:
        """Analyze cluster characteristics"""
        cluster_info = []
        
        for cluster_id in range(kmeans.n_clusters):
            # Get cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Get top terms for this cluster
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [cluster_center[i] for i in top_indices]
            
            # Get cluster data
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            # Count datasets and products
            n_datasets = len(cluster_data)
            n_products = cluster_data['product_name'].nunique()
            
            # Determine cluster theme based on top terms
            theme = self.determine_cluster_theme(top_terms)
            
            # Domain distribution
            domain_dist = cluster_data['domain'].value_counts().to_dict()
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'theme': theme,
                'top_terms': top_terms[:5],
                'top_scores': [round(score, 3) for score in top_scores[:5]],
                'n_datasets': n_datasets,
                'n_products': n_products,
                'domain_distribution': domain_dist,
                'sample_datasets': cluster_data['dataset_name'].head(3).tolist()
            })
        
        return pd.DataFrame(cluster_info)
    
    def determine_cluster_theme(self, top_terms: List[str]) -> str:
        """Determine cluster theme based on top terms"""
        
        # Define theme keywords
        theme_keywords = {
            'Trading & Execution': ['trading', 'trade', 'execution', 'market', 'order', 'price', 'equity', 'bond'],
            'Risk Management': ['risk', 'var', 'credit', 'market', 'operational', 'liquidity', 'exposure'],
            'Portfolio Management': ['portfolio', 'asset', 'allocation', 'investment', 'client', 'advisor'],
            'Regulatory & Compliance': ['regulatory', 'compliance', 'reporting', 'basel', 'mifid', 'dodd'],
            'Derivatives & Complex Products': ['derivative', 'option', 'future', 'swap', 'structured', 'complex'],
            'Client Services': ['client', 'customer', 'service', 'advisor', 'relationship', 'onboarding'],
            'Operations & Settlement': ['settlement', 'clearing', 'custody', 'operation', 'processing', 'transaction'],
            'Analytics & Reporting': ['analytics', 'calculation', 'metric', 'performance', 'reporting', 'dashboard']
        }
        
        # Score each theme
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for term in top_terms if any(keyword in term for keyword in keywords))
            theme_scores[theme] = score
        
        # Return theme with highest score
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
            if theme_scores[best_theme] > 0:
                return best_theme
        
        return 'General Financial Services'
    
    def visualize_clusters(self, df_clustered: pd.DataFrame, cluster_info: pd.DataFrame):
        """Visualize cluster analysis results"""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster size distribution
        cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values)
        axes[0, 0].set_title('Cluster Size Distribution (Datasets)')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Datasets')
        
        # 2. Domain distribution across clusters
        domain_cluster = pd.crosstab(df_clustered['domain'], df_clustered['cluster'])
        domain_cluster.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Domain Distribution Across Clusters')
        axes[0, 1].set_xlabel('Domain')
        axes[0, 1].set_ylabel('Number of Datasets')
        axes[0, 1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Products vs Datasets per cluster
        axes[1, 0].scatter(cluster_info['n_products'], cluster_info['n_datasets'], 
                          s=100, alpha=0.7, c=cluster_info['cluster_id'], cmap='viridis')
        axes[1, 0].set_title('Products vs Datasets per Cluster')
        axes[1, 0].set_xlabel('Number of Products')
        axes[1, 0].set_ylabel('Number of Datasets')
        
        # Add cluster labels
        for i, row in cluster_info.iterrows():
            axes[1, 0].annotate(f'C{row["cluster_id"]}', 
                              (row['n_products'], row['n_datasets']),
                              xytext=(5, 5), textcoords='offset points')
        
        # 4. Cluster themes
        themes = [theme.split(' ')[0] for theme in cluster_info['theme']]  # Shorten for display
        axes[1, 1].barh(range(len(themes)), cluster_info['n_datasets'])
        axes[1, 1].set_yticks(range(len(themes)))
        axes[1, 1].set_yticklabels(themes)
        axes[1, 1].set_title('Datasets per Cluster Theme')
        axes[1, 1].set_xlabel('Number of Datasets')
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_cluster_summary(self, cluster_info: pd.DataFrame):
        """Print detailed cluster summary"""
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS SUMMARY")
        print("="*80)
        
        total_datasets = cluster_info['n_datasets'].sum()
        total_products = cluster_info['n_products'].sum()
        
        print(f"Total Datasets: {total_datasets}")
        print(f"Total Products: {total_products}")
        print(f"Number of Clusters: {len(cluster_info)}")
        
        print("\nCLUSTER DETAILS:")
        print("-" * 80)
        
        for _, cluster in cluster_info.iterrows():
            print(f"\nCluster {cluster['cluster_id']}: {cluster['theme']}")
            print(f"  Datasets: {cluster['n_datasets']}")
            print(f"  Products: {cluster['n_products']}")
            print(f"  Top Terms: {', '.join(cluster['top_terms'])}")
            print(f"  Domain Distribution: {cluster['domain_distribution']}")
            print(f"  Sample Datasets: {', '.join(cluster['sample_datasets'])}")
    
    def run_complete_analysis(self, filename: str):
        """Run complete analysis pipeline"""
        print("Loading data products...")
        data_products = self.load_data_products(filename)
        
        print("Extracting text features...")
        df = self.extract_text_features(data_products)
        
        print(f"Analyzed {len(df)} datasets from {df['product_name'].nunique()} products")
        
        print("\nGenerating word clouds...")
        self.create_multiple_wordclouds(df)
        
        print("\nPerforming clustering analysis...")
        clusters, df_clustered, cluster_info = self.perform_clustering(df)
        
        print("\nVisualizing results...")
        self.visualize_clusters(df_clustered, cluster_info)
        
        print("\nGenerating summary...")
        self.print_cluster_summary(cluster_info)
        
        return df_clustered, cluster_info

def main():
    """Main function to run the analysis"""
    analyzer = FinancialDataAnalyzer()
    
    # Run analysis on the generated data
    try:
        df_clustered, cluster_info = analyzer.run_complete_analysis('financial_data_products.json')
        
        # Save results
        df_clustered.to_csv('clustered_datasets.csv', index=False)
        cluster_info.to_csv('cluster_summary.csv', index=False)
        
        print("\nResults saved to:")
        print("- clustered_datasets.csv")
        print("- cluster_summary.csv")
        print("- Word cloud images")
        print("- cluster_analysis.png")
        
    except FileNotFoundError:
        print("Error: financial_data_products.json not found.")
        print("Please run the data product generator first to create the input file.")

if __name__ == "__main__":
    main()
