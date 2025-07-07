import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
from typing import List, Dict, Any, Tuple, Optional

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


class DataProductRecommender:
    """
    A recommender system that uses embeddings to match natural language queries
    to relevant financial data products and datasets.
    """
    
    def __init__(self, max_features=512, n_components=128):
        """
        Initialize the recommender with TF-IDF parameters.
        
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
        
        # For dimensionality reduction - using SVD for sparse matrix compatibility
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Will store our data products dataframe
        self.data_df = None
        
        # Will store our embeddings
        self.embeddings = None
        
        # Keywords and their mapping to financial concepts for query enhancement
        self.concept_keywords = {
            'risk_averse': ['low risk', 'risk mitigation', 'conservative', 'stable', 
                          'protection', 'secure', 'safety', 'preservation', 'defensive'],
            'risk_seeking': ['high risk', 'aggressive', 'growth', 'volatile', 
                           'opportunistic', 'speculative', 'high yield', 'emerging'],
            'affluent': ['high net worth', 'wealth', 'premium', 'luxury', 'elite',
                       'sophisticated', 'private banking', 'exclusive', 'personalized'],
            'retail': ['mass market', 'consumer', 'individual', 'personal', 'small', 
                     'basic', 'standard', 'everyday'],
            'investment': ['portfolio', 'returns', 'asset allocation', 'securities',
                         'stocks', 'bonds', 'funds', 'opportunities', 'strategies'],
            'retirement': ['pension', 'long term', 'savings', 'future', 'nest egg',
                         'income', 'post-career', '401k', 'ira']
        }
        
        print(f"Initialized with {max_features} TF-IDF features and {n_components} SVD components")

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
                'full_text': product_text,
                'entity_id': f"product_{product_name.replace(' ', '_').lower()}"
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
                    'product_name': product_name,
                    'entity_id': f"dataset_{dataset_name.replace(' ', '_').lower()}"
                })
                
                # Process schema fields if available
                if 'physicalSchema' in dataset:
                    # Combine all field information for this dataset's schema
                    schema_fields = []
                    schema_texts = []
                    
                    for field in dataset['physicalSchema']:
                        field_name = field['name']
                        field_type = field['type']
                        field_desc = field['description']
                        
                        schema_fields.append({
                            'name': field_name,
                            'type': field_type,
                            'description': field_desc
                        })
                        
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
                            'dataset_name': dataset_name,
                            'schema_fields': schema_fields,
                            'entity_id': f"schema_{dataset_name.replace(' ', '_').lower()}"
                        })
        
        return pd.DataFrame(records)
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate TF-IDF embeddings for the text data
        
        Args:
            df (pd.DataFrame): DataFrame with processed text
            
        Returns:
            np.ndarray: Embeddings matrix
        """
        if 'processed_text' not in df.columns:
            raise ValueError("DataFrame must contain 'processed_text' column")
            
        # Fit and transform the processed text
        X = self.vectorizer.fit_transform(df['processed_text'])
        
        # Apply dimensionality reduction
        X_reduced = self.svd.fit_transform(X)
        
        return X_reduced
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the query by identifying key concepts and expanding with related terms
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Enhanced query with expanded concepts
        """
        # Lowercase the query for matching
        query_lower = query.lower()
        
        # Initialize enhanced terms with the original query
        enhanced_terms = [query]
        
        # Check for each concept in the query and add related terms
        for concept, related_terms in self.concept_keywords.items():
            # Check if the concept or any of its terms are in the query
            concept_present = concept.replace('_', ' ') in query_lower
            terms_present = any(term.lower() in query_lower for term in related_terms)
            
            if concept_present or terms_present:
                # Add all related terms to enhance the query
                enhanced_terms.extend(related_terms)
        
        # Join all terms into a single enhanced query
        enhanced_query = ' '.join(enhanced_terms)
        
        return enhanced_query
    
    def load_and_prepare_data(self, filename: str) -> None:
        """
        Load data products and prepare embeddings
        
        Args:
            filename (str): Path to JSON file with data products
        """
        print("Loading data products...")
        data_products = self.load_data_products(filename)
        print(f"Loaded {len(data_products)} data products")
        
        print("\nExtracting text features...")
        self.data_df = self.extract_text_features(data_products)
        print(f"Extracted features for {len(self.data_df)} items:")
        print(self.data_df['level'].value_counts())
        
        print("\nGenerating embeddings...")
        self.embeddings = self.generate_embeddings(self.data_df)
        print(f"Generated embeddings of shape {self.embeddings.shape}")
        
        # Save DataFrame with original data but not embeddings (too large for CSV)
        self.data_df.to_csv('data_products_features.csv', index=False)
        print("Saved data features to data_products_features.csv")
    
    def find_relevant_data_products(self, query: str, top_k: int = 10, 
                                  level_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Find relevant data products for a given query
        
        Args:
            query (str): User query
            top_k (int): Number of top results to return
            level_filter (str, optional): Filter by level ('product', 'dataset', 'schema')
            
        Returns:
            pd.DataFrame: DataFrame with relevant products
        """
        if self.data_df is None or self.embeddings is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data first.")
            
        # Enhance the query with related terms
        enhanced_query = self.enhance_query(query)
        print(f"Enhanced query: {enhanced_query}")
        
        # Preprocess the query
        processed_query = self.preprocess_text(enhanced_query)
        print(f"Preprocessed query: {processed_query}")
        
        # Create query embedding
        query_vec = self.vectorizer.transform([processed_query])
        query_embedding = self.svd.transform(query_vec)
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Create results DataFrame
        results_df = self.data_df.copy()
        results_df['similarity_score'] = similarities
        
        # Apply level filter if specified
        if level_filter:
            results_df = results_df[results_df['level'] == level_filter]
            
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Return top k results
        return results_df.head(top_k)
    
    def analyze_query_intent(self, query: str) -> Dict[str, float]:
        """
        Analyze the intent of the query to determine what types of data products might be relevant
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, float]: Dictionary of intent categories and their scores
        """
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Define intent categories and their keywords
        intent_categories = {
            'investment_strategy': [
                'strategy', 'invest', 'portfolio', 'allocation', 'diversification',
                'opportunities', 'returns', 'performance'
            ],
            'risk_assessment': [
                'risk', 'averse', 'conservative', 'safe', 'protection', 'downside',
                'volatility', 'stability', 'secure'
            ],
            'customer_segmentation': [
                'affluent', 'wealthy', 'high net worth', 'client', 'customer',
                'segment', 'demographic', 'profile', 'target'
            ],
            'product_development': [
                'create', 'develop', 'build', 'product', 'service', 'offering',
                'solution', 'design', 'launch'
            ],
            'market_analysis': [
                'market', 'trend', 'analysis', 'research', 'industry', 'sector',
                'competition', 'outlook', 'forecast'
            ]
        }
        
        # Calculate intent scores
        intent_scores = {}
        query_tokens = processed_query.split()
        
        for category, keywords in intent_categories.items():
            # Count matches between query tokens and category keywords
            matches = sum(1 for token in query_tokens if any(
                keyword in token or token in keyword for keyword in keywords
            ))
            
            # Calculate score as percentage of matched tokens
            score = matches / len(query_tokens) if query_tokens else 0
            intent_scores[category] = score
            
        # Normalize scores so they sum to 1
        total_score = sum(intent_scores.values())
        if total_score > 0:
            for category in intent_scores:
                intent_scores[category] /= total_score
                
        return intent_scores
    
    def recommend_data_products(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recommend data products based on a natural language query
        
        Args:
            query (str): User query
            top_k (int): Number of top results to return for each category
            
        Returns:
            Dict[str, Any]: Dictionary with recommendations
        """
        # Analyze query intent
        intent_scores = self.analyze_query_intent(query)
        print("Query Intent Analysis:")
        for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent}: {score:.2f}")
            
        # Find relevant data across all levels
        all_results = self.find_relevant_data_products(query, top_k=top_k*3)
        
        # Get top results by level
        products = all_results[all_results['level'] == 'product'].head(top_k)
        datasets = all_results[all_results['level'] == 'dataset'].head(top_k)
        
        # For schema, get the top 2*top_k to ensure we have enough after filtering
        schemas = all_results[all_results['level'] == 'schema'].head(top_k*2)
        
        # Determine primary domains from the query results
        domain_counts = all_results.head(top_k*2)['domain'].value_counts()
        primary_domains = domain_counts.index[:3].tolist()  # Top 3 domains
        
        recommendations = {
            'query': query,
            'intent_analysis': {
                'scores': intent_scores,
                'primary_intent': max(intent_scores.items(), key=lambda x: x[1])[0]
            },
            'primary_domains': primary_domains,
            'recommended_products': products.to_dict('records'),
            'recommended_datasets': datasets.to_dict('records'),
            'relevant_schemas': schemas.to_dict('records'),
            'summary': self._generate_recommendation_summary(query, intent_scores, primary_domains, products)
        }
        
        return recommendations
    
    def _generate_recommendation_summary(self, query: str, intent_scores: Dict[str, float],
                                       primary_domains: List[str], products: pd.DataFrame) -> str:
        """
        Generate a summary of the recommendations
        
        Args:
            query (str): Original user query
            intent_scores (Dict[str, float]): Intent analysis scores
            primary_domains (List[str]): Primary domains identified
            products (pd.DataFrame): Top recommended products
            
        Returns:
            str: Summary text
        """
        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0].replace('_', ' ')
        
        # Get top product names
        top_product_names = products['name'].tolist()[:3]
        
        # Format domains as string
        domains_str = ', '.join(primary_domains)
        
        # Generate summary
        summary = (
            f"Based on your query about '{query}', I've identified that you're primarily interested "
            f"in {primary_intent}. The most relevant financial domains for this query are {domains_str}.\n\n"
            f"Top recommended data products include: {', '.join(top_product_names)}."
        )
        
        return summary
    
    def format_recommendations_as_markdown(self, recommendations: Dict[str, Any]) -> str:
        """
        Format recommendations as markdown for display
        
        Args:
            recommendations (Dict[str, Any]): Recommendations from recommend_data_products
            
        Returns:
            str: Markdown formatted recommendations
        """
        markdown = [
            f"# Recommendations for: \"{recommendations['query']}\"",
            "",
            f"## Summary",
            recommendations['summary'],
            "",
            f"## Primary Domains",
            "- " + "\n- ".join(recommendations['primary_domains']),
            "",
            "## Query Intent Analysis",
            "| Intent | Score |",
            "| ------ | ----- |",
        ]
        
        # Add intent scores
        for intent, score in sorted(recommendations['intent_analysis']['scores'].items(), 
                                 key=lambda x: x[1], reverse=True):
            markdown.append(f"| {intent.replace('_', ' ')} | {score:.2f} |")
            
        # Add recommended products
        markdown.extend([
            "",
            "## Recommended Data Products",
            "| Name | Domain | Relevance |",
            "| ---- | ------ | --------- |",
        ])
        
        for product in recommendations['recommended_products']:
            markdown.append(
                f"| {product['name']} | {product['domain']} | {product['similarity_score']:.2f} |"
            )
            
        # Add recommended datasets
        markdown.extend([
            "",
            "## Recommended Datasets",
            "| Name | Product | Domain | Relevance |",
            "| ---- | ------- | ------ | --------- |",
        ])
        
        for dataset in recommendations['recommended_datasets']:
            product_name = dataset.get('product_name', 'N/A')
            markdown.append(
                f"| {dataset['name']} | {product_name} | {dataset['domain']} | {dataset['similarity_score']:.2f} |"
            )
            
        return '\n'.join(markdown)


def main():
    """Main function to demonstrate the recommender system"""
    print("Initializing Data Product Recommender...")
    recommender = DataProductRecommender(max_features=512, n_components=128)
    
    try:
        # Load and prepare data
        print("\nLoading and preparing data...")
        recommender.load_and_prepare_data('financial_data_products.json')
        
        # Example query
        example_query = "I want to create a data product that suggests investment opportunities for risk averse affluent investors"
        print(f"\nExample query: '{example_query}'")
        
        # Get recommendations
        recommendations = recommender.recommend_data_products(example_query)
        
        # Format as markdown
        markdown_output = recommender.format_recommendations_as_markdown(recommendations)
        
        # Print recommendations in markdown format
        print("\nRecommendations:")
        print(markdown_output)
        
        # Save markdown output to file
        with open('recommendations_output.md', 'w') as f:
            f.write(markdown_output)
        print("\nRecommendations saved to 'recommendations_output.md'")
        
        # Interactive mode
        print("\n" + "="*50)
        print("Entering interactive mode. Type 'exit' to quit.")
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
                
            # Get and display recommendations
            recommendations = recommender.recommend_data_products(query)
            markdown_output = recommender.format_recommendations_as_markdown(recommendations)
            print("\nRecommendations:")
            print(markdown_output)
            
    except FileNotFoundError:
        print("Error: financial_data_products.json not found.")
        print("Please make sure the JSON file exists in the current directory.")
    except Exception as e:
        print(f"Error during recommendation: {str(e)}")


if __name__ == "__main__":
    main()
