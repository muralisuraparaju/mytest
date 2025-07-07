import json
import random
from typing import List, Dict, Any

class FinancialDataProductGenerator:
    def __init__(self):
        # Investment Banking subject areas
        self.investment_banking_subjects = [
            "equity_trading", "fixed_income_trading", "derivatives_trading", "fx_trading",
            "prime_brokerage", "securities_lending", "repo_transactions", "margin_lending",
            "ipo_underwriting", "bond_issuance", "syndicated_loans", "project_finance",
            "merger_acquisition", "leveraged_buyouts", "debt_restructuring", "capital_raising",
            "equity_research", "credit_research", "market_making", "algorithmic_trading",
            "high_frequency_trading", "dark_pools", "electronic_trading", "voice_trading",
            "settlement_processing", "clearing_operations", "custody_services", "corporate_actions",
            "dividend_processing", "interest_calculations", "accrual_management", "trade_lifecycle",
            "regulatory_reporting", "mifid_reporting", "dodd_frank_compliance", "basel_reporting",
            "capital_adequacy", "leverage_ratios", "liquidity_ratios", "stress_testing"
        ]
        
        # Wealth Management subject areas
        self.wealth_management_subjects = [
            "portfolio_management", "asset_allocation", "investment_advisory", "financial_planning",
            "retirement_planning", "estate_planning", "tax_optimization", "trust_services",
            "private_banking", "family_office", "ultra_high_net_worth", "mass_affluent",
            "robo_advisory", "digital_wealth", "hybrid_advisory", "goal_based_investing",
            "mutual_funds", "etf_management", "alternative_investments", "hedge_funds",
            "private_equity", "real_estate_investment", "commodities_investment", "structured_products",
            "client_onboarding", "kyc_processes", "suitability_assessment", "risk_profiling",
            "investment_proposals", "portfolio_reporting", "performance_attribution", "benchmark_analysis",
            "fee_calculation", "billing_management", "custody_reporting", "tax_reporting",
            "compliance_monitoring", "fiduciary_oversight", "investment_committee", "research_distribution"
        ]
        
        # Risk Management subject areas
        self.risk_management_subjects = [
            "market_risk", "credit_risk", "operational_risk", "liquidity_risk",
            "counterparty_risk", "concentration_risk", "model_risk", "reputational_risk",
            "var_calculations", "expected_shortfall", "stress_testing", "scenario_analysis",
            "credit_scoring", "probability_default", "loss_given_default", "exposure_at_default",
            "credit_limits", "collateral_management", "margin_requirements", "haircut_calculations",
            "risk_appetite", "risk_tolerance", "risk_limits", "risk_monitoring",
            "risk_reporting", "risk_dashboard", "risk_metrics", "risk_indicators",
            "basel_compliance", "ccar_testing", "dfast_analysis", "capital_planning",
            "liquidity_coverage", "net_stable_funding", "leverage_calculations", "rwa_calculations",
            "fraud_detection", "aml_monitoring", "sanctions_screening", "transaction_monitoring",
            "cyber_risk", "information_security", "business_continuity", "disaster_recovery",
            "vendor_risk", "third_party_risk", "governance_risk", "regulatory_risk"
        ]
        
        # Connection types and formats
        self.connection_types = ["ADLS", "RDBMS", "Kafka"]
        self.data_formats = [
            "parquet", "csv", "json", "avro", "delta", "orc", 
            "xml", "excel", "fixed_width", "delimited"
        ]
        
        # Common dataset patterns
        self.dataset_patterns = [
            "raw_data", "cleansed_data", "aggregated_data", "reference_data",
            "master_data", "transactional_data", "historical_data", "real_time_data",
            "derived_data", "enriched_data", "validated_data", "reconciled_data"
        ]
        
        # Port name patterns
        self.input_port_patterns = [
            "source_feed", "external_data", "upstream_system", "raw_input",
            "market_data_feed", "trade_data_feed", "reference_feed", "pricing_feed"
        ]
        
        self.output_port_patterns = [
            "processed_output", "analytics_output", "reporting_output", "downstream_feed",
            "risk_metrics", "performance_data", "compliance_report", "dashboard_data"
        ]

    def generate_subject_areas(self) -> List[str]:
        """Generate all subject areas across the three domains"""
        all_subjects = []
        all_subjects.extend(self.investment_banking_subjects)
        all_subjects.extend(self.wealth_management_subjects)
        all_subjects.extend(self.risk_management_subjects)
        return all_subjects

    def generate_physical_schema(self, subject: str, domain: str, pattern: str) -> List[Dict[str, str]]:
        """Generate physical schema columns for a dataset based on subject and pattern"""
        
        # Common columns across all financial datasets
        common_columns = [
            {"name": "record_id", "type": "string", "description": "Unique record identifier"},
            {"name": "source_system", "type": "string", "description": "Source system name"},
            {"name": "created_timestamp", "type": "timestamp", "description": "Record creation timestamp"},
            {"name": "updated_timestamp", "type": "timestamp", "description": "Record last update timestamp"},
            {"name": "business_date", "type": "date", "description": "Business date for the record"}
        ]
        
        # Subject-specific column mappings
        subject_columns = {
            # Investment Banking columns
            "equity_trading": [
                {"name": "trade_id", "type": "string", "description": "Unique trade identifier"},
                {"name": "symbol", "type": "string", "description": "Stock symbol"},
                {"name": "quantity", "type": "decimal", "description": "Number of shares traded"},
                {"name": "price", "type": "decimal", "description": "Trade execution price"},
                {"name": "trade_value", "type": "decimal", "description": "Total trade value"},
                {"name": "side", "type": "string", "description": "Buy or Sell indicator"},
                {"name": "trader_id", "type": "string", "description": "Trader identifier"},
                {"name": "client_id", "type": "string", "description": "Client identifier"},
                {"name": "execution_venue", "type": "string", "description": "Trading venue"},
                {"name": "order_type", "type": "string", "description": "Order type (Market, Limit, etc.)"}
            ],
            "fixed_income_trading": [
                {"name": "bond_id", "type": "string", "description": "Bond identifier"},
                {"name": "cusip", "type": "string", "description": "CUSIP identifier"},
                {"name": "isin", "type": "string", "description": "ISIN identifier"},
                {"name": "coupon_rate", "type": "decimal", "description": "Bond coupon rate"},
                {"name": "maturity_date", "type": "date", "description": "Bond maturity date"},
                {"name": "yield_to_maturity", "type": "decimal", "description": "Yield to maturity"},
                {"name": "duration", "type": "decimal", "description": "Modified duration"},
                {"name": "credit_rating", "type": "string", "description": "Credit rating"},
                {"name": "notional_amount", "type": "decimal", "description": "Notional amount traded"}
            ],
            "derivatives_trading": [
                {"name": "contract_id", "type": "string", "description": "Derivative contract ID"},
                {"name": "underlying_asset", "type": "string", "description": "Underlying asset symbol"},
                {"name": "contract_type", "type": "string", "description": "Contract type (Option, Future, Swap)"},
                {"name": "strike_price", "type": "decimal", "description": "Strike price"},
                {"name": "expiration_date", "type": "date", "description": "Contract expiration date"},
                {"name": "premium", "type": "decimal", "description": "Option premium"},
                {"name": "delta", "type": "decimal", "description": "Option delta"},
                {"name": "gamma", "type": "decimal", "description": "Option gamma"},
                {"name": "theta", "type": "decimal", "description": "Option theta"},
                {"name": "vega", "type": "decimal", "description": "Option vega"}
            ],
            "fx_trading": [
                {"name": "currency_pair", "type": "string", "description": "Currency pair (e.g., EUR/USD)"},
                {"name": "base_currency", "type": "string", "description": "Base currency"},
                {"name": "quote_currency", "type": "string", "description": "Quote currency"},
                {"name": "exchange_rate", "type": "decimal", "description": "Exchange rate"},
                {"name": "notional_amount", "type": "decimal", "description": "Notional amount"},
                {"name": "settlement_date", "type": "date", "description": "Settlement date"},
                {"name": "spot_rate", "type": "decimal", "description": "Spot exchange rate"},
                {"name": "forward_points", "type": "decimal", "description": "Forward points"}
            ],
            
            # Wealth Management columns
            "portfolio_management": [
                {"name": "portfolio_id", "type": "string", "description": "Portfolio identifier"},
                {"name": "client_id", "type": "string", "description": "Client identifier"},
                {"name": "asset_class", "type": "string", "description": "Asset class"},
                {"name": "security_id", "type": "string", "description": "Security identifier"},
                {"name": "position_value", "type": "decimal", "description": "Position market value"},
                {"name": "quantity", "type": "decimal", "description": "Position quantity"},
                {"name": "weight", "type": "decimal", "description": "Portfolio weight percentage"},
                {"name": "cost_basis", "type": "decimal", "description": "Cost basis"},
                {"name": "unrealized_pnl", "type": "decimal", "description": "Unrealized P&L"},
                {"name": "advisor_id", "type": "string", "description": "Investment advisor ID"}
            ],
            "asset_allocation": [
                {"name": "allocation_id", "type": "string", "description": "Allocation identifier"},
                {"name": "portfolio_id", "type": "string", "description": "Portfolio identifier"},
                {"name": "asset_class", "type": "string", "description": "Asset class"},
                {"name": "target_allocation", "type": "decimal", "description": "Target allocation percentage"},
                {"name": "actual_allocation", "type": "decimal", "description": "Actual allocation percentage"},
                {"name": "allocation_drift", "type": "decimal", "description": "Allocation drift"},
                {"name": "rebalance_threshold", "type": "decimal", "description": "Rebalancing threshold"},
                {"name": "strategic_allocation", "type": "decimal", "description": "Strategic allocation"}
            ],
            "financial_planning": [
                {"name": "plan_id", "type": "string", "description": "Financial plan identifier"},
                {"name": "client_id", "type": "string", "description": "Client identifier"},
                {"name": "goal_type", "type": "string", "description": "Financial goal type"},
                {"name": "target_amount", "type": "decimal", "description": "Target amount"},
                {"name": "current_value", "type": "decimal", "description": "Current value"},
                {"name": "monthly_contribution", "type": "decimal", "description": "Monthly contribution"},
                {"name": "expected_return", "type": "decimal", "description": "Expected annual return"},
                {"name": "time_horizon", "type": "integer", "description": "Time horizon in years"},
                {"name": "risk_tolerance", "type": "string", "description": "Risk tolerance level"}
            ],
            
            # Risk Management columns
            "market_risk": [
                {"name": "position_id", "type": "string", "description": "Position identifier"},
                {"name": "instrument_id", "type": "string", "description": "Instrument identifier"},
                {"name": "market_value", "type": "decimal", "description": "Market value"},
                {"name": "var_1d", "type": "decimal", "description": "1-day Value at Risk"},
                {"name": "var_10d", "type": "decimal", "description": "10-day Value at Risk"},
                {"name": "expected_shortfall", "type": "decimal", "description": "Expected Shortfall"},
                {"name": "beta", "type": "decimal", "description": "Beta coefficient"},
                {"name": "volatility", "type": "decimal", "description": "Volatility measure"},
                {"name": "correlation", "type": "decimal", "description": "Correlation coefficient"}
            ],
            "credit_risk": [
                {"name": "exposure_id", "type": "string", "description": "Exposure identifier"},
                {"name": "counterparty_id", "type": "string", "description": "Counterparty identifier"},
                {"name": "exposure_amount", "type": "decimal", "description": "Exposure amount"},
                {"name": "probability_of_default", "type": "decimal", "description": "Probability of default"},
                {"name": "loss_given_default", "type": "decimal", "description": "Loss given default"},
                {"name": "expected_loss", "type": "decimal", "description": "Expected loss"},
                {"name": "credit_rating", "type": "string", "description": "Credit rating"},
                {"name": "maturity_date", "type": "date", "description": "Maturity date"},
                {"name": "collateral_value", "type": "decimal", "description": "Collateral value"}
            ],
            "operational_risk": [
                {"name": "event_id", "type": "string", "description": "Risk event identifier"},
                {"name": "business_line", "type": "string", "description": "Business line"},
                {"name": "risk_category", "type": "string", "description": "Risk category"},
                {"name": "loss_amount", "type": "decimal", "description": "Loss amount"},
                {"name": "event_date", "type": "date", "description": "Event date"},
                {"name": "discovery_date", "type": "date", "description": "Discovery date"},
                {"name": "status", "type": "string", "description": "Event status"},
                {"name": "root_cause", "type": "string", "description": "Root cause analysis"},
                {"name": "recovery_amount", "type": "decimal", "description": "Recovery amount"}
            ],
            "liquidity_risk": [
                {"name": "asset_id", "type": "string", "description": "Asset identifier"},
                {"name": "liquidity_bucket", "type": "string", "description": "Liquidity time bucket"},
                {"name": "cash_flow_amount", "type": "decimal", "description": "Cash flow amount"},
                {"name": "maturity_date", "type": "date", "description": "Maturity date"},
                {"name": "liquidity_coverage_ratio", "type": "decimal", "description": "Liquidity coverage ratio"},
                {"name": "net_stable_funding_ratio", "type": "decimal", "description": "Net stable funding ratio"},
                {"name": "bid_ask_spread", "type": "decimal", "description": "Bid-ask spread"}
            ]
        }
        
        # Get subject-specific columns or use generic financial columns
        specific_columns = subject_columns.get(subject, [
            {"name": "entity_id", "type": "string", "description": f"{subject} entity identifier"},
            {"name": "amount", "type": "decimal", "description": f"{subject} amount"},
            {"name": "status", "type": "string", "description": f"{subject} status"},
            {"name": "category", "type": "string", "description": f"{subject} category"}
        ])
        
        # Combine common and specific columns
        all_columns = common_columns + specific_columns
        
        # Add pattern-specific columns
        pattern_columns = {
            "aggregated_data": [
                {"name": "aggregation_level", "type": "string", "description": "Aggregation level"},
                {"name": "period_start", "type": "date", "description": "Aggregation period start"},
                {"name": "period_end", "type": "date", "description": "Aggregation period end"},
                {"name": "record_count", "type": "integer", "description": "Number of records aggregated"}
            ],
            "historical_data": [
                {"name": "version_number", "type": "integer", "description": "Data version number"},
                {"name": "effective_date", "type": "date", "description": "Effective date"},
                {"name": "expiry_date", "type": "date", "description": "Expiry date"}
            ],
            "real_time_data": [
                {"name": "sequence_number", "type": "integer", "description": "Message sequence number"},
                {"name": "processing_timestamp", "type": "timestamp", "description": "Processing timestamp"},
                {"name": "latency_ms", "type": "integer", "description": "Processing latency in milliseconds"}
            ],
            "reference_data": [
                {"name": "reference_code", "type": "string", "description": "Reference code"},
                {"name": "reference_value", "type": "string", "description": "Reference value"},
                {"name": "is_active", "type": "boolean", "description": "Active indicator"}
            ]
        }
        
        # Add pattern-specific columns if applicable
        if pattern in pattern_columns:
            all_columns.extend(pattern_columns[pattern])
        
        return all_columns

    def generate_dataset(self, subject: str, domain: str) -> Dict[str, Any]:
        """Generate a dataset for a given subject and domain"""
        pattern = random.choice(self.dataset_patterns)
        
        descriptions = {
            "raw_data": f"Raw {subject} data collected from various sources",
            "cleansed_data": f"Cleansed and validated {subject} data",
            "aggregated_data": f"Aggregated {subject} metrics and summaries",
            "reference_data": f"Reference data for {subject} operations",
            "master_data": f"Master data repository for {subject}",
            "transactional_data": f"Transactional records for {subject}",
            "historical_data": f"Historical time series data for {subject}",
            "real_time_data": f"Real-time streaming data for {subject}",
            "derived_data": f"Derived analytics and calculations for {subject}",
            "enriched_data": f"Enriched {subject} data with additional context",
            "validated_data": f"Validated and quality-checked {subject} data",
            "reconciled_data": f"Reconciled {subject} data across systems"
        }
        
        return {
            "name": f"{domain}_{subject}_{pattern}",
            "description": descriptions[pattern],
            "physicalSchema": self.generate_physical_schema(subject, domain, pattern)
        }

    def generate_input_port(self, subject: str, domain: str) -> Dict[str, str]:
        """Generate an input port for a given subject and domain"""
        pattern = random.choice(self.input_port_patterns)
        connection_type = random.choice(self.connection_types)
        data_format = random.choice(self.data_formats)
        
        return {
            "name": f"{domain}_{subject}_{pattern}",
            "type": connection_type,
            "format": data_format
        }

    def generate_output_port(self, subject: str, domain: str) -> Dict[str, str]:
        """Generate an output port for a given subject and domain"""
        pattern = random.choice(self.output_port_patterns)
        connection_type = random.choice(self.connection_types)
        data_format = random.choice(self.data_formats)
        
        return {
            "name": f"{domain}_{subject}_{pattern}",
            "type": connection_type,
            "format": data_format
        }

    def get_domain_for_subject(self, subject: str) -> str:
        """Determine which domain a subject belongs to"""
        if subject in self.investment_banking_subjects:
            return "investment_banking"
        elif subject in self.wealth_management_subjects:
            return "wealth_management"
        elif subject in self.risk_management_subjects:
            return "risk_management"
        else:
            return "unknown"

    def generate_product_description(self, subject: str, domain: str) -> str:
        """Generate a description for the data product"""
        domain_descriptions = {
            "investment_banking": f"Investment banking data product for {subject.replace('_', ' ')} operations, providing comprehensive data management and analytics capabilities",
            "wealth_management": f"Wealth management data product for {subject.replace('_', ' ')} services, enabling client portfolio management and advisory functions",
            "risk_management": f"Risk management data product for {subject.replace('_', ' ')} assessment, supporting enterprise risk monitoring and compliance"
        }
        
        return domain_descriptions.get(domain, f"Financial services data product for {subject.replace('_', ' ')}")

    def generate_data_product(self, subject: str) -> Dict[str, Any]:
        """Generate a complete data product for a given subject"""
        domain = self.get_domain_for_subject(subject)
        
        # Generate multiple datasets (1-3 per product)
        num_datasets = random.randint(1, 3)
        datasets = [self.generate_dataset(subject, domain) for _ in range(num_datasets)]
        
        # Generate input ports (1-2 per product)
        num_input_ports = random.randint(1, 2)
        input_ports = [self.generate_input_port(subject, domain) for _ in range(num_input_ports)]
        
        # Generate output ports (1-3 per product)
        num_output_ports = random.randint(1, 3)
        output_ports = [self.generate_output_port(subject, domain) for _ in range(num_output_ports)]
        
        return {
            "info": {
                "name": f"{domain}_{subject}_product",
                "description": self.generate_product_description(subject, domain)
            },
            "datasets": datasets,
            "outputports": output_ports,
            "inputports": input_ports
        }

    def generate_test_data_products(self, num_products: int = None) -> List[Dict[str, Any]]:
        """Generate test data products for financial services"""
        all_subjects = self.generate_subject_areas()
        
        if num_products is None:
            num_products = len(all_subjects)
        elif num_products > len(all_subjects):
            print(f"Warning: Requested {num_products} products but only {len(all_subjects)} subjects available. Using all subjects.")
            num_products = len(all_subjects)
        
        # Select subjects (random sample if less than total available)
        if num_products < len(all_subjects):
            selected_subjects = random.sample(all_subjects, num_products)
        else:
            selected_subjects = all_subjects
        
        # Generate data products
        data_products = []
        for subject in selected_subjects:
            product = self.generate_data_product(subject)
            data_products.append(product)
        
        return data_products

    def save_to_file(self, data_products: List[Dict[str, Any]], filename: str = "financial_data_products.json"):
        """Save generated data products to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(data_products, f, indent=2)
        print(f"Generated {len(data_products)} data products and saved to {filename}")

    def print_summary(self, data_products: List[Dict[str, Any]]):
        """Print a summary of generated data products"""
        print(f"\n=== Financial Services Data Products Summary ===")
        print(f"Total products generated: {len(data_products)}")
        
        # Count by domain
        domain_counts = {}
        for product in data_products:
            product_name = product['info']['name']
            if 'investment_banking' in product_name:
                domain = 'Investment Banking'
            elif 'wealth_management' in product_name:
                domain = 'Wealth Management'
            elif 'risk_management' in product_name:
                domain = 'Risk Management'
            else:
                domain = 'Other'
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print("\nProducts by domain:")
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count}")
        
        # Show first 3 products as examples with schema details
        print(f"\nFirst 3 products (examples):")
        for i, product in enumerate(data_products[:3]):
            print(f"  {i+1}. {product['info']['name']}")
            print(f"     Description: {product['info']['description'][:80]}...")
            print(f"     Datasets: {len(product['datasets'])}, Input ports: {len(product['inputports'])}, Output ports: {len(product['outputports'])}")
            
            # Show schema details for first dataset
            if product['datasets']:
                first_dataset = product['datasets'][0]
                print(f"     Sample Dataset: {first_dataset['name']}")
                print(f"     Schema columns: {len(first_dataset['physicalSchema'])}")
                print(f"     Sample columns: {', '.join([col['name'] for col in first_dataset['physicalSchema'][:5]])}...")

def main():
    """Main function to demonstrate the data product generator"""
    generator = FinancialDataProductGenerator()
    
    # Generate all available data products (100+ subjects)
    print("Generating financial services data products...")
    all_products = generator.generate_test_data_products()
    
    # Print summary
    generator.print_summary(all_products)
    
    # Save to file
    generator.save_to_file(all_products)
    
    # Generate a smaller sample (e.g., 20 products)
    print(f"\n=== Generating sample of 20 products ===")
    sample_products = generator.generate_test_data_products(20)
    generator.print_summary(sample_products)
    generator.save_to_file(sample_products, "sample_financial_data_products.json")

if __name__ == "__main__":
    main()
