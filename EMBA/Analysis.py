# Analysis.py (Fixed - No Syntax Errors)
"""
Multi-Sector Bubble Analyzer - Core analysis logic
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  pandas as pd
import  numpy as np
from    scipy import stats
from    datetime import datetime
import  json
import  os
import  hashlib
import  Tools
from    Tools   import Display_info

class MultiSectorBubbleAnalyzer:
    def __init__(self, cache_dir=Tools.analysis_cache_path):
        self.metrics_framework = self._initialize_metrics_framework()
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _get_cache_key(self, data):
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key):
        """Load analysis from text cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
        
    def _save_to_cache(self, cache_key, analysis, app):
        """Save analysis to text cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        try:
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
                
            with open(cache_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=convert_datetime)
            return True
        except Exception as e:
            Display_info(f"Warning: Could not save to cache: {e}", app)
            return False

    def _initialize_metrics_framework(self):
        """Define comprehensive bubble detection framework"""
        return {
            'valuation': {
                'weight': 0.35,
                'metrics': {
                    'pe_ratio': {'weight': 0.25, 'direction': 'positive'},
                    'price_to_sales': {'weight': 0.25, 'direction': 'positive'},
                    'price_to_book': {'weight': 0.20, 'direction': 'positive'},
                    'ev_to_ebitda': {'weight': 0.15, 'direction': 'positive'},
                    'market_cap_to_gdp': {'weight': 0.15, 'direction': 'positive'}
                }
            },
            'momentum': {
                'weight': 0.25,
                'metrics': {
                    'price_momentum_1y': {'weight': 0.30, 'direction': 'positive'},
                    'volume_momentum': {'weight': 0.25, 'direction': 'positive'},
                    'volatility': {'weight': 0.25, 'direction': 'positive'},
                    'relative_strength': {'weight': 0.20, 'direction': 'positive'}
                }
            },
            'sentiment': {
                'weight': 0.20,
                'metrics': {
                    'short_interest': {'weight': 0.35, 'direction': 'positive'},
                    'institutional_ownership': {'weight': 0.35, 'direction': 'negative'},
                    'analyst_recommendations': {'weight': 0.30, 'direction': 'positive'}
                }
            },
            'fundamental': {
                'weight': 0.20,
                'metrics': {
                    'profit_margin': {'weight': 0.25, 'direction': 'negative'},
                    'return_on_equity': {'weight': 0.25, 'direction': 'negative'},
                    'earnings_growth': {'weight': 0.25, 'direction': 'positive'},
                    'revenue_growth': {'weight': 0.25, 'direction': 'positive'}
                }
            }
        }
    
    async def analyze_all_sectors(self, sector_data, app):
        """Analyze bubble risks across all sectors with caching"""
        cache_key = self._get_cache_key(sector_data)
        
        # Try to load from cache
        cached_analysis = self._load_from_cache(cache_key)
        if cached_analysis:
            Display_info("Loading analysis from cache...", app)
            # Convert string timestamps back to datetime objects
            for sector_name, analysis in cached_analysis.items():
                if 'timestamp' in analysis and isinstance(analysis['timestamp'], str):
                    cached_analysis[sector_name]['timestamp'] = datetime.fromisoformat(analysis['timestamp'])
            return cached_analysis
      
        # Perform fresh analysis
        sector_analysis = {}
        
        for sector_name, sector_info in sector_data.items():
            app.Tag_OK_Last_Line_N()
            Display_info(f"Analyzing {sector_name} sector...", app)
            sector_analysis[sector_name] = await self._analyze_single_sector(sector_info)
        
        # Calculate relative bubble risks
        sector_analysis = self._calculate_relative_risks(sector_analysis)
        
        # Save to cache
        self._save_to_cache(cache_key, sector_analysis, app)
        app.Tag_OK_Last_Line_N()
        Display_info("Analysis saved to cache", app)
        
        return sector_analysis
    
    async def _analyze_single_sector(self, sector_info):
        """Comprehensive analysis for a single sector"""
        analysis = {
            'sector_name': sector_info['metadata']['sector_name'],
            'timestamp': datetime.now(),
            'metrics': {},
            'composite_scores': {},
            'risk_assessment': {}
        }
        
        # Calculate all metric categories
        analysis['metrics']['valuation'] = self._calculate_valuation_metrics(sector_info)
        analysis['metrics']['momentum'] = self._calculate_momentum_metrics(sector_info)
        analysis['metrics']['sentiment'] = self._calculate_sentiment_metrics(sector_info)
        analysis['metrics']['fundamental'] = self._calculate_fundamental_metrics(sector_info)
        
        # Calculate composite scores
        analysis['composite_scores'] = self._calculate_composite_scores(analysis['metrics'])
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_sector_risk(analysis['composite_scores'])
        
        return analysis
    
    def _calculate_valuation_metrics(self, sector_info):
        """Calculate comprehensive valuation metrics using REAL data"""
        stocks = sector_info['stocks']
        
        if not stocks:
            return self._create_empty_metrics()
        
        # Extract valuation metrics from real data
        valuation_data = []
        for ticker, data in stocks.items():
            if 'valuation' in data:
                valuation_data.append(data['valuation'])
        
        if not valuation_data:
            return self._create_empty_metrics()
        
        valuation_df = pd.DataFrame(valuation_data)
        
        # Calculate sector aggregates using ACTUAL data
        metrics = {
            'pe_ratio': self._safe_median(valuation_df['pe_ratio']),
            'price_to_sales': self._safe_median(valuation_df['price_to_sales']),
            'price_to_book': self._safe_median(valuation_df['price_to_book']),
            'ev_to_ebitda': self._safe_median(valuation_df['ev_to_ebitda']),
            'market_cap_to_gdp': self._calculate_real_mcap_to_gdp(sector_info)
        }
        
        # Calculate percentiles based on realistic historical ranges
        percentiles = {}
        historical_ranges = {
            'pe_ratio': (8, 40),
            'price_to_sales': (1, 12),
            'price_to_book': (1, 8),
            'ev_to_ebitda': (6, 25),
            'market_cap_to_gdp': (0.1, 3.0)
        }
        
        for metric, value in metrics.items():
            if value and not np.isnan(value):
                if metric in historical_ranges:
                    min_val, max_val = historical_ranges[metric]
                    if max_val > min_val:
                        percentile = (value - min_val) / (max_val - min_val)
                        percentiles[metric] = max(0, min(1, percentile))
                    else:
                        percentiles[metric] = 0.5
                else:
                    percentiles[metric] = 0.5
            else:
                percentiles[metric] = 0.5
        
        return {
            'raw_metrics': metrics,
            'percentiles': percentiles,
            'composite_score': np.mean(list(percentiles.values()))
        }
    
    def _calculate_real_mcap_to_gdp(self, sector_info):
        """Calculate real market cap to GDP ratio"""
        stocks = sector_info['stocks']
        total_mcap = 0
        valid_stocks = 0
        
        for stock_data in stocks.values():
            mcap = stock_data['valuation'].get('market_cap', 0)
            if mcap and mcap > 0:
                total_mcap += mcap
                valid_stocks += 1
        
        if valid_stocks == 0:
            return 1.0
            
        avg_sector_mcap = total_mcap / valid_stocks
        
        # Use US GDP as reference (approx $25T)
        us_gdp = 25.0e12
        return (avg_sector_mcap / us_gdp) * 1000
    
    def _calculate_momentum_metrics(self, sector_info):
        """Calculate momentum and technical indicators using REAL price data"""
        stocks = sector_info['stocks']
        
        if not stocks:
            return self._create_empty_metrics()
        
        momentum_data = []
        for ticker, data in stocks.items():
            if 'historical' in data and not data['historical'].empty:
                prices = data['historical']['Close']
                volumes = data['historical']['Volume'] if 'Volume' in data['historical'].columns else None
                
                if len(prices) > 252:
                    momentum_metrics = {
                        'price_momentum_1y': self._calculate_price_momentum(prices),
                        'volume_momentum': self._calculate_volume_momentum(prices, volumes) if volumes is not None else 0.5,
                        'volatility': self._calculate_volatility(prices),
                        'relative_strength': self._calculate_relative_strength(prices)
                    }
                    momentum_data.append(momentum_metrics)
        
        if not momentum_data:
            return self._create_empty_metrics()
        
        momentum_df = pd.DataFrame(momentum_data)
        
        # Aggregate sector momentum using median to reduce outlier impact
        metrics = {
            'price_momentum_1y': momentum_df['price_momentum_1y'].median(),
            'volume_momentum': momentum_df['volume_momentum'].median(),
            'volatility': momentum_df['volatility'].median(),
            'relative_strength': momentum_df['relative_strength'].median()
        }
        
        # Calculate percentiles based on realistic ranges
        percentiles = {}
        momentum_ranges = {
            'price_momentum_1y': (-0.3, 0.8),
            'volume_momentum': (0.5, 3.0),
            'volatility': (0.15, 0.60),
            'relative_strength': (0.2, 0.9)
        }
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                if metric in momentum_ranges:
                    min_val, max_val = momentum_ranges[metric]
                    if max_val > min_val:
                        percentile = (value - min_val) / (max_val - min_val)
                        percentiles[metric] = max(0, min(1, percentile))
                    else:
                        percentiles[metric] = 0.5
                else:
                    percentiles[metric] = 0.5
            else:
                percentiles[metric] = 0.5
        
        return {
            'raw_metrics': metrics,
            'percentiles': percentiles,
            'composite_score': np.mean(list(percentiles.values()))
        }
    
    def _calculate_sentiment_metrics(self, sector_info):
        """Calculate market sentiment indicators using REAL data"""
        stocks = sector_info['stocks']
        
        if not stocks:
            return self._create_empty_metrics()
        
        sentiment_data = []
        for ticker, data in stocks.items():
            info = data.get('info', {})
            
            short_interest = info.get('shortPercentOfFloat', 0) or info.get('shortPercent', 0) or 0
            institutional_ownership = info.get('heldPercentInstitutions', 0) or 0
            analyst_recommendations = self._calculate_analyst_sentiment(info)
            
            sentiment_metrics = {
                'short_interest': short_interest,
                'institutional_ownership': institutional_ownership,
                'analyst_recommendations': analyst_recommendations
            }
            sentiment_data.append(sentiment_metrics)
        
        if not sentiment_data:
            return self._create_empty_metrics()
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Aggregate sector sentiment
        metrics = {
            'short_interest': sentiment_df['short_interest'].median(),
            'institutional_ownership': sentiment_df['institutional_ownership'].median(),
            'analyst_recommendations': sentiment_df['analyst_recommendations'].median()
        }
        
        # Calculate percentiles based on realistic ranges
        percentiles = {}
        sentiment_ranges = {
            'short_interest': (0.01, 0.20),
            'institutional_ownership': (0.3, 0.95),
            'analyst_recommendations': (1.0, 5.0)
        }
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                if metric in sentiment_ranges:
                    min_val, max_val = sentiment_ranges[metric]
                    if max_val > min_val:
                        if metric == 'institutional_ownership':
                            percentile = 1 - ((value - min_val) / (max_val - min_val))
                        else:
                            percentile = (value - min_val) / (max_val - min_val)
                        percentiles[metric] = max(0, min(1, percentile))
                    else:
                        percentiles[metric] = 0.5
                else:
                    percentiles[metric] = 0.5
            else:
                percentiles[metric] = 0.5
        
        return {
            'raw_metrics': metrics,
            'percentiles': percentiles,
            'composite_score': np.mean(list(percentiles.values()))
        }
    
    def _calculate_fundamental_metrics(self, sector_info):
        """Calculate fundamental health metrics using REAL data"""
        stocks = sector_info['stocks']
        
        if not stocks:
            return self._create_empty_metrics()
        
        fundamental_data = []
        for ticker, data in stocks.items():
            if 'valuation' in data:
                valuation = data['valuation']
                
                fundamental_metrics = {
                    'profit_margin': abs(valuation.get('profit_margins', 0) or 0),
                    'return_on_equity': abs(valuation.get('return_on_equity', 0) or 0),
                    'earnings_growth': abs(valuation.get('earnings_growth', 0) or 0),
                    'revenue_growth': abs(valuation.get('revenue_growth', 0) or 0)
                }
                fundamental_data.append(fundamental_metrics)
        
        if not fundamental_data:
            return self._create_empty_metrics()
        
        fundamental_df = pd.DataFrame(fundamental_data)
        
        # Aggregate sector fundamentals
        metrics = {
            'profit_margin': fundamental_df['profit_margin'].median(),
            'return_on_equity': fundamental_df['return_on_equity'].median(),
            'earnings_growth': fundamental_df['earnings_growth'].median(),
            'revenue_growth': fundamental_df['revenue_growth'].median()
        }
        
        # Calculate percentiles (inverse for risk - higher fundamentals = lower bubble risk)
        percentiles = {}
        fundamental_ranges = {
            'profit_margin': (0.05, 0.40),
            'return_on_equity': (0.08, 0.30),
            'earnings_growth': (-0.1, 0.5),
            'revenue_growth': (-0.05, 0.3)
        }
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                if metric in fundamental_ranges:
                    min_val, max_val = fundamental_ranges[metric]
                    if max_val > min_val:
                        raw_percentile = (value - min_val) / (max_val - min_val)
                        percentiles[metric] = 1 - max(0, min(1, raw_percentile))
                    else:
                        percentiles[metric] = 0.5
                else:
                    percentiles[metric] = 0.5
            else:
                percentiles[metric] = 0.5
        
        return {
            'raw_metrics': metrics,
            'percentiles': percentiles,
            'composite_score': np.mean(list(percentiles.values()))
        }
    
    def _calculate_composite_scores(self, metrics):
        """Calculate weighted composite bubble scores"""
        composite_scores = {}
        
        for category, config in self.metrics_framework.items():
            category_data = metrics.get(category, {})
            if category_data and 'composite_score' in category_data:
                composite_scores[category] = {
                    'score': category_data['composite_score'],
                    'weight': config['weight']
                }
        
        # Overall composite score
        total_weight = sum(score['weight'] for score in composite_scores.values())
        if total_weight > 0:
            overall_score = sum(score['score'] * score['weight'] for score in composite_scores.values()) / total_weight
        else:
            overall_score = 0.5
        
        composite_scores['overall'] = {
            'score': overall_score,
            'weight': 1.0,
            'risk_level': self._classify_risk_level(overall_score)
        }
        
        return composite_scores
    
    def _classify_risk_level(self, score):
        """Classify bubble risk level"""
        if score >= 0.8:
            return "EXTREME_BUBBLE_RISK"
        elif score >= 0.7:
            return "HIGH_BUBBLE_RISK"
        elif score >= 0.6:
            return "ELEVATED_RISK"
        elif score >= 0.4:
            return "MODERATE_RISK"
        else:
            return "LOW_RISK"
    
    def _calculate_relative_risks(self, sector_analysis):
        """Calculate relative risks across sectors"""
        if not sector_analysis:
            return sector_analysis
            
        # Normalize scores for relative comparison
        scores = [analysis['composite_scores']['overall']['score'] 
                 for analysis in sector_analysis.values()]
        
        if scores:
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1
            
            for sector_name, analysis in sector_analysis.items():
                raw_score = analysis['composite_scores']['overall']['score']
                # Normalize to 0-1 range
                normalized_score = (raw_score - min_score) / score_range
                analysis['composite_scores']['overall']['normalized_score'] = normalized_score
                analysis['composite_scores']['overall']['relative_rank'] = stats.percentileofscore(scores, raw_score) / 100
        
        return sector_analysis
    
    def _assess_sector_risk(self, composite_scores):
        """Assess sector-specific risk"""
        overall_score = composite_scores['overall']['score']
        
        return {
            'risk_level': composite_scores['overall']['risk_level'],
            'overall_score': overall_score,
            'recommendation': self._generate_recommendation(overall_score)
        }
    
    def _generate_recommendation(self, score):
        """Generate investment recommendation based on bubble score"""
        if score >= 0.8:
            return "EXTREME CAUTION: Sector shows classic bubble characteristics. Consider reducing exposure."
        elif score >= 0.7:
            return "HIGH RISK: Sector significantly overvalued. Exercise caution with new investments."
        elif score >= 0.6:
            return "ELEVATED RISK: Sector expensive but not in bubble territory. Maintain diversification."
        elif score >= 0.4:
            return "MODERATE RISK: Sector fairly valued. Normal investment approach appropriate."
        else:
            return "LOW RISK: Sector appears reasonably valued or undervalued. Potential buying opportunity."
    
    # REAL CALCULATION METHODS
    def _calculate_price_momentum(self, prices):
        """Calculate 1-year price momentum"""
        if len(prices) < 252:
            return 0
        current_price = prices.iloc[-1]
        price_1y_ago = prices.iloc[-252] if len(prices) >= 252 else prices.iloc[0]
        return (current_price - price_1y_ago) / price_1y_ago
    
    def _calculate_volume_momentum(self, prices, volumes):
        """Calculate volume momentum"""
        if volumes is None or len(volumes) < 50:
            return 1.0
        
        recent_volume = volumes.iloc[-30:].mean()
        historical_volume = volumes.iloc[-252:-30].mean() if len(volumes) >= 282 else volumes.mean()
        
        if historical_volume > 0:
            return recent_volume / historical_volume
        return 1.0
    
    def _calculate_volatility(self, prices):
        """Calculate annualized volatility"""
        if len(prices) < 30:
            return 0.3
        
        returns = prices.pct_change().dropna()
        if len(returns) == 0:
            return 0.3
            
        return returns.std() * np.sqrt(252)
    
    def _calculate_relative_strength(self, prices):
        """Calculate relative strength indicator (simplified)"""
        if len(prices) < 30:
            return 0.5
            
        gains = 0
        losses = 0
        for i in range(1, min(30, len(prices))):
            change = (prices.iloc[-i] - prices.iloc[-i-1]) / prices.iloc[-i-1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        
        if gains + losses == 0:
            return 0.5
            
        return gains / (gains + losses)
    
    def _calculate_analyst_sentiment(self, info):
        """Calculate analyst sentiment from real data"""
        recommendation_mean = info.get('recommendationMean', 2.5)
        recommendation_key = info.get('recommendationKey', 'hold')
        
        recommendation_map = {
            'strong_buy': 1.0, 'buy': 2.0, 'hold': 3.0, 
            'sell': 4.0, 'strong_sell': 5.0
        }
        
        if recommendation_key in recommendation_map:
            return recommendation_map[recommendation_key]
        elif recommendation_mean:
            return max(1.0, min(5.0, recommendation_mean))
        else:
            return 3.0
    
    # Helper methods
    def _safe_median(self, series):
        """Safely calculate median handling NaN values"""
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        return clean_series.median() if not clean_series.empty else 0
    
    def _create_empty_metrics(self):
        """Create empty metrics structure"""
        return {
            'raw_metrics': {},
            'percentiles': {},
            'composite_score': 0.5
        }