"""
Historical Comparator - Compare current sectors with historical bubbles
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import numpy as np

class HistoricalComparator:
    def __init__(self):
        self.historical_bubbles = {
            'Dot-com Bubble (2000)': {
                'period': ('1995-01-01', '2000-03-24'),
                'characteristics': {
                    'valuation_extreme': True,
                    'momentum_high': True,
                    'sentiment_euphoric': True
                }
            },
            '2008 Financial Crisis': {
                'period': ('2007-10-01', '2009-03-09'),
                'characteristics': {
                    'leverage_high': True,
                    'complex_products': True
                }
            },
            'COVID-19 Crash': {
                'period': ('2020-02-01', '2020-03-23'),
                'characteristics': {
                    'sharp_correction': True,
                    'recovery_swift': True
                }
            }
        }
    
    async def compare_with_historical_bubbles(self, sector_analysis):
        """Compare current sectors with historical bubble periods"""
        historical_comparison = {}
        
        for sector_name, analysis in sector_analysis.items():
            historical_comparison[sector_name] = await self._compare_sector_with_history(
                sector_name, analysis
            )
        
        return historical_comparison
    
    async def _compare_sector_with_history(self, sector_name, analysis):
        """Compare a single sector with historical bubbles"""
        comparison = {
            'sector': sector_name,
            'current_risk_score': analysis['composite_scores']['overall']['score'],
            'historical_comparisons': {},
            'similarity_scores': {}
        }
        
        for bubble_name, bubble_info in self.historical_bubbles.items():
            historical_metrics = await self._calculate_historical_metrics(bubble_info['period'])
            similarity_score = self._calculate_similarity_score(analysis, historical_metrics)
            
            comparison['historical_comparisons'][bubble_name] = {
                'historical_metrics': historical_metrics,
                'similarity_score': similarity_score,
                'bubble_characteristics': bubble_info['characteristics']
            }
            
            comparison['similarity_scores'][bubble_name] = similarity_score
        
        # Identify most similar historical bubble
        if comparison['similarity_scores']:
            most_similar = max(comparison['similarity_scores'].items(), key=lambda x: x[1])
            comparison['most_similar_bubble'] = {
                'bubble_name': most_similar[0],
                'similarity_score': most_similar[1]
            }
        
        return comparison
    
    async def _calculate_historical_metrics(self, period):
        """Calculate metrics for historical bubble period"""
        # Simulate historical metrics (in real implementation, use actual historical data)
        return {
            'valuation_score': np.random.uniform(0.7, 0.95),
            'momentum_score': np.random.uniform(0.6, 0.9),
            'sentiment_score': np.random.uniform(0.7, 0.95),
            'composite_score': np.random.uniform(0.65, 0.9)
        }
    
    def _calculate_similarity_score(self, current_analysis, historical_metrics):
        """Calculate similarity score between current and historical metrics"""
        current_scores = current_analysis['composite_scores']
        
        # Compare composite scores
        current_composite = current_scores['overall']['score']
        historical_composite = historical_metrics['composite_score']
        
        # Calculate similarity (1 - normalized difference)
        similarity = 1 - abs(current_composite - historical_composite)
        
        return max(0, min(1, similarity))