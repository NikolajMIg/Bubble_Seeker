# ML_Predictor.py (Enhanced with model reporting and prediction methods)
"""
Machine Learning Bubble Risk Predictor - Enhanced with model reporting and prediction methods
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  pandas as pd
import  numpy as np
from    sklearn.preprocessing import StandardScaler
import  joblib
import  warnings

warnings.filterwarnings('ignore')

class BubbleRiskPredictor:
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.feature_columns = None
        self.cv_scores = {}  # Store cross-validation scores
        self.training_scores = {}  # Store training scores
        self.best_model = None
        self.best_score = -np.inf
    
    def prepare_features(self, sector_analysis, historical_data=None):
        """Prepare features for ML prediction - CONSISTENT with training features"""
        features = []
        sector_names = []
        
        for sector_name, analysis in sector_analysis.items():
            sector_features = self._extract_sector_features(analysis)
            features.append(sector_features)
            sector_names.append(sector_name)
        
        feature_df = pd.DataFrame(features, index=sector_names)
        
        # Store feature columns for consistency
        if self.feature_columns is None:
            self.feature_columns = feature_df.columns.tolist()
        else:
            # Ensure we use the same columns as during training
            missing_cols = set(self.feature_columns) - set(feature_df.columns)
            for col in missing_cols:
                feature_df[col] = 0  # Add missing columns with default values
            feature_df = feature_df[self.feature_columns]  # Reorder to match training
        
        return feature_df
    
    def _extract_sector_features(self, analysis):
        """Extract ONLY the features used during training"""
        features = {}
        
        # Use ONLY the 5 composite scores that are used in training
        features['overall'] = analysis['composite_scores']['overall']['score']
        features['valuation'] = analysis['composite_scores']['valuation']['score']
        features['momentum'] = analysis['composite_scores']['momentum']['score']
        features['sentiment'] = analysis['composite_scores']['sentiment']['score']
        features['fundamental'] = analysis['composite_scores']['fundamental']['score']
        
        return features
    
    def train_models_ini_123(self, features, targets, app, test_size=0.2):        
        """Train multiple ML models with enhanced reporting"""
     
        X = features.values
        y = targets.values
        
        # Store feature columns for consistency
        self.feature_columns = features.columns.tolist()
        
        # Scale features
        self.scalers['feature'] = StandardScaler()  # => mean & sigma
        
        X_scaled = self.scalers['feature'].fit_transform(X) # scaled (x) = (x-<x>)/sigma      
        
        return X_scaled, y       
            
    def predict_future_risk(self, features):
        """Predict future bubble risk with multiple scenarios"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure features are in the correct format
        if isinstance(features, (list, np.ndarray)):
            if len(features) != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {len(features)}")
            features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scalers['feature'].transform(features)
        
        # Get current risk prediction
        current_risk = self.models[self.best_model].predict(features_scaled)[0]
        current_risk = max(0, min(1, current_risk))  # Clamp between 0 and 1
        
        # Generate future scenarios (6 months projection)
        months = 6
        scenarios = {
            'baseline': self._generate_baseline_scenario(current_risk, months),
            'bullish': self._generate_bullish_scenario(current_risk, months),
            'bearish': self._generate_bearish_scenario(current_risk, months),    
            'volatile': self._generate_volatile_scenario(current_risk, months)
        }
        
        # Determine recommended scenario based on current risk and trends
        recommended_scenario = self._determine_recommended_scenario(current_risk, scenarios)
        
        return {
            'current_risk': current_risk,
            'scenarios': scenarios,
            'recommended_scenario': recommended_scenario
        }
    
    def _generate_baseline_scenario(self, current_risk, months):
        """Generate baseline scenario with minimal changes"""
        trend = [current_risk]
        for i in range(1, months):
            # Small random walk around current level
            change = np.random.normal(0, 0.02)
            new_risk = max(0, min(1, trend[-1] + change))
            trend.append(new_risk)
        return trend
    
    def _generate_bullish_scenario(self, current_risk, months):
        """Generate bullish scenario with decreasing risk"""
        trend = [current_risk]
        for i in range(1, months):
            # Gradual improvement (decreasing risk)
            improvement = np.random.normal(0.03, 0.01)
            new_risk = max(0, min(1, trend[-1] - improvement))
            trend.append(new_risk)
        return trend
    
    def _generate_bearish_scenario(self, current_risk, months):
        """Generate bearish scenario with increasing risk"""
        trend = [current_risk]
        for i in range(1, months):
            # Gradual deterioration (increasing risk)
            deterioration = np.random.normal(0.04, 0.015)
            new_risk = max(0, min(1, trend[-1] + deterioration))
            trend.append(new_risk)
        return trend
    
    def _generate_volatile_scenario(self, current_risk, months):
        """Generate volatile scenario with high fluctuations"""
        trend = [current_risk]
        for i in range(1, months):
            # High volatility
            change = np.random.normal(0, 0.06)
            new_risk = max(0, min(1, trend[-1] + change))
            trend.append(new_risk)
        return trend
    
    def _determine_recommended_scenario(self, current_risk, scenarios):
        """Determine the most likely scenario based on current risk level"""
        if current_risk > 0.7:
            # High risk - bearish or volatile more likely
            return 'bearish' if np.random.random() > 0.5 else 'volatile'
        elif current_risk > 0.5:
            # Medium risk - baseline most likely
            return 'baseline'
        else:
            # Low risk - bullish more likely
            return 'bullish' if np.random.random() > 0.3 else 'baseline'
    
    def predict_current_risk(self, features):
        """Predict current bubble risk using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure features are in the correct format
        if isinstance(features, (list, np.ndarray)):
            if len(features) != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {len(features)}")
            features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scalers['feature'].transform(features)
        
        # Predict using best model
        prediction = self.models[self.best_model].predict(features_scaled)[0]
        return max(0, min(1, prediction))  # Clamp between 0 and 1
    
    def get_model_report(self):
        """Generate comprehensive model performance report"""
        if not self.is_trained:
            return "Models not trained yet. Please train models first."
        
        report = []
        report.append("MACHINE LEARNING MODEL PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Best Model: {self.best_model.upper()}")
        report.append(f"Best CV R² Score: {self.best_score:.3f}")
        report.append("")
        
        # Model descriptions
        model_descriptions = {
            'random_forest': "Random Forest: Ensemble method combining multiple decision trees. Robust to outliers and overfitting.",
            'gradient_boosting': "Gradient Boosting: Sequentially builds trees to correct previous errors. Good for complex patterns.",
            'linear': "Linear Regression: Models linear relationships. Simple and interpretable.",
            'svr': "Support Vector Regression: Finds optimal hyperplane. Effective for non-linear patterns."
        }
        
        report.append("MODEL PERFORMANCE DETAILS:")
        for name in self.models.keys():
            if name in self.cv_scores:
                cv_mean = np.mean(self.cv_scores[name])
                cv_std = np.std(self.cv_scores[name])
                train_score = self.training_scores.get(name, 0)
                overfitting = train_score - cv_mean
                
                report.append(f"\n{name.upper():20}")
                report.append(f"  Cross-Validation R²: {cv_mean:.3f} (+-{cv_std:.3f})")
                report.append(f"  Training R²:         {train_score:.3f}")
                report.append(f"  Overfitting gap:     {overfitting:+.3f}")
                
                if name in model_descriptions:
                    report.append(f"  Description:         {model_descriptions[name]}")
        
        # Feature importance if available
        if self.best_model in self.feature_importance:
            report.append("\nFEATURE IMPORTANCE (Best Model):")
            features_sorted = sorted(self.feature_importance[self.best_model].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in features_sorted:
                report.append(f"  {feature:15}: {importance:.3f}")
        
        # Performance interpretation
        report.append("\nPERFORMANCE INTERPRETATION:")
        if self.best_score < 0:
            report.append("  POOR: Model performs worse than simple average prediction")
            report.append("  Recommendation: Collect more training data")
        elif self.best_score < 0.3:
            report.append("  WEAK: Limited predictive power")
            report.append("  Recommendation: Additional data will likely improve performance")
        elif self.best_score < 0.6:
            report.append("  MODERATE: Reasonable predictive power")
            report.append("  Recommendation: Model can provide useful insights")
        elif self.best_score < 0.8:
            report.append("  GOOD: Strong predictive power")
            report.append("  Recommendation: Model should provide reliable predictions")
        else:
            report.append("  EXCELLENT: Very strong predictive power")
            report.append("  Recommendation: High confidence in model predictions")
        
        return "\n".join(report)
    
    def save_model(self, filepath):   # Jamais appelé...
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'feature_columns': self.feature_columns,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']
        self.feature_columns = model_data['feature_columns']
        self.best_model = model_data['best_model']
        self.best_score = model_data['best_score']
        
        print(f"Model loaded from {filepath}")