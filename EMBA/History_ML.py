# History_ML.py (Fixed datetime sorting)
"""
Enhanced Historical Data Manager for ML Training - Fixed datetime sorting
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-


from    datetime import datetime
import  json
import  os

class MLHistoryManager:
    def __init__(self, storage_file="ml_historical_data.json"):
        self.storage_file = storage_file
        self.historical_data = self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical analysis data with proper datetime conversion"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for sector, analyses in data.items():
                    for analysis in analyses:
                        # Convert all timestamp fields
                        timestamp_fields = ['timestamp', 'saved_timestamp']
                        for field in timestamp_fields:
                            if field in analysis and isinstance(analysis[field], str):
                                try:
                                    analysis[field] = datetime.fromisoformat(analysis[field])
                                except ValueError:
                                    # Handle older timestamp formats if needed
                                    try:
                                        analysis[field] = datetime.strptime(analysis[field], '%Y-%m-%d %H:%M:%S')
                                    except ValueError:
                                        # If all else fails, use current time
                                        analysis[field] = datetime.now()
                                        print(f"Warning: Could not parse {field} for {sector}, using current time")
                return data
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return {}
        return {}
    
    def save_current_analysis(self, sector_analysis):
        """Save current analysis to historical data"""
        current_time = datetime.now()
        
        for sector_name, analysis in sector_analysis.items():
            if sector_name not in self.historical_data:
                self.historical_data[sector_name] = []
            
            # Add timestamp to analysis
            analysis_with_timestamp = analysis.copy()
            analysis_with_timestamp['saved_timestamp'] = current_time
            
            self.historical_data[sector_name].append(analysis_with_timestamp)
            
            # Keep only last 50 analyses to prevent file from growing too large
            if len(self.historical_data[sector_name]) > 50:
                self.historical_data[sector_name] = self.historical_data[sector_name][-50:]
        
        self._save_historical_data()
        return True
    
    def _save_historical_data(self):
        """Save historical data to file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(self.storage_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2, default=convert_datetime)
            return True
        except Exception as e:
            print(f"Error saving historical data: {e}")
            return False
    
    def get_training_data(self, min_analyses=3):
        """Get formatted training data for ML models with proper datetime sorting"""
        training_data = {}
        
        for sector_name, analyses in self.historical_data.items():
            if len(analyses) >= min_analyses:
                # Sort by saved_timestamp - ensure all are datetime objects
                try:
                    analyses_sorted = sorted(analyses, key=lambda x: x['saved_timestamp'])
                except TypeError as e:
                    print(f"Error sorting analyses for {sector_name}: {e}")
                    print("Attempting to fix datetime objects...")
                    # Fix any remaining string timestamps
                    for analysis in analyses:
                        if isinstance(analysis.get('saved_timestamp'), str):
                            try:
                                analysis['saved_timestamp'] = datetime.fromisoformat(analysis['saved_timestamp'])
                            except ValueError:
                                analysis['saved_timestamp'] = datetime.now()
                    # Try sorting again
                    analyses_sorted = sorted(analyses, key=lambda x: x['saved_timestamp'])
                
                features = []
                targets = []
                
                for i in range(1, len(analyses_sorted)):
                    current = analyses_sorted[i-1]
                    next_analysis = analyses_sorted[i]
                    
                    # Extract features from current analysis
                    feature_vector = self._extract_features(current)
                    target = next_analysis['composite_scores']['overall']['score']
                    
                    features.append(feature_vector)
                    targets.append(target)
                
                training_data[sector_name] = {
                    'features': features,
                    'targets': targets,
                    'timestamps': [a['saved_timestamp'] for a in analyses_sorted[1:]]
                }
        
        return training_data
    
    def _extract_features(self, analysis):
        """Extract feature vector from analysis"""
        features = []
        
        # Overall score
        features.append(analysis['composite_scores']['overall']['score'])
        
        # Category scores
        for category, scores in analysis['composite_scores'].items():
            if category != 'overall':
                features.append(scores['score'])
        
        return features
    
    def get_sector_history(self, sector_name, window=12):
        """Get recent history for a specific sector"""
        if sector_name in self.historical_data:
            analyses = self.historical_data[sector_name]
            # Ensure proper sorting by datetime
            try:
                analyses_sorted = sorted(analyses, key=lambda x: x['saved_timestamp'])
                return analyses_sorted[-window:]
            except TypeError:
                # Fallback: sort by string representation if datetime fails
                analyses_sorted = sorted(analyses, key=lambda x: str(x['saved_timestamp']))
                return analyses_sorted[-window:]
        return []
    
    def get_analysis_count(self):
        """Get total number of analyses stored"""
        count = 0
        for sector, analyses in self.historical_data.items():
            count += len(analyses)
        return count