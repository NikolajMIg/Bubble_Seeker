# Visualization.py (Fixed - Enhanced simultaneous plotting)
"""
Enhanced Visualization with ML Predictions - Better readability and colors
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd
import  plotly.graph_objects as go
from    plotly.subplots import make_subplots
import  seaborn as sns   # jamais Utilisé???
from    Tools   import Display_info
import  os
import  Tools

def Plt_STYL():
    if os.path.isfile(Tools.Mon_Modele_STYLE1):
        plt.style.use(Tools.Mon_Modele_STYLE1)
    else:
        plt.style.use('default')

class SectorDashboards:
    def __init__(self):
        Plt_STYL()
        # Enhanced color palette for better readability
        self.color_palette = {
            'high_risk': '#FF6B6B',
            'medium_risk': '#FFD166', 
            'low_risk': '#06D6A0',
            'neutral': '#118AB2',
            'extreme_risk': '#8B0000'
        }
        self.risk_colors = {
            'EXTREME_BUBBLE_RISK': '#8B0000',
            'HIGH_BUBBLE_RISK': '#FF4500',
            'ELEVATED_RISK': '#FFA500',
            'MODERATE_RISK': '#FFD700',
            'LOW_RISK': '#32CD32'
        }

    def HH_Pos(h, m):

        if h < 0.8:
            return h+m
        return h/2
    
    def create_bubble_risk_chart(self, sector_analysis, app, show=True, block=True):
        """Create bubble risk comparison chart"""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            sectors = list(sector_analysis.keys())
            scores = [analysis['composite_scores']['overall']['score'] 
                     for analysis in sector_analysis.values()]
            risk_levels = [analysis['composite_scores']['overall']['risk_level']
                          for analysis in sector_analysis.values()]
            
            # Color coding based on risk level
            colors = []
            for risk_level in risk_levels:
                if risk_level in self.risk_colors:
                    colors.append(self.risk_colors[risk_level])
                else:
                    # Fallback based on score
                    score = next((s for s, r in zip(scores, risk_levels) if r == risk_level), 0.5)
                    if score >= 0.7:
                        colors.append(self.color_palette['high_risk'])
                    elif score >= 0.5:
                        colors.append(self.color_palette['medium_risk'])
                    else:
                        colors.append(self.color_palette['low_risk'])
            
            bars = ax.bar(sectors, scores, color=colors, zorder=3, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Bubble Risk Score', fontweight='bold', fontsize=12)
            ax.set_xlabel('Sectors', fontweight='bold', fontsize=12)
            ax.set_title('Sector Bubble Risk Comparison', fontweight='bold', fontsize=16, pad=20)
            ax.set_ylim(0, 1.0)
            #ax.grid(True, alpha=0.3, linestyle='--')
            ax.grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)
            
            # Add value labels on bars
            for bar, score, risk_level in zip(bars, scores, risk_levels):
                height = bar.get_height()
                if height > 0.85:
                    height= height/2
                else:
                    height += 0.01
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}\n({risk_level.split("_")[0]})', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.xticks(rotation=45, fontweight='bold')
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            Display_info(f"Error creating bubble risk chart: {e}", app)
            return None
    
    def create_radar_chart(self, sector_analysis, show=True, block=True):
        """Create radar chart for sector comparison"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            categories = ['Valuation', 'Momentum', 'Sentiment', 'Fundamental']
            sectors = list(sector_analysis.keys()) # Innutilisé ???
            
            # Prepare data for radar chart
            for sector_name, analysis in sector_analysis.items():
                scores = [
                    analysis['composite_scores']['valuation']['score'],
                    analysis['composite_scores']['momentum']['score'], 
                    analysis['composite_scores']['sentiment']['score'],
                    analysis['composite_scores']['fundamental']['score']
                ]
                # Close the radar chart
                scores += scores[:1]
                
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                ax.plot(angles, scores, 'o-', linewidth=2, label=sector_name, markersize=6, zorder=3)
                ax.fill(angles, scores, alpha=0.1, zorder=3)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.legend(bbox_to_anchor=(1.2, 1.0), fontsize=10)
            ax.set_title('Sector Metrics Radar Chart', fontweight='bold', fontsize=16, pad=20)
            #ax.grid(True, alpha=0.3)
            ax.grid(True, axis='both', alpha=0.5, linestyle='dashed', zorder=-1.0)
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            print(f"Error creating radar chart: {e}")
            return None
    
    def create_detailed_sector_report(self, sector_analysis, sector_name, app, show=True, block=True):
        """Create detailed report for a specific sector"""
        try:
            if sector_name not in sector_analysis:
                Display_info(f"Sector {sector_name} not found in analysis", app)
                return None
            
            analysis = sector_analysis[sector_name]
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Detailed Analysis: {sector_name}', fontweight='bold', fontsize=18)
            
            # Plot 1: Category scores
            categories = ['Valuation', 'Momentum', 'Sentiment', 'Fundamental']
            cat_scores = [
                analysis['composite_scores']['valuation']['score'],
                analysis['composite_scores']['momentum']['score'],
                analysis['composite_scores']['sentiment']['score'], 
                analysis['composite_scores']['fundamental']['score']
            ]
            
            colors = [self.risk_colors['HIGH_BUBBLE_RISK'] if score > 0.7 else 
                     self.risk_colors['ELEVATED_RISK'] if score > 0.5 else 
                     self.risk_colors['LOW_RISK'] for score in cat_scores]
            
            bars = axes[0,0].bar(categories, cat_scores, color=colors, edgecolor='black', zorder=3)
            axes[0,0].set_ylabel('Score', fontweight='bold')
            axes[0,0].set_title('Category Scores', fontweight='bold', fontsize=14)
            axes[0,0].set_ylim(0, 1.0)
            #axes[0,0].grid(True, alpha=0.3, linestyle='--')
            axes[0,0].grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)
            
            # Add value labels
            for bar, score in zip(bars, cat_scores):
                height = bar.get_height()
                if height > 0.85:
                    height= height/2
                else:
                    height += 0.02
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Valuation metrics breakdown
            if 'valuation' in analysis['metrics']:
                val_metrics = analysis['metrics']['valuation']['percentiles']
                bars = axes[0,1].bar(val_metrics.keys(), val_metrics.values(), 
                                   color=self.color_palette['high_risk'], zorder=3)
                axes[0,1].set_title('Valuation Metrics (Percentiles)', fontweight='bold', fontsize=14)
                axes[0,1].tick_params(axis='x', rotation=45)
                axes[0,1].set_ylim(0, 1.0)
                #axes[0,1].grid(True, alpha=0.3, linestyle='--')
                axes[0,1].grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)
                
                # Add value labels
                for bar, (metric, value) in zip(bars, val_metrics.items()):
                    height = bar.get_height()
                    if height > 0.85:
                        height= height/2
                    else:
                        height += 0.02
                    axes[0,1].text(bar.get_x() + bar.get_width()/2., height ,
                                  f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Risk assessment
            risk_level = analysis['risk_assessment']['risk_level']
            overall_score = analysis['composite_scores']['overall']['score']
            recommendation = analysis['risk_assessment']['recommendation']
            
            # Create a more visual risk assessment
            risk_color = self.risk_colors.get(risk_level, '#666666')
            axes[1,0].add_patch(plt.Rectangle((0.1, 0.6), 0.8, 0.3, fill=True, 
                                            color=risk_color, alpha=0.3))
            axes[1,0].text(0.5, 0.8, f'RISK LEVEL: {risk_level}', 
                          fontsize=16, ha='center', va='center', fontweight='bold',
                          transform=axes[1,0].transAxes)
            axes[1,0].text(0.5, 0.7, f'Score: {overall_score:.3f}/1.0', 
                          fontsize=14, ha='center', va='center', 
                          transform=axes[1,0].transAxes)
            axes[1,0].text(0.5, 0.4, 'Recommendation:', 
                          fontsize=12, ha='center', va='center', fontweight='bold',
                          transform=axes[1,0].transAxes)
            axes[1,0].text(0.5, 0.3, recommendation, 
                          fontsize=10, ha='center', va='center', 
                          transform=axes[1,0].transAxes, wrap=True)
            axes[1,0].set_title('Risk Assessment', fontweight='bold', fontsize=14)
            axes[1,0].axis('off')
            
            # Plot 4: Historical comparison placeholder
            axes[1,1].text(0.5, 0.7, 'Historical Context', 
                          fontsize=14, ha='center', va='center', fontweight='bold',
                          transform=axes[1,1].transAxes)
            axes[1,1].text(0.5, 0.5, 'Analysis based on current market data\nand historical bubble patterns', 
                          fontsize=11, ha='center', va='center', 
                          transform=axes[1,1].transAxes)
            axes[1,1].text(0.5, 0.3, f'Analysis Date:\n{analysis["timestamp"].strftime("%Y-%m-%d %H:%M")}', 
                          fontsize=10, ha='center', va='center', 
                          transform=axes[1,1].transAxes)
            axes[1,1].set_title('Historical Analysis', fontweight='bold', fontsize=14)
            axes[1,1].axis('off')
            
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            Display_info(f"Error creating detailed sector report: {e}", app)
            return None
    
    def create_metrics_comparison_chart(self, sector_analysis, show=True, block=True):
        """Create metrics comparison across sectors"""
        try:
            sectors = list(sector_analysis.keys())
            metrics = ['Valuation', 'Momentum', 'Sentiment', 'Fundamental']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Metrics Comparison Across Sectors', fontweight='bold', fontsize=18)
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                scores = []
                for sector in sectors:
                    score = sector_analysis[sector]['composite_scores'][metric.lower()]['score']
                    scores.append(score)
                
                colors = [self.risk_colors['HIGH_BUBBLE_RISK'] if s > 0.7 else 
                         self.risk_colors['ELEVATED_RISK'] if s > 0.5 else 
                         self.risk_colors['LOW_RISK'] for s in scores]
                
                bars = ax.bar(sectors, scores, color=colors, zorder=3, edgecolor='black')
                ax.set_title(f'{metric} Scores', fontweight='bold', fontsize=14)
                ax.set_ylim(0, 1.0)
                ax.tick_params(axis='x', rotation=45)
                #ax.grid(True, alpha=0.3, linestyle='--')
                ax.grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    if height > 0.85:
                        height= height/2
                    else:
                        height += 0.01
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            print(f"Error creating metrics comparison chart: {e}")
            return None
    
    def create_historical_indicators_chart(self, sector_analysis, show=True, block=True):
        """Create historical indicators chart"""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            sectors = list(sector_analysis.keys())
            scores = [analysis['composite_scores']['overall']['score'] 
                     for analysis in sector_analysis.values()]
            risk_levels = [analysis['composite_scores']['overall']['risk_level']
                          for analysis in sector_analysis.values()]
            
            # Color points based on risk level
            colors = [self.risk_colors.get(level, '#666666') for level in risk_levels]
            
            # Create scatter plot with larger points
            scatter = ax.scatter(range(len(sectors)), scores, c=colors, s=200, alpha=0.7, 
                               edgecolors='black', linewidths=2, zorder=3)
            
            # Connect points with lines
            ax.plot(range(len(sectors)), scores, 'gray', alpha=0.5, linestyle='--')
            
            ax.set_ylabel('Bubble Risk Score', fontweight='bold', fontsize=12)
            ax.set_xlabel('Sectors', fontweight='bold', fontsize=12)
            ax.set_title('Current Bubble Risk Scores by Sector', fontweight='bold', fontsize=16, pad=20)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(len(sectors)))
            ax.set_xticklabels(sectors, rotation=45, fontweight='bold')
            #ax.grid(True, alpha=0.3, linestyle='--')
            ax.grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)
            
            # Add value annotations with risk level
            for i, (sector, score, risk_level) in enumerate(zip(sectors, scores, risk_levels)):
                ax.annotate(f'{score:.3f}\n{risk_level.split("_")[0]}', 
                           (i, score), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            print(f"Error creating historical indicators chart: {e}")
            return None
    
    def show_plots_simultaneously(self, sector_analysis, app):
        """Show all plots simultaneously in separate windows"""
        try:
            # Turn on interactive mode
            plt.ion()
            
            # Create all plots without blocking
            self.create_bubble_risk_chart(sector_analysis, app, show=True, block=False)
            self.create_radar_chart(sector_analysis, show=True, block=False)
            self.create_metrics_comparison_chart(sector_analysis, show=True, block=False)
            
            # Show detailed reports for all sectors without blocking
            for sector_name in sector_analysis.keys():
                self.create_detailed_sector_report(sector_analysis, sector_name, app, show=True, block=False)
            
            # Keep windows open until user closes them
            Display_info("All plots displayed simultaneously in separate windows", app)
            Display_info("Close all plot windows to continue...", app)
            plt.show(block=True)  # This blocks until all windows are closed
            plt.ioff()  # Turn off interactive mode
            
        except Exception as e:
            Display_info(f"Error showing plots simultaneously: {e}", app)
            plt.ioff()  # Ensure interactive mode is turned off

    def create_interactive_dashboard(self, sector_analysis, historical_analysis=None):
        """Create interactive Plotly dashboard"""
        try:
            sectors = list(sector_analysis.keys())
            overall_scores = [analysis['composite_scores']['overall']['score'] 
                            for analysis in sector_analysis.values()]
            risk_levels = [analysis['composite_scores']['overall']['risk_level']
                          for analysis in sector_analysis.values()]
            
            # Convert risk levels to colors
            colors = []
            for risk_level in risk_levels:
                if risk_level in self.risk_colors:
                    colors.append(self.risk_colors[risk_level])
                else:
                    colors.append('#666666')  # Default gray
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=sectors,
                y=overall_scores,
                marker_color=colors,
                text=[f'{score:.3f}' for score in overall_scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>Risk Level: %{customdata}<extra></extra>',
                customdata=risk_levels
            ))
            
            fig.update_layout(
                title=dict(
                    text='Interactive Bubble Risk Dashboard',
                    font=dict(size=20, family='Arial', color='black')
                ),
                xaxis=dict(
                    title='Sectors',
                    titlefont=dict(size=14, family='Arial'),
                    tickfont=dict(size=12, family='Arial')
                ),
                yaxis=dict(
                    title='Bubble Risk Score',
                    titlefont=dict(size=14, family='Arial'),
                    range=[0, 1.0],
                    tickfont=dict(size=12, family='Arial')
                ),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial', size=12, color='black')
            )
            
            fig.show()
            
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")

class MLVisualization:
    def __init__(self):
        Plt_STYL()
        # Enhanced color palette for better readability
        self.color_palette = {
            'baseline':  '#2E86AB',     # Calm blue
            'bullish':   '#A23B72',     # Optimistic purple  
            'bearish':   '#F18F01',     # Warning orange
            'volatile':  '#C73E1D',     # Dangerous red
            'current':   '#b0c4de',     # lightsteelblue
            'predicted': '#CD5C5C',     # indianred 
            'predicted__': '#8B1E3F'    # Future magenta

        }
        
        self.risk_colors = {
            'EXTREME_BUBBLE_RISK': '#8B0000',
            'HIGH_BUBBLE_RISK': '#FF4500',
            'ELEVATED_RISK': '#FFA500',
            'MODERATE_RISK': '#FFD700',
            'LOW_RISK': '#32CD32'
        }
    def _AX_(self,axes, i, j, st, BOTH, st2, sectors, Bsectors) :
        axes[i, j].set_title(st, fontsize=12 , color="#2e55FF", style='italic')
        if BOTH:
            axes[i,j].grid(True, axis='both', alpha=0.7, linestyle='dashed', zorder=-1.0)
        else:
            axes[i,j].grid(True, axis='y', alpha=0.7, linestyle='dashed', zorder=-1.0)

        axes[i,j].set_ylabel(st2, fontsize=12 , color="#3e3e3e")
        if Bsectors:
            axes[i,j].set_xticklabels(sectors, rotation=30, color='#5c5c5c', size=12)
    
    def create_risk_evolution_chart(self, future_predictions, current_analysis=None, show=True, block=True):
        """Create enhanced chart showing predicted risk evolution with better readability"""
        try:       
            from matplotlib.font_manager import FontProperties

            Tools._STYLE(True)  
            plt.rc('axes', axisbelow=True)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            fp = FontProperties(family='serif', style='italic')
            #plt.title(st, pad=25, fontdict = font1)
            fig.suptitle('ML-Predicted Bubble Risk Evolution (6-Month Forecast)\n', 
                          fontproperties = fp, color='blue', fontsize=16)
            
            sectors = list(future_predictions.keys())
            
            # Plot 1: Current vs Predicted Risk with enhanced styling
            current_risks = [fp['current_risk'] for fp in future_predictions.values()]
            predicted_risks = [fp['scenarios'][fp['recommended_scenario']][-1] 
                              for fp in future_predictions.values()]
            
            x_pos = np.arange(len(sectors))
            width = 0.35
                                   
            self._AX_(axes,0,0,'Current vs Predicted Bubble Risk', False,'Bubble Risk Score',sectors, True)
            axes[0,0].set_xticks(x_pos)            
            axes[0,0].legend(fontsize=10)
            
            # Use consistent colors
            Categories=['Current', 'Predicted (6M)']
            bars1 = axes[0,0].bar(x_pos - width/2, current_risks, width, 
                                 label=Categories[0], 
                                 color=self.color_palette['current'],
                                 edgecolor='black', linewidth=1, zorder=3)
            bars2 = axes[0,0].bar(x_pos + width/2, predicted_risks, width, 
                                 label=Categories[1], 
                                 color=self.color_palette['predicted'],
                                 edgecolor='black', linewidth=1, zorder=3)
            
            axes[0,0].set_ylim(0, 1)                       
            axes[0,0].legend(Categories,loc=2)
            
            # Add value labels with better positioning
            for bar, score in zip(bars1, current_risks):
                height = bar.get_height()
                if height > 0.85:
                    height= height/2
                else:
                    height += 0.02
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height ,
                              f'{score:.3f}', ha='center', va='bottom', 
                              fontweight='bold', fontsize=9, color='#000000', rotation = 90)
            
            for bar, score in zip(bars2, predicted_risks):
                height = bar.get_height()
                if height > 0.85:
                    height= height/2
                else:
                    height += 0.02
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{score:.3f}', ha='center', va='bottom', 
                              fontweight='bold', fontsize=9, color='#000000', rotation = 90)
            
            # Plot 2: Risk Change with enhanced color coding
            risk_changes = [predicted - current for current, predicted in zip(current_risks, predicted_risks)]
            
            # Enhanced color coding based on risk change magnitude
            colors = []
            for change in risk_changes:
                if change > 0.05:
                    colors.append(self.risk_colors['HIGH_BUBBLE_RISK'])  # Significant increase
                elif change > 0.02:
                    colors.append(self.risk_colors['ELEVATED_RISK'])     # Moderate increase
                elif change > -0.02:
                    colors.append(self.risk_colors['MODERATE_RISK'])     # Stable
                elif change > -0.05:
                    colors.append('#90EE90')                             # Slight decrease
                else:
                    colors.append(self.risk_colors['LOW_RISK'])          # Significant decrease
            
            bars = axes[0,1].bar(sectors, risk_changes, color=colors,  
                                edgecolor='black', linewidth=1, zorder=3)
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            self._AX_(axes,0,1,'Predicted Risk Change Over 6 Months', False,'Risk Change (Predicted - Current)',sectors, True)
            #axes[0,1].tick_params(axis='x', rotation=30, color='#5c5c5c', size=12)
            #ax0.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
            axes[0,1].set_ylim(-0.2, 0.2)  # Fixed scale for better comparison
            
            # Add value labels for risk changes
            for bar, change in zip(bars, risk_changes):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                y_pos = height + 0.01 if height >= 0 else height - 0.01
                if abs(y_pos) > 0.16:
                    y_pos= y_pos/2
                axes[0,1].text(bar.get_x() + bar.get_width()/2., y_pos,
                              f'{change:+.3f}', ha='center', va=va, 
                              fontweight='bold', fontsize=9)
            
            # Plot 3: Scenario Analysis for Top 3 Riskiest Sectors with enhanced styling
            top_sectors = sorted(future_predictions.items(), 
                               key=lambda x: x[1]['current_risk'], reverse=True)[:3]
            
            for i, (sector, prediction) in enumerate(top_sectors):
                periods = len(prediction['scenarios']['baseline'])
                time_points = range(periods)
                
                for scenario, trend in prediction['scenarios'].items():
                    linestyle = '-' if scenario == prediction['recommended_scenario'] else '--'
                    linewidth = 2.5 if scenario == prediction['recommended_scenario'] else 1.5
                    alpha = 1.0 if scenario == prediction['recommended_scenario'] else 0.7
                    
                    axes[1,0].plot(time_points, trend, 
                                  label=f'{sector} - {scenario}', 
                                  linestyle=linestyle, 
                                  linewidth=linewidth,
                                  alpha=alpha,
                                  zorder=3,
                                  color=self.color_palette[scenario])
            
            axes[1,0].set_xlabel('Months Ahead', fontsize=12, color = '#3c3c3c')
            self._AX_(axes,1,0,'Risk Evolution Scenarios (Top 3 Riskiest Sectors)', True,'Bubble Risk Score',sectors, False )
            axes[1,0].legend(fontsize=9, loc='upper right')
            
            axes[1,0].set_ylim(0, 1.0)
            
            # Plot 4: Risk Probability Distribution with enhanced styling
            risk_probs_data = []
            for sector, prediction in future_predictions.items():
                final_risks = {scenario: trend[-1] for scenario, trend in prediction['scenarios'].items()}
                total = sum(final_risks.values())
                if total > 0:
                    probs = {scenario: risk/total for scenario, risk in final_risks.items()}
                    risk_probs_data.append({
                        'sector': sector,
                        'probabilities': probs
                    })
            
            if risk_probs_data:
                prob_df = pd.DataFrame([{**{'sector': rp['sector']}, **rp['probabilities']} 
                                      for rp in risk_probs_data])
                prob_df.set_index('sector', inplace=True)
                
                # Use consistent colors for scenarios
                colors = [self.color_palette[col] for col in prob_df.columns]
                ax = prob_df.plot(kind='bar', stacked=True, ax=axes[1,1], 
                                 color=colors, edgecolor='black', zorder=3)
                
                self._AX_(axes,1,1,'Risk Scenario Probabilities (Final 6-Month Outcome)', False, 'Probability',sectors, True)
                
                axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                axes[1,1].set_ylim(0, 1.0)
                
                # Add percentage labels on stacked bars
                self._add_stacked_bar_labels(ax, prob_df)
                
                axes[1,1].set_xlabel('Sector', fontsize=12, color = '#3c3c3c')
            
            plt.tight_layout()
            
            if show:
                plt.show(block=block)
            return fig
            
        except Exception as e:
            print(f"Error creating risk evolution chart: {e}")
            return None
    
    def _add_stacked_bar_labels(self, ax, df):
        """Add percentage labels to stacked bar chart"""
        for container in ax.containers:
            # Skip if there are too many small segments
            if len(container) > 0 and any(height > 0.1 for height in container.datavalues):
                ax.bar_label(container, labels=[f'{v:.1%}' if v > 0.05 else '' for v in container.datavalues],
                           label_type='center', fontsize=8, fontweight='bold', color='white')
    
    def create_interactive_ml_dashboard(self, future_predictions, risk_probabilities):
        """Create enhanced interactive Plotly dashboard for ML predictions"""
        try:
            sectors = list(future_predictions.keys())
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Current vs Predicted Risk',
                    'Risk Evolution Forecast',
                    'Scenario Probabilities', 
                    'Risk Change Analysis'
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # Current vs Predicted with enhanced colors
            current_risks = [fp['current_risk'] for fp in future_predictions.values()]
            predicted_risks = [fp['scenarios'][fp['recommended_scenario']][-1] 
                              for fp in future_predictions.values()]
            
            fig.add_trace(
                go.Bar(name='Current Risk', x=sectors, y=current_risks,
                      marker_color=self.color_palette['current'],
                      text=[f'{score:.3f}' for score in current_risks],
                      textposition='auto'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(name='Predicted Risk (6M)', x=sectors, y=predicted_risks,
                      marker_color=self.color_palette['predicted'],
                      text=[f'{score:.3f}' for score in predicted_risks],
                      textposition='auto'),
                row=1, col=1
            )
            
            # Risk Evolution for selected sector with enhanced styling
            selected_sector = sectors[0]  # Show first sector
            prediction = future_predictions[selected_sector]
            periods = len(prediction['scenarios']['baseline'])
            
            for scenario, trend in prediction['scenarios'].items():
                fig.add_trace(
                    go.Scatter(x=list(range(periods)), y=trend, mode='lines+markers',
                              name=f'{scenario}', 
                              line=dict(dash='dash' if scenario != prediction['recommended_scenario'] else 'solid',
                                      width=3 if scenario == prediction['recommended_scenario'] else 2),
                              marker=dict(size=8 if scenario == prediction['recommended_scenario'] else 6),
                              line_color=self.color_palette[scenario]),
                    row=1, col=2
                )
            
            # Scenario Probabilities with enhanced colors
            scenario_names = list(risk_probabilities[sectors[0]]['probabilities'].keys())
            for scenario in scenario_names:
                probs = [risk_probabilities[sector]['probabilities'][scenario] for sector in sectors]
                fig.add_trace(
                    go.Bar(name=scenario, x=sectors, y=probs,
                          marker_color=self.color_palette[scenario]),
                    row=2, col=1
                )
            
            # Risk Changes with enhanced color scale
            risk_changes = [risk_probabilities[sector]['risk_change'] for sector in sectors]
            
            # Create color scale based on risk change values
            colors = []
            for change in risk_changes:
                if change > 0.05:
                    colors.append(self.risk_colors['HIGH_BUBBLE_RISK'])
                elif change > 0.02:
                    colors.append(self.risk_colors['ELEVATED_RISK'])
                elif change > -0.02:
                    colors.append(self.risk_colors['MODERATE_RISK'])
                elif change > -0.05:
                    colors.append('#90EE90')
                else:
                    colors.append(self.risk_colors['LOW_RISK'])
            
            fig.add_trace(
                go.Bar(x=sectors, y=risk_changes, 
                      marker_color=colors,
                      text=[f'{change:+.3f}' for change in risk_changes],
                      textposition='auto',
                      name='Risk Change'),
                row=2, col=2
            )
            
            # Update layout for better readability
            fig.update_layout(
                height=900,
                title_text="Enhanced ML Bubble Risk Prediction Dashboard",
                title_font_size=20,
                title_x=0.5,
                showlegend=True,
                barmode='group',
                template='plotly_white',
                font=dict(family="Arial", size=12)
            )
            
            # Update axes labels and ranges
            fig.update_yaxes(title_text="Bubble Risk Score", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="Risk Score", range=[0, 1], row=1, col=2)
            fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
            fig.update_yaxes(title_text="Risk Change", range=[-0.2, 0.2], row=2, col=2)
            fig.update_xaxes(title_text="Sectors", row=1, col=1)
            fig.update_xaxes(title_text="Months Ahead", row=1, col=2)
            fig.update_xaxes(title_text="Sectors", row=2, col=1)
            fig.update_xaxes(title_text="Sectors", row=2, col=2)
            
            fig.show()
            
        except Exception as e:
            print(f"Error creating interactive ML dashboard: {e}")