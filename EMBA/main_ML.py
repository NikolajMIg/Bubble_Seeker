# main_ML.py (Updated - Cleaned up plotting code)
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  sys
import  math
import  os  
import  shutil
import  General_ML
import  socket

PC_Connected = True 
try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("www.google.com", 80))
except :  #socket.gaierror, e:
        print ("PC Unconnected to internet: Impossible to run the sofware without this connection ...")
        print('Check your connection and try again')
        PC_Connected = False
finally:
    sock.close()

if PC_Connected:
    N = len(sys.argv)
    print ('N  :',N, sys.argv[0])# C:\...\main_ML.py
    if N > 1:
        A=sys.argv[1]
        #print ('A  :',A)# C:\...\main_ML.py
        if A == 'NICO_BAT_Mode':  # first choosed argument in batch file
            General_ML.From_BAT_File =  True # in that case don't install libreries in main python area
    # Detect OS
    print(General_ML.From_BAT_File)
    st, General_ML.MAC_99  =   General_ML.detection_os()

    print('Your computer  : ' , st)
    CMP_ = "✓ Your computer  : " +  st
    st  =   sys.version   # Your python :  3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]
    print('Your python    : ' , st)
    PYT_="✓ Your python  : " +  st
    st =  st.strip(' ')
    st04 = st[0:4]# 3.13
    st04 =  st04.strip(' ')
    try:
        _V = float(st04)
    except ValueError:
        _V=-1        
        if st[3]=='.': #3.9.
            st1 =st[0] + st[1] + '0' + st[2]   #3.09            
            try:
                _V = float(st1)
            except ValueError:
                _V=-1 

    if _V >0:
        _V  =   0.00001 + _V*100
        _V  =   math.trunc(_V)
        if _V < 313:
            print('Minumum PYTHON version 3.13.x required in order to access multithreading functions')
            os.sys.exit()
        elif  _V==313:
            st04=st[5:7]
            st04= st04.strip()
            try:
                _minor_V = int(st04)
            except ValueError:
                _minor_V=-1 
            if _minor_V < 9: # 9: OK 1: NOT OK   2..8 ?
                print('WARNING: in earlier python 3.13.x version, multithreading present a bug leading to potential problem when leaving the soft, fixed im 3.13.9')

    print('Program loading in progress. Please wait a few seconds...')
 
    General_ML.Preliminary_info.clear()
    General_ML.Preliminary_info.append(CMP_)
    General_ML.Preliminary_info.append(PYT_)
    if not General_ML.From_BAT_File:
        from    pip_version_checker   import  main_PIP_CHECK
        from    requirements_checker  import check_and_install_requirements
        print('main_PIP_CHECK')
        main_PIP_CHECK()
        print('Done')

        print('check_and_install_requirements')
        check_and_install_requirements()
        print('Done')    
    else:
        General_ML.Preliminary_info.append("✓ Ready to start")

    if not General_ML.Must_Restart:
      
        import  Tools
        import  DISTRIB
        from    Tools import Display_info, built_logo      
        from    Tools import built_styl 
        from    Tools import Replace_Unallowed_Char_in_TXT_File
        from    Tools import INFO_001_   

        import  tkinter as tk
        from    tkinter import ttk, messagebox, scrolledtext, filedialog        
        if _V <0:
                response = messagebox.askyesno(
                        'Enable to extract your python version out of "' + st + '"',
                        'Probably you will have somme troubles if you try to continue '            
                        "Do you want to quit the application? "
                        )
                if response:
                    os.sys.exit()

        import  threading
        import  asyncio
        from    datetime import datetime
        import  pandas as pd
        #import  numpy as np

        # Import our modules
        from    Sector_Data     import SectorDataEngine
        from    Analysis        import MultiSectorBubbleAnalyzer
        from    History         import HistoricalComparator
        from    Visualization   import SectorDashboards, MLVisualization
        from    ML_Predictor    import BubbleRiskPredictor
        from    History_ML      import MLHistoryManager
    
        Tools.Tools_QQUUIITT    =   True# en 1er    
    else:
        print('')
        print('')
        print('')
        print('                    New libraries installed. Please restart the application')
        print('                    =======================================================')
        print('')
        print('')
        print('')


class EnhancedBubbleAnalysisGUI:
    def __init__(self):
        #[0..5][0..8]
        self.sectors = {
            Tools.Sektor[0]     : Tools.Sect_Ticket[0],
            Tools.Sektor[1]     : Tools.Sect_Ticket[1],
            Tools.Sektor[2]     : Tools.Sect_Ticket[2],
            Tools.Sektor[3]     : Tools.Sect_Ticket[3],
            Tools.Sektor[4]     : Tools.Sect_Ticket[4],
            Tools.Sektor[5]     : Tools.Sect_Ticket[5]
        }
        #print(self.sectors)
        self.current_analysis = None
        self.current_historical_analysis = None
        self.future_predictions = None
        self.risk_probabilities = None
        
        # Initialize components
        self.visualizer = SectorDashboards()
        self.ml_visualizer = MLVisualization()
        self.ml_predictor = BubbleRiskPredictor()
        self.ml_history_manager = MLHistoryManager()
        
        self.setup_enhanced_gui()
        self.setup_graph_selection()
        
        # Check if we need initial data collection
        self.root.after(1000, self.check_initial_data)
        #print(General_ML.MAC_99)
        if General_ML.MAC_99==1:
            if os.path.isfile(Tools.User_Defined_logo2):
                self.root.iconbitmap(Tools.User_Defined_logo2) 
        elif General_ML.MAC_99==0:
            if os.path.isfile(Tools.User_Defined_logo):
                self.root.iconbitmap(Tools.User_Defined_logo) 
        
    def setup_graph_selection(self):
        """Initialize graph selection settings"""
        self.selected_graphs = {
            'bubble_risk': True,
            'radar': True,
            'detailed': False,
            'metrics': False,
            'historical': False
        }
        self.graph_window = None

    def show_graph_selection(self):
        """Show graph selection dialog"""
        if not self.current_analysis:
            messagebox.showwarning("No Data", "Please run analysis first")
            return

        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Select Graphs to Display")
        #self.graph_window.geometry("300x400")
  
        width_123   =   300
        height_123  =   400

        x_coordinate_123 = Tools.screen_width_123 // 2 - width_123 // 2
        y_coordinate_123 = Tools.screen_height_123 // 2 - height_123 // 2
        self.graph_window.geometry(f"{width_123}x{height_123}+{x_coordinate_123}+{y_coordinate_123}")

        # Bring graph selection window to front
        self.graph_window.attributes('-topmost', True)
        self.graph_window.lift()
        self.graph_window.focus_force()
        
        # Graph selection checkboxes
        ttk.Label(self.graph_window, text="Select Graphs to Display:", 
                  font=('Arial', 10, 'bold')).pack(pady=10)
        
        graphs = [
            ('Bubble Risk Chart', 'bubble_risk'),
            ('Radar Chart', 'radar'), 
            ('Detailed Sector Report', 'detailed'),
            ('Metrics Comparison', 'metrics'),
            ('Historical Indicators', 'historical')
        ]
        
        self.checkbox_vars = {}
        for graph_name, graph_key in graphs:
            var = tk.BooleanVar(value=self.selected_graphs[graph_key])
            self.checkbox_vars[graph_key] = var
            cb = ttk.Checkbutton(self.graph_window, text=graph_name, variable=var)
            cb.pack(anchor=tk.W, padx=20, pady=5)
        
        # Sector selection for detailed report
        ttk.Label(self.graph_window, text="Select Sector for Detailed Report:").pack(pady=10)
        self.detailed_sector_var = tk.StringVar()
        sector_combo = ttk.Combobox(self.graph_window, textvariable=self.detailed_sector_var,
                                   values=list(self.current_analysis.keys()))
        sector_combo.pack(pady=5)
        sector_combo.set(list(self.current_analysis.keys())[0] if self.current_analysis else "")
        
        # Show buttons
        ttk.Button(self.graph_window, text="Show Selected Graphs", 
                   command=self._show_selected_graphs).pack(pady=10)
        ttk.Button(self.graph_window, text="Show All Graphs", 
                   command=self._show_all_graphs).pack(pady=5)
        ttk.Button(self.graph_window, text="Cancel", 
                   command=self.graph_window.destroy).pack(pady=5)
        
        # Remove topmost after a short delay
        self.graph_window.after(100, lambda: self.graph_window.attributes('-topmost', False))

    def _show_selected_graphs(self):
        """Show only selected graphs"""
        # Update selections from checkboxes
        for graph_key, var in self.checkbox_vars.items():
            self.selected_graphs[graph_key] = var.get()
            
        if self.graph_window:
            self.graph_window.destroy()
        
        try:
            if self.selected_graphs['bubble_risk']:
                self.visualizer.create_bubble_risk_chart(self.current_analysis, self)
            
            if self.selected_graphs['radar']:
                self.visualizer.create_radar_chart(self.current_analysis)
            
            if self.selected_graphs['detailed']:
                sector = self.detailed_sector_var.get()
                if sector in self.current_analysis:
                    self.visualizer.create_detailed_sector_report(self.current_analysis, sector, self)
            
            if self.selected_graphs['metrics']:
                self.visualizer.create_metrics_comparison_chart(self.current_analysis)
            
            if self.selected_graphs['historical']:
                self.visualizer.create_historical_indicators_chart(self.current_analysis)
            
            self.log_message("Selected graphs displayed")
            
            # Bring main window back to front after showing plots
            self.bring_to_front()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plots: {str(e)}")

    def _show_all_graphs(self):
        """Show all graphs"""
        if self.graph_window:
            self.graph_window.destroy()
        
        self.selected_graphs = {k: True for k in self.selected_graphs}
        self._show_selected_graphs()

    def bring_to_front(self):
        """Bring the main window to the front"""
        self.root.attributes('-topmost', True)
        self.root.lift()
        self.root.focus_force()
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

    def show_all_plots_simultaneously(self):
        """Show ALL plots simultaneously with comprehensive sector analysis"""
        if not self.current_analysis:
            messagebox.showwarning("No Data", "Please run analysis first")
            return
            
        try:
            self.log_message("\n" + "=" * 80)
            self.log_message("DISPLAYING ALL PLOTS SIMULTANEOUSLY")
            self.log_message("=" * 80)
            
            # Use the visualizer's built-in simultaneous plotting
            self.visualizer.show_plots_simultaneously(self.current_analysis, self)
            
            self.log_message("✓ All plots displayed simultaneously in separate windows")
            self.log_message("✓ Close all plot windows to continue...")
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plots: {str(e)}")
            self.log_message(f"Error displaying plots: {str(e)}")

    def check_initial_data(self):
        while True:
            """Check if we have enough historical data and offer to collect more"""
            training_data = self.ml_history_manager.get_training_data(min_analyses=1)
            total_samples = sum(len(data['features']) for data in training_data.values())
            if total_samples < 294:  # total_samples in ml_historical_data.BAK =294
                if os.path.isfile('ml_historical_data.BAK'):
                    response = messagebox.askyesno(
                        "ML Data Collection, ml_historical_data.json seems to have lossed data",
                        f"Currently only {total_samples} training samples available.\n\n"
                        f"294 samples in ml_historical_data.BAK.\n\n"                        
                        f"Would you like to retrive data from ml_historical_data.BAK now? "
                        )
                    if response:  # OK for retrive
                        fp          =   open('ml_historical_data.BAK', 'r') 
                        f2          =   open('ml_historical_data.json', 'w')
                        while True:
                            line    =   fp.readline()
                            if not line:
                                break
                            f2.write(line)
                        f2.close()
                        fp.close()  
                        self.ml_history_manager.historical_data = self.ml_history_manager._load_historical_data()
                    else: # ml_historical_data.BAK' found nut Not OK to retrive data
                        break
                else: # end if os.path.isfile('ml_historical_data.BAK')
                    break  # 'ml_historical_data.BAK' not found => No retrive possible 
                # end if total_samples < 294:
            else: # total_samples >= 294 => tout est OK
                break
            #end while True

        Tools.Tools_QQUUIITT    =   False  # OK Now Display_info can fully run 
        print('Program runing!')
        for i in range ( 0, len(General_ML.Preliminary_info)):
            Display_info(General_ML.Preliminary_info[i], self)
        if total_samples < 50:  # Minimum for good ML performance
            response = messagebox.askyesno(
                "ML Data Collection",
                f"Currently only {total_samples} training samples available.\n\n"
                f"For accurate ML predictions, we recommend at least 50 samples.\n\n"
                f"Would you like to collect additional data now? "
                f"This will run multiple analyses to build the training dataset.\n\n"
                f"Note: This may take 10-15 minutes."
            )
            
            if response:
                self.collect_initial_data()
    
    def collect_initial_data(self):
        """Collect initial training data by running multiple analyses"""
        self.clear_output()
        self.log_message("STARTING INITIAL DATA COLLECTION FOR ML TRAINING")
        self.log_message("=" * 60)
        self.log_message("This will run multiple sector analyses to build a robust training dataset.")
        self.log_message("This process may take 10-15 minutes. Please be patient...")
        
        def collect_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.collect_data_async())
            finally:
                loop.close()
            
            self.root.after(0, self.data_collection_complete)
            
        thread = threading.Thread(target=collect_async)
        thread.daemon = True
        thread.start()
    
    async def collect_data_async(self):
        """Run multiple analyses to collect training data"""
        try:
            # Initialize components
            data_engine = SectorDataEngine(self.sectors)
            analyzer = MultiSectorBubbleAnalyzer()
            
            target_samples = 50
            current_samples = 0
            
            # Check current data
            training_data = self.ml_history_manager.get_training_data(min_analyses=1)
            current_samples = sum(len(data['features']) for data in training_data.values())
            
            analyses_needed = max(0, (target_samples - current_samples) // len(self.sectors) + 2)
            
            self.log_message(f"Current training samples: {current_samples}")
            self.log_message(f"Target training samples: {target_samples}")
            self.log_message(f"Running {analyses_needed} additional analyses...")
            
            for i in range(analyses_needed):
                self.log_message(f"Running analysis {i+1}/{analyses_needed}...")
                
                # Collect data
                sector_data = await data_engine.collect_all_sector_data(self) # app)
                
                # Analyze sectors
                sector_analysis = await analyzer.analyze_all_sectors(sector_data, self)
                
                # Save to history
                self.ml_history_manager.save_current_analysis(sector_analysis)
                
                # Update progress
                training_data = self.ml_history_manager.get_training_data(min_analyses=1)
                current_samples = sum(len(data['features']) for data in training_data.values())
                
                self.log_message(f"Progress: {current_samples}/{target_samples} samples collected")
                
                # Add delay to avoid rate limiting (2 minutes between analyses)
                if i < analyses_needed - 1:
                    #self.log_message("Waiting 2 minutes before next analysis...")
                    await asyncio.sleep(0.1)# 120)  # 2 minutes delay    why?
            
            self.log_message("Initial data collection completed successfully!")
            self.log_message(f"Total training samples: {current_samples}")
            
        except Exception as e:
            self.log_message(f"ERROR during data collection: {str(e)}")
    
    def data_collection_complete(self):
        """Called when initial data collection is complete"""
        self.log_message("Data collection process finished.")
        messagebox.showinfo("Data Collection Complete", 
                          "Initial data collection completed successfully!\n"
                          "You can now train ML models for more accurate predictions.")
        self.ml_history_manager.historical_data = self.ml_history_manager._load_historical_data()
        self.check_initial_data()
    
    def setup_enhanced_gui(self):
        
        """Setup enhanced GUI with ML features"""
        self.root = tk.Tk()
        self.root.title("Enhanced Multi-Sector Bubble Analysis with ML Prediction")

        # récupération de la taille de l'écran
        if Tools.screen_width_123 <0: #initialisation -1  
            Tools.screen_width_123 = self.root.winfo_screenwidth()
        if Tools.screen_height_123<0:#initialisation -1
            Tools.screen_height_123 = self.root.winfo_screenheight()
        # adjusting window to screen size
        width_123 = Tools.screen_width_123 - 150
        height_123  =  Tools.screen_height_123 - 150
        # calcul position x et y de la fenêtre
        x_coordinate_123 = Tools.screen_width_123 // 2 - width_123 // 2
        y_coordinate_123 = max (0,Tools.screen_height_123 // 2 - height_123 // 2 -20)
        self.root.geometry(f"{width_123}x{height_123}+{x_coordinate_123}+{y_coordinate_123}")
        
        # Make window appear in front initially
        self.root.attributes('-topmost', True)
        self.root.lift()
        self.root.focus_force()
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Enhanced Multi-Sector Bubble Analysis with Machine Learning",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Analyze bubble risks across multiple sectors using comprehensive financial metrics and ML predictions",
                              font=('Arial', 10))
        desc_label.pack(pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Analysis buttons
        analysis_frame = ttk.Frame(control_frame)
        analysis_frame.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(analysis_frame, text="Run Sector Analysis", 
                                     command=self.run_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=2)
        
        self.ml_analyze_btn = ttk.Button(analysis_frame, text="Run ML Prediction", 
                                        command=self.run_ml_prediction, state='disabled')
        self.ml_analyze_btn.pack(side=tk.LEFT, padx=2)
        
        # ML-specific buttons
        ml_frame = ttk.Frame(control_frame)
        ml_frame.pack(side=tk.LEFT, padx=10)
        
        self.train_ml_btn = ttk.Button(ml_frame, text="Train ML Models", 
                                      command=self.train_ml_models_threaded, state='disabled')
        self.train_ml_btn.pack(side=tk.LEFT, padx=2)
        
        self.ml_viz_btn = ttk.Button(ml_frame, text="Show ML Predictions", 
                                    command=self.show_ml_predictions, state='disabled')
        self.ml_viz_btn.pack(side=tk.LEFT, padx=2)
        
        # Visualization buttons
        viz_frame = ttk.Frame(control_frame)
        viz_frame.pack(side=tk.RIGHT, padx=5)
        
        self.select_graphs_btn = ttk.Button(viz_frame, text="Select Graphs", 
                                          command=self.show_graph_selection, state='disabled')
        self.select_graphs_btn.pack(side=tk.LEFT, padx=2)
        
        self.plot_btn = ttk.Button(viz_frame, text="Show All Plots", 
                                  command=self.show_all_plots_simultaneously, state='disabled')
        self.plot_btn.pack(side=tk.LEFT, padx=2)
        
        self.ml_dashboard_btn = ttk.Button(viz_frame, text="ML Dashboard", 
                                          command=self.show_ml_dashboard, state='disabled')
        self.ml_dashboard_btn.pack(side=tk.LEFT, padx=2)
        
        # Export buttons
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(side=tk.RIGHT, padx=10)
        
        self.csv_btn = ttk.Button(export_frame, text="Export to CSV", 
                                 command=self.export_to_csv, state='disabled')
        self.csv_btn.pack(side=tk.LEFT, padx=2)
                
        self.Distrib_btn = ttk.Button(export_frame, text="Distribution analysis", 
                                 command=self.Distibution_Analisys , state='disabled')
        self.Distrib_btn.pack(side=tk.LEFT, padx=2)
        
        self.save_btn = ttk.Button(export_frame, text="Save Results", 
                                  command=self.save_results, state='disabled')
        self.save_btn.pack(side=tk.LEFT, padx=2)
                
        # Progress bar
        if General_ML.MAC_99==0:
            self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
            self.progress.config(maximum=100, mode="indeterminate", orient="horizontal", value=0)
            self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Results notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Current Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, height=25, width=100, 
                                                     font=('Consolas', 9))
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # ML Predictions tab
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="ML Predictions")
        
        self.ml_text = scrolledtext.ScrolledText(self.ml_frame, height=25, width=100, 
                                               font=('Consolas', 9))
        self.ml_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to analyze")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
        
        # Remove topmost after window is shown
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

    def train_ml_models_threaded(self): # appelé par train ml model
        """Train ML models in a separate thread"""
        if General_ML.MAC_99==0:
            self.progress.start()
        self.clear_ml_output()
        self.train_ml_btn.config(state='disabled') 
        self.status_var.set("Training ML models...   Be patient, this operation could take some time ( 4 models to evaluate with best parameters reaserch)")
        self.root.update()
        def train_async():
            success, x, y, CF  = self.train_ml_models()
            self.root.after(5, lambda: self.ml_training_complete(success, x, y, CF))
            
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
    

    def train_ml_models(self): # train ML model
        """Train ML models using historical data with feature consistency"""
        Display_info("TRAINING MACHINE LEARNING MODELS", self)
        Display_info("=" * 50, self)
        
        try:
            # Get training data with minimum 1 analysis required
            training_data = self.ml_history_manager.get_training_data(min_analyses=1) # history_ML
            
            if not training_data:
                Display_info("ERROR: No historical data available for training.", self)
                Display_info("INFO: Please run sector analysis at least 2 times to accumulate data.", self)
                Display_info("INFO: Each analysis creates training pairs (current -> next analysis).", self)
                return False, None, None, None
            
            # Check if we have enough data
            total_samples = 0
            for sector, data in training_data.items():
                total_samples += len(data['features'])
            
            Display_info(f"INFO: Found {total_samples} total training samples across {len(training_data)} sectors", self)
            
            if total_samples < 1:
                Display_info("ERROR: Insufficient training data. Need at least 1 training sample.", self)
                Display_info("INFO: Run sector analysis at least 2 times to accumulate data.", self)
                return False, None, None, None
            
            # Prepare features and targets
            all_features = []
            all_targets = []
            
            for sector, data in training_data.items():
                if len(data['features']) > 0:
                    # Convert features to proper format
                    features_list = data['features']
                    targets_list = data['targets']
                    
                    # Create DataFrame with proper column names
                    feature_columns = ['overall', 'valuation', 'momentum', 'sentiment', 'fundamental']
                    
                    # Ensure we have the right number of features
                    validated_features = []
                    for feature_vec in features_list:
                        if len(feature_vec) == len(feature_columns):
                            validated_features.append(feature_vec)
                        else:
                            Display_info(f"WARNING: Feature vector length mismatch in {sector}: expected {len(feature_columns)}, got {len(feature_vec)}", self)
                    
                    if validated_features:
                        features_df = pd.DataFrame(validated_features, columns=feature_columns)
                        targets_series = pd.Series(targets_list[:len(validated_features)])
                        
                        all_features.append(features_df)
                        all_targets.append(targets_series)
                        
                        Display_info(f"SUCCESS: {sector}: {len(validated_features)} valid training samples", self)
            
            if not all_features:
                Display_info("ERROR: No valid training data after validation.", self)
                return False, None, None, None
            
            # Combine all sector data
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            
            Display_info(f"INFO: Final training dataset: {combined_features.shape[0]} samples, {combined_features.shape[1]} features", self)
            Display_info(f"INFO: Features: {list(combined_features.columns)}", self)
            
            # Display sample of training data
            Display_info(f"INFO: Sample training data:", self)
            Display_info(f"      Features shape: {combined_features.shape}", self)
            Display_info(f"      Targets shape: {combined_targets.shape}", self)
            Display_info(f"      Target range: {combined_targets.min():.3f} to {combined_targets.max():.3f}", self)
            
            # Train models
            Display_info("INFO: Starting model training...", self)
            x, y = self.ml_predictor.train_models_ini_123(combined_features, combined_targets, self)
                            
            # Enable prediction button
            self.ml_analyze_btn.config(state='normal')

            return True, x, y, combined_features
            
        except Exception as e:
            Display_info(f"ERROR during ML training: {str(e)}", self)
            import traceback
            Display_info(f"Detailed error: {traceback.format_exc()}", self)
            return False, None, None, None
        
    def ml_training_complete(self, success, x, y, CF):
        from Tools import JPM123        
        """Called when ML training is complete"""
        self.train_ml_btn.config(state='normal')
        
        if (len(x) >0) and (len(y) >0):    
            if General_ML.MAC_99==0:
                self.progress.start()        
            self.ml_predictor.is_trained, Tools.PLT_123 = JPM123(CF, x, y, self)
            if self.ml_predictor.is_trained:  
                if General_ML.MAC_99==0:
                   self.progress.stop()  
                if Tools.PLT_123!= None:
                    Tools.PLT_123.show()
        
        if success:
            self.status_var.set("ML models trained successfully")
            self.ml_viz_btn.config(state='normal')
            self.ml_dashboard_btn.config(state='normal')
        else:
            self.status_var.set("ML training failed")
        if General_ML.MAC_99==0:
            self.progress.stop()
    
    def run_analysis(self):
        """Run analysis in a separate thread"""
        self.clear_output()
        self.analyze_btn.config(state='disabled')
        if General_ML.MAC_99==0:
            self.progress.start()
        self.status_var.set("Analyzing sectors...")
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_analysis_async())
            finally:
                loop.close()
                
            self.root.after(0, self.analysis_complete)
            
        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    async def run_analysis_async(self):
        """Run the analysis asynchronously"""
        try:
            self.log_message("STARTING MULTI-SECTOR BUBBLE ANALYSIS")
            self.log_message("=" * 60)
            
            # Initialize components
            data_engine             =   SectorDataEngine(self.sectors)            
            analyzer                =   MultiSectorBubbleAnalyzer()
            historical_comparator   =   HistoricalComparator()
            
            # Collect data
            self.log_message("Collecting sector data...")
            sector_data = await data_engine.collect_all_sector_data(self) # app)
            #print(sector_data)   CSV
            
            # Analyze sectors
            self.log_message("Analyzing bubble risks...")
            sector_analysis = await analyzer.analyze_all_sectors(sector_data, self)
            
            # Historical comparison
            self.log_message("Performing historical comparisons...")
            historical_analysis = await historical_comparator.compare_with_historical_bubbles(sector_analysis)
            self.Tag_OK_Last_Line_N()
            
            # Store results
            self.current_analysis = sector_analysis
            self.current_historical_analysis = historical_analysis
            
            # Save to ML history
            self.ml_history_manager.save_current_analysis(sector_analysis)
            
            # Display results
            self.display_results(sector_analysis, historical_analysis)
            
            # Enable buttons
            self.root.after(0, self.enable_analysis_buttons)

            Tools.PUSH_in_CSV(self)
            
            self.log_message("ANALYSIS COMPLETE")
                        
        except Exception as e:
            self.log_message(f"ERROR during analysis: {str(e)}")
            import traceback
            self.log_message(f"Detailed error: {traceback.format_exc()}")
            
    def analysis_complete(self):
        """Called when analysis is complete"""
        if General_ML.MAC_99==0:
            self.progress.stop()
        #self.analyze_btn.config(state='normal')
        self.status_var.set("Analysis complete")
        
        # Enable ML training button since we now have data
        self.train_ml_btn.config(state='normal')
    
    def enable_analysis_buttons(self):
        """Enable buttons after successful analysis"""
        self.csv_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.select_graphs_btn.config(state='normal')
        self.plot_btn.config(state='normal')
        self.train_ml_btn.config(state='normal')
    
    def run_ml_prediction(self):
        """Run ML prediction in a separate thread"""
        if not self.current_analysis:
            messagebox.showwarning("No Data", "Please run sector analysis first")
            return
            
        self.clear_ml_output()
        self.ml_analyze_btn.config(state='disabled')
        if General_ML.MAC_99==0:
          self.progress.start()
        self.status_var.set("Running ML predictions...")
        
        def predict_async():
            success = self.run_ml_prediction_sync()
            self.root.after(0, lambda: self.ml_prediction_complete(success))
            
        thread = threading.Thread(target=predict_async)
        thread.daemon = True
        thread.start()
    
    def run_ml_prediction_sync(self):
        """Run ML prediction synchronously"""
        try:
            Display_info("RUNNING ML BUBBLE RISK PREDICTIONS", self)
            Display_info("=" * 50, self)
            
            if not self.ml_predictor.is_trained:
                Display_info("ERROR: ML models not trained. Please train models first.", self)
                return False
            
            # Prepare features for prediction
            features = self.ml_predictor.prepare_features(self.current_analysis)
            
            # Generate predictions
            self.future_predictions = {}
            self.risk_probabilities = {}
            
            for sector_name in self.current_analysis.keys():
                if sector_name in features.index:
                    sector_features = features.loc[[sector_name]]
                    prediction = self.ml_predictor.predict_future_risk(sector_features.values[0])
                    self.future_predictions[sector_name] = prediction
                    
                    # Calculate risk probabilities
                    final_risks = {scenario: trend[-1] for scenario, trend in prediction['scenarios'].items()}
                    total_risk = sum(final_risks.values())
                    probabilities = {scenario: risk/total_risk for scenario, risk in final_risks.items()}
                    
                    self.risk_probabilities[sector_name] = {
                        'probabilities': probabilities,
                        'risk_change': prediction['scenarios'][prediction['recommended_scenario']][-1] - prediction['current_risk']
                    }
            
            # Display predictions
            self.display_ml_predictions()
            return True
            
        except Exception as e:
            Display_info(f"ERROR during ML prediction: {str(e)}", self)
            import traceback
            Display_info(f"Detailed error: {traceback.format_exc()}", self)
            return False
    
    def ml_prediction_complete(self, success):
        """Called when ML prediction is complete"""
        if General_ML.MAC_99==0:
            self.progress.stop()
        self.ml_analyze_btn.config(state='normal')
        
        if success:
            self.status_var.set("ML predictions complete")
            self.ml_viz_btn.config(state='normal')
            self.ml_dashboard_btn.config(state='normal')
        else:
            self.status_var.set("ML prediction failed")
    
    def display_results(self, sector_analysis, historical_analysis):
        """Display analysis results"""
        self.log_message("\n" + "=" * 80)
        self.log_message("SECTOR BUBBLE RISK ANALYSIS RESULTS")
        self.log_message("=" * 80)
        
        # Sort sectors by bubble risk
        sorted_sectors = sorted(sector_analysis.items(), 
                               key=lambda x: x[1]['composite_scores']['overall']['score'], 
                               reverse=True)
        
        for sector_name, analysis in sorted_sectors:
            overall_score = analysis['composite_scores']['overall']['score']
            risk_level = analysis['composite_scores']['overall']['risk_level']
            
            Display_info(f"\n   {sector_name.upper()} SECTOR", self)
            Display_info("-" * 40, self)
            Display_info(f"                    Overall Bubble Score: {overall_score:.2f}/1.0", self)
            Display_info(f"                    Risk Level: {risk_level}", self)
            Display_info(f"                    Recommendation: {analysis['risk_assessment']['recommendation']}", self)
    
    def display_ml_predictions(self):
        """Display ML prediction results"""
        Display_info("\n" + "=" * 80, self)
        Display_info("ML BUBBLE RISK PREDICTIONS (6-Month Forecast)", self)
        Display_info("=" * 80, self)
        
        for sector_name, prediction in self.future_predictions.items():
            current_risk = prediction['current_risk']
            predicted_risk = prediction['scenarios'][prediction['recommended_scenario']][-1]
            risk_change = predicted_risk - current_risk
            
            Display_info(f"\n{sector_name.upper()} SECTOR", self)
            Display_info("-" * 40, self)
            Display_info(f"Current Risk: {current_risk:.3f}", self)
            Display_info(f"Predicted Risk (6M): {predicted_risk:.3f}", self)
            Display_info(f"Risk Change: {risk_change:+.3f}", self)
            Display_info(f"Recommended Scenario: {prediction['recommended_scenario']}", self)
            
            # Show probabilities
            if sector_name in self.risk_probabilities:
                probs = self.risk_probabilities[sector_name]['probabilities']
                Display_info("Scenario Probabilities:", self)
                for scenario, prob in probs.items():
                    Display_info(f"  {scenario}: {prob:.1%}", self)
    
    def show_ml_predictions(self):
        """Show ML prediction visualizations"""
        if not self.future_predictions:
            messagebox.showwarning("No Predictions", "Please run ML prediction first")
            return
            
        try:
            self.ml_visualizer.create_risk_evolution_chart(self.future_predictions, self.current_analysis)
            Display_info("ML prediction charts displayed", self)
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create ML charts: {str(e)}")
    
    def show_ml_dashboard(self):
        """Show interactive ML dashboard"""
        if not self.future_predictions or not self.risk_probabilities:
            messagebox.showwarning("No Predictions", "Please run ML prediction first")
            return
            
        try:
            self.ml_visualizer.create_interactive_ml_dashboard(self.future_predictions, self.risk_probabilities)
            Display_info("Interactive ML dashboard opened in browser", self)
        except Exception as e:
            messagebox.showerror("Dashboard Error", f"Failed to create ML dashboard: {str(e)}")
    
    def Distibution_Analisys(self):
        General_ML.Preliminary_info.clear()
        DISTRIB.Distrib_Start(self.root)

    def export_to_csv(self):
        if os.path.isfile(Tools.tempory_csv):
            try:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    title="Export results to CSV"
                    )
            
                if filename:
                    fp          =   open(Tools.tempory_csv, 'r') 
                    f2          =   open(filename, 'w')
                    while True:
                        line    =   fp.readline()
                        if not line:
                            break
                        f2.write(line)
                    f2.close()
                    fp.close()
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV: {str(e)}")
            
        else:
            Tools.Msg("Warning", 
                "Your data was detected as unconsistent...\n" +
                "Saved CSV will be unorganized",
                'warning'
                )
                
            Excel_Csv_Access    =       pd.DataFrame()
            """Export results to CSV"""
            if not self.current_analysis:
                messagebox.showwarning("No Data", "Please run analysis first")
                return
            
            try:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    title="Export results to CSV"
                )
            
                if filename:
                    # Implementation would go here

                    Excel_Csv_Access=pd.DataFrame(self.current_analysis)
                    Excel_Csv_Access.to_csv( # Excel_Csv_Access.to_excel(output_file) devrait marcher ....  
                        filename,     # pour faire un xlsx mais ......???
                        sep = ';'
                        )

                    self.log_message(f"Results exported to: {filename}")
                    messagebox.showinfo("Export Successful", f"Data exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV: {str(e)}")
            #end de else:
        #end export_to_csv
    
    def save_results(self):
        """Save results to JSON"""
        if not self.current_analysis:
            messagebox.showwarning("No Data", "Please run analysis first")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save results"
            )
            
            if filename:
                                
                cur_inp = self.analysis_text.get("1.0", tk.END)
                # some specials chars are not supported  in .jsno/.txt files and have to be replaced by 'usuel' chars
                cur_inp1 = Replace_Unallowed_Char_in_TXT_File(cur_inp) 
                
                fl = open(filename, "w")
                fl.write(cur_inp1)
                fl.close()

                FNA = Tools.Copy_filename_in_PDF(filename, cur_inp, False)

                self.log_message(f"Results saved to: {filename}  &  {FNA}")
                messagebox.showinfo("Save Successful", f"Results saved to {filename}  &  {FNA}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")
                
    def log_message(self, message):    
        """Add message to analysis output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_text.insert(tk.END, f"[{timestamp}] {message}\n")   #scrolledtext
        self.analysis_text.see(tk.END)
        self.root.update()
        
    def supprimer_ligne(self, n):
        debut = f"{n}.0"
        fin = f"{n+1}.0"
 
        self.analysis_text.delete(debut, fin)
        # self.root.update()
        
    def Get_ligne(self, n):
        debut = f"{n}.0"
        fin = f"{n+1}.0"
 
        return self.analysis_text.get(debut, fin)
        # self.root.update() 
 
    def modifier_ligne(self, n, nouveau_texte):
        debut = f"{n}.0"
        fin = f"{n}.end" 
 
        self.analysis_text.delete(debut, fin)
 
        self.analysis_text.insert(debut, nouveau_texte)

    def get_line_count(self):

        last_index = self.analysis_text.index('end-1c')
 
        num_lines = int(last_index.split('.')[0])
 
        return num_lines

    def Tag_OK_Last_Line_N(self):          

        NP=self.get_line_count()
        
        A = self.Get_ligne(NP-1)
        
        if A[len(A)-1]=='\n':
            A=A[:-1]   # remove \n  char
        self.modifier_ligne( NP-1, A + '        ✔')

    def log_ml_message(self, message):
        """Add message to ML output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ml_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.ml_text.see(tk.END)
        self.root.update()
    
    def clear_output(self):
        """Clear analysis output"""
        self.analysis_text.delete(1.0, tk.END)
        Display_info(Tools.INFO_001, self)
    
    def clear_ml_output(self):
        """Clear ML output"""
        self.ml_text.delete(1.0, tk.END)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()  

def Quit_Application():
     if messagebox.askokcancel("Quit", "Do you really want to quit the application ?"):
        #remove tempory files
        General_ML.Klean(Tools.Mon_Modele_STYLE1)
        General_ML.Klean(Tools.Mon_Modele_STYLE2)
        General_ML.Klean(Tools.User_Defined_logo)
        General_ML.Klean(Tools.tempory_csv)

        shutil.rmtree(Tools.analysis_cache_path, ignore_errors=True) # If directory does not exist, you get a No such file or directory error.(dispayded if ignore_errors=False)
        
        # Q U I T!
        print('Application stopped')
        sys.exit()
        #app.root.destroy()
     else:
         Tools.Tools_QQUUIITT    =   False

def on_window_close_via_X():
    Tools.Tools_QQUUIITT    =   True # close Display_info full access to avoid to use app.log_message when destoying app!
    Quit_Application()

def Start_SW():
    Tools.INFO_001 = INFO_001_()
    built_styl()
    built_logo()
    General_ML.Klean(Tools.tempory_csv)

if __name__ == "__main__":
    if PC_Connected and not General_ML.Must_Restart:
        Start_SW()
        Tools.app = EnhancedBubbleAnalysisGUI()
        Tools.app.root.protocol("WM_DELETE_WINDOW", on_window_close_via_X) # Don't stop immediatly but 1st 1sr run on_window_close_via_X()
        Tools.app.run()