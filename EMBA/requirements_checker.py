# requirements_checker.py (New - Simple dependency checker)
"""
Simple requirements checker without external dependencies
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import      subprocess
import      sys
import      os
import      General_ML

def check_and_install_requirements():
    """Check and install required packages"""
    requirements = {
        'yfinance': 'yfinance',
        'pandas': 'pandas', 
        'fpdf':'fpdf',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'sklearn': 'sklearn',
        #'scikit-learn': 'scikit-learn',
        'plotly': 'plotly',
        'seaborn': 'seaborn',# jamais Utilisé???
        'tkinter': 'tkinter'  # Usually comes with Python     pip install scikit-learn
        
    }
    '''
        Le module shutil fait partie de la bibliothèque standard de Python, 
        ce qui signifie qu'il est inclus par défaut lors de l'installation de Python . =>
        inutile de le mettre dans la liste ci-dessus
    '''
    
    General_ML.Preliminary_info.append("Checking dependencies...")

    General_ML.Preliminary_info.append("Checking dependencies...")
    SAP = os.getenv('SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL')
    
    os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = 'True'
    
    
    for package, install_name in requirements.items():
        try:
            if package == 'tkinter':
                # tkinter is usually bundled with Python
                #import tkinter
                General_ML.Preliminary_info.append(f"✓ {package} is available")
            else:
                __import__(package)
                General_ML.Preliminary_info.append(f"✓ {package} is available")
        except ImportError:
            if install_name:
                General_ML.Preliminary_info.append(f"✗ {package} not found. Installing...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install",  install_name ])
                    General_ML.Preliminary_info.append(f"✓ {package} installed successfully")
                    General_ML.Must_Restart            =  True
                except subprocess.CalledProcessError:
                    General_ML.Preliminary_info.append(f"✗ Failed to install {package}")
                # end if install_name
            else:
                General_ML.Preliminary_info.append(f"✗ {package} is required but cannot be installed via pip")
            #end except ImportError:
        # end for package, install_name in requirements.items():
    
    #python -m pip show scikit-learn > scikitlearn.jpm123
    result = subprocess.run([sys.executable, "-m", "pip","show" ,"scikit-learn"], capture_output=True, text=True)
   
    a=len(result.stdout) # arround 350 when it's OK
    '''
    'Name: scikit-learn\nVersion: 1.8.0\nSummary: A set of python modules for machine learning and data mining
    \nHome-page: https://scikit-learn.org\nAuthor: \nAuthor-email: \nLicense-Expression: BSD-3-Clause\nLocation: 
    C:\\Users\\Jean-Pierre Mignot\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\nRequires: joblib, 
    numpy, scipy, threadpoolctl\nRequired-by: \n'
    '''
        
    if a < 10: # si non installé avec result.stdout=''
        subprocess.check_call([sys.executable, "-m", "pip", "install",  'scikit-learn' ])
        General_ML.Must_Restart            =  True
  
    if SAP==None:
        os.environ.pop('SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL')
    else:
        os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = SAP
    #end check_and_install_requirements


