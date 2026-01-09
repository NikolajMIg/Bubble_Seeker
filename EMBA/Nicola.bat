Python -m venv Nicola__env
cd Nicola__env
cd scripts
call activate.bat
cd..
cd..
pip3 install -r Nico_Requirements.txt
rem pip3 install scipy
rem pip3 install seaborn
rem pip3 install python-dateutil
rem pip3 install matplotlib
rem pip3 install scikit-learn
rem pip3 install fpdf
rem pip3 install pandas
rem pip3 install yfinance
rem pip3 install tk
rem pip3 install plotly
rem pip3 install seaborn
rem pip3 freeze > requirements.txt
Python main_ml.py NICO_BAT_Mode
cd Nicola__env
cd scripts

call deactivate.bat
cd..
cd..


echo Delete your current virtual environment "Nicola__env"?
echo ======================================================

rmdir /s Nicola__env
