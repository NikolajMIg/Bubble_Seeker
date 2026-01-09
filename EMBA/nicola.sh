#!/bin/sh
python3 -m venv Nicola__env
source Nicola__env/bin/activate
pip3 install -r Nico_Requirements.txt
python3 main_ml.py NICO_BAT_Mode
source Nicola__env/bin/deactivate
echo "Delete your current virtual environment \"Nicola__env\"?"
echo "======================================================"
read -p "Confirm (y/N): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    rm -rf Nicola__env
