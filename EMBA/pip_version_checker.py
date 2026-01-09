# pip_version_checker.py (Fixed)
"""
Pip Version Checker - Utility to check and update pip to latest version
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import subprocess
import sys
import re
import General_ML

def PR(st):
    General_ML.Preliminary_info.append(st)
    print(st)

class PipVersionChecker:
    def __init__(self):
        self.current_version = None
        self.latest_version = None
        
    def get_current_pip_version(self):
        """Get currently installed pip version"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, check=True)
            
            # Extract version from output like "pip 21.3.1 from ..."
            match = re.search(r'pip\s+([\d.]+)', result.stdout)
            if match:
                self.current_version = match.group(1)
                return self.current_version
            else:
                return None
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
            
    def get_latest_pip_version(self):
        """Get latest available pip version from PyPI"""
        try:
            # Use pip index versions to get available versions
            result = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "pip"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse output to find latest version
                for line in result.stdout.split('\n'):
                    if 'LATEST:' in line:
                        match = re.search(r'LATEST:\s+([\d.]+)', line)
                        if match:
                            self.latest_version = match.group(1)
                            return self.latest_version
            
            # Fallback: try the old method
            result = subprocess.run([sys.executable, "-m", "pip", "install", "pip==999.0.0"], 
                                  capture_output=True, text=True)
            
            # Extract latest version from error message
            for line in result.stderr.split('\n'):
                if 'from versions:' in line:
                    versions_text = line.split('from versions:')[-1].split(')')[0]
                    versions = [v.strip() for v in versions_text.split(',')]
                    if versions:
                        self.latest_version = versions[-1]  # Last version is usually latest
                        return self.latest_version
                        
            return None
            
        except subprocess.CalledProcessError:
            return None
            
    def compare_versions(self, v1, v2):
        """Simple version comparison without external dependencies"""
        def parse_version(v):
            return [int(x) for x in v.split('.')]
        
        try:
            v1_parts = parse_version(v1)
            v2_parts = parse_version(v2)
            
            # Pad with zeros if different length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts < v2_parts
        except:
            return False
            
    def is_update_needed(self):
        """Check if pip update is needed"""
        current = self.get_current_pip_version()
        latest = self.get_latest_pip_version()
        
        if not current or not latest:
            return False, latest
            
        return self.compare_versions(current, latest), latest
        
    def update_pip(self):
        """Update pip to latest version"""
        try:
            PR("Updating pip to latest version...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            PR("pip updated successfully!")
            return True
        except subprocess.CalledProcessError as e:
            PR(f"Failed to update pip: {e}")
            return False
            
    def check_and_update(self):
        """Main method to check and update pip if needed"""
        PR("Checking pip version...")
        
        current = self.get_current_pip_version()
        if not current:
            PR("Could not determine current pip version")
            return False
            
        PR(f"Current pip version: {current}")

        '''
        
        latest = self.get_latest_pip_version()
        if not latest:
            PR("Could not determine latest pip version")
            return False
            
        PR(f"Latest pip version: {latest}")
        '''
        OK, latest = self.is_update_needed()
        if OK:
            PR(f"Update available: {current} -> {latest}")
            return self.update_pip()
        else:
            PR(f"Latest pip version: {latest}")
            PR("Your pip is already up to date!")
            return True

def main_PIP_CHECK():  # renomé. Main un peu dangereux. Il peut y avoir des risques de confusion
    """Standalone pip version checker"""
    checker = PipVersionChecker()
    checker.check_and_update()
