# General_ML.py quelques variables et fonctions pouvant être loadée AVANT checker

#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code
# -*- coding: utf-8 -*-

import  platform
import  os

Preliminary_info        =   list   (  []  )
Must_Restart            =   False
From_BAT_File           =   False
MAC_99                  =   0xFF

def detection_os(): #  Retourne le nom du système d'exploitation.
     
    systeme = platform.system()

    #systeme = "Darwin" just for test

    if systeme == "Darwin":
        return "Mac   (Darwin)", 1
    elif systeme == "Windows":
        return "PC   (Windows)", 0
    else:
        return "Neither PC nor MAC (" + systeme + "). The programm could present somme problems", -1
    

def Klean(filename):
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except:
            pass


