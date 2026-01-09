# DISTRIB.py (New - Simple dependency checker)
"""
Simple requirements checker without external dependencies
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  os
import  tkinter                 as tk
from    turtle                  import bgcolor

from    matplotlib              import legend
from    numpy                   import dtype
import  Tools
from    tkinter                 import  ttk
from    tkinter                 import  *
from    matplotlib.transforms   import  Bbox
import  matplotlib.pyplot       as      plt
from    Mathematik              import  Weighting_POLY
from    Mathematik              import  Poly_Fit_XX_YY_deg      
from    Mathematik              import  Make_Y_model           
from    Mathematik              import  sort             
from    Mathematik              import  smooth      
from    Mathematik              import  Compute_Splines  
import  General_ML

import  math
from    math                    import  * 


Seelc                   =   list (   [ list ([]) ] )
Plot_Trace_selected     =   list (   [ list ([]) ] )   # =   tk.IntVar()   # checked => 1    Unchecked => 0

def onRadioButton_FIX_LIN_LOG_EXPO():
    i   = Tools.Choix_History.get()    # '0' '1' ou '2'
    j   = Tools.selection.get()
    XX = Tools.FIX_LIN_LOG_EXPO
    if j==0:   # normally always TRUE 
        Tools.FIX_LIN_LOG_EXPO= int(i)   
    if XX != Tools.FIX_LIN_LOG_EXPO:
        Tools.Enable_disable(Tools.Button_Distibution, False)
    #print('Older history (FIX_LIN_LOG_EXPO) : ', Tools.FIX_LIN_LOG_EXPO)

def Expo_Mode():
    XX = Tools.Expo_Mode_
    i = Tools.Plot_expo_selected.get() 
    j   = Tools.selection.get()
    if j==0:   # normally always TRUE 
        Tools.Expo_Mode_ = int(i) 
    if XX != Tools.Expo_Mode_:
        Tools.Enable_disable(Tools.Button_Distibution, False)
    #print('Expo Mode   (Expo_Mode_): ', Tools.Expo_Mode_)

def Closed_or_close_star_volume():
    XX = Tools.Closed_or_close_star_volume_
    i = Tools.Plot_Closed_or_close_star_volume_selected.get() 
    j   = Tools.selection.get()
    if j==0:   # normally always TRUE 
        Tools.Closed_or_close_star_volume_ = int(i) ==1 
    if XX != Tools.Closed_or_close_star_volume_:
        Tools.Enable_disable(Tools.Button_Distibution, False)
    #print('Tools.Closed_or_close_star_volume_', Tools.Closed_or_close_star_volume_)
    #print('Expo Mode   (Expo_Mode_): ', Tools.Expo_Mode_)

def Mean_Values_Chk():
    XX = Tools.Mean_Values__selected
    i = Tools.Plot_Mean_selected.get() 
    #print(i, XX)
    j   = Tools.selection.get()
    if j==0:   # normally always TRUE 
        Tools.Mean_Values__selected = int(i) ==1 
    if XX != Tools.Mean_Values__selected:
        Tools.Enable_disable(Tools.Button_Distibution, False)
    #print(Tools.Mean_Values__selected)
    #print('Expo Mode   (Expo_Mode_): ', Tools.Expo_Mode_)

def onRadioButton_Day_Week_Month():
    i   = Tools.Choix_Average.get()    # '0' '1' ou '2'
    j   = Tools.selection.get()
    if j==0:   # normally always TRUE 
        Tools.Average_lev = int(i) 
    #print('Average D/W/M (Average_lev): ' , Tools.Average_lev)

def on_window_close_via_X2():
    if Tools.Subwindow2 != None:
        Tools.Subwindow2.destroy()
        Tools.Subwindow2=None

def End_Analyse():
    if Tools.Subwindow___ != None:
        on_window_close_via_X2() # Subwindow2 => Sub window of Subwindow___  => close Subwindow2 if necessary when closing Subwindow
        Tools.Subwindow___.destroy()
        Tools.Subwindow___ = None
    for i in range ( 0, len(General_ML.Preliminary_info)):
        Tools.Display_info(General_ML.Preliminary_info[i], Tools.app)
            
def on_window_close_via_X():
    End_Analyse()

def PP_p(a, MX, MY):
    Tools.X_Split.append(MX/a)
    Tools.Y_Split.append(MY/a)
    
def YVL(i, j, n, b):                    
    X = Tools.Value_at_Close[i][j][n]
    if b:
        return X
    else:
        return X* Tools.Volume_at_dt[i][j][n]

def MAKE_XY_split_detail():
    Tools.X_Split2.clear()
    Tools.Y_Split2.clear()
    Tools.LAB_88.clear()
    a=0
    
    for h in range(0, len(Tools.X_Split)):
        t=Tools.X_Split[h] 
        while True :
            if a < len(Tools.DDT_ALL):
                if Tools.DDT_ALL[a]>=t:
                    break
                a+=1
                
        for i in range(0, len(Tools.Sektor)):
            for j in range(0, len(Tools.Sect_Ticket[i])):
                if  Plot_Trace_selected[i][j].get()==1:
                    Tools.X_Split2.append(t) 
                    Tools.Y_Split2.append(Tools.Value_at_Close[i][j][a])
                    if h==0:
                        Tools.LAB_88.append(Tools.Sektor[i] + '/' + Tools.Sect_Ticket[i][j])
    #print(Tools.LAB_88)                    
    #end  MAKE_XY_split_detail  
    
def Check_NN():
    NN = 0 
    for i in range(0, len(Tools.Sektor)):
        for j in range(0, len(Tools.Sect_Ticket[i])):
            if Plot_Trace_selected[i][j].get()==1:
                NN+=1
    if NN==0:
        Tools.Msg("Warning", 
                "Your have to select at least 1 curve !!!",
                'warning')
        return 0  
    Tools.Y_Raw.clear()
    Tools.X_Raw.clear()
    Tools.N_DDT_ALL_LEN = len(Tools.DDT_ALL)
    Tools.X_Split.clear()
    Tools.Y_Split.clear()  
    for i in range(0, Tools.N_DDT_ALL_LEN):
        Tools.Y_Raw.append(0)
        Tools.X_Raw.append(0)

    for i in range(0, len(Tools.Sektor)):
        for j in range(0, len(Tools.Sect_Ticket[i])):
            if  Plot_Trace_selected[i][j].get()==1:
                for n in range(0, Tools.N_DDT_ALL_LEN):
                    Tools.Y_Raw[n] += Tools.Value_at_Close[i][j][n]                    
    for n in range(0, Tools.N_DDT_ALL_LEN):
        Tools.Y_Raw[n] = Tools.Y_Raw[n] / NN
        Tools.X_Raw[n] = Tools.DDT_ALL[n]
    if Tools.Average_lev ==0: # per day
        for n in range(0, Tools.N_DDT_ALL_LEN):
            Tools.X_Split.append(Tools.X_Raw[n])
            Tools.Y_Split.append(Tools.Y_Raw[n])
    elif Tools.Average_lev==1: # per week        
        a=0
        MX =0
        MY =0
        
        for n in range(0, Tools.N_DDT_ALL_LEN):           
            MX += Tools.X_Raw[n]
            MY += Tools.Y_Raw[n]
            a+=1
            if a==7:
                PP_p(a, MX, MY)
                a=0
                MX =0
                MY =0
        if a >0:  # il rese encore quelques jous
            PP_p(a, MX, MY)
    elif Tools.Average_lev==2: # per moi        
        old_m=-1
        a=0
        for n in range(0, Tools.N_DDT_ALL_LEN):
            m = Tools.DDT_M[n]
            if m != old_m:
                if a >0:
                    PP_p(a, MX, MY)
                MX  = 0
                MY  = 0
                a   = 0
                old_m = m
            a+=1
            MX += Tools.X_Raw[n]
            MY += Tools.Y_Raw[n]
        if a >0: # il rese encore quelques jous
            PP_p(a, MX, MY)              
    MAKE_XY_split_detail() 
    Tools.NNVAL8 = NN
    return NN 


def onRadioButton_deg_Change():
    i = Tools.Choix_deg_poly.get()    # '1' ou '2'
    j= Tools.selection.get()
    XX = Tools.Poly_deg
    if j==0:   # normally always TRUE 
        Tools.Poly_deg = int(i)    
    if XX != Tools.Poly_deg:
        Tools.Enable_disable(Tools.Button_Distibution, False)
    #print('Poly deg : ' , Tools.Poly_deg)
    Tools.HID(Tools.Frame_SMTH)
    #end onRadioButton_deg_Change

def action_More_M_No_Event(): 
    st = Tools.listeCombo_AditiveMonth.get()  # ' n   =   ' + 0,1,2,3,.., 12* MAX_YEARS_EXTEND = 240
    st = st[-3:] # les 2 derniers caractères
    st.strip() # Removes leading and trailing characters (whitespace by default).
    Tools.More_M = int(st)
    #print('Tools.More_M',Tools.More_M)
    #end action_More_M

def action_More_M(event): 
    action_More_M_No_Event()

def action_kill_last_No_Event():
    st = Tools.listeCombo_Remove_last_Month.get()  # ' n   =   ' + 0,1,2,3,..,12* MAX_YEARS_EXTEND = 240
    st = st[-3:] # les 3 derniers caractères
    st.strip() # Removes leading and trailing characters (whitespace by default).
    Tools.Kill_last = int(st)
    #print('Tools.Kill_last',Tools.Kill_last)
    #end action_kill_last

def action_kill_last(event):
    action_kill_last_No_Event()

def action_kill_1st_No_Event():
    st = Tools.listeCombo_Remove_1st_Month.get()  # ' n   =   ' + 0,1,2,3,..,12* MAX_YEARS_EXTEND = 240
    st = st[-3:] # les 3 derniers caractères
    st.strip() # Removes leading and trailing characters (whitespace by default).
    Tools.Kill_1st = int(st)
    #print('Tools.Kill_1st',Tools.Kill_1st)
    #end action_kill_1st

def action_kill_1st(event):
    action_kill_1st_No_Event()

def Listbox_infos_Add(s):
    General_ML.Preliminary_info.append(s)
    return
    
def Buit_matrx_(More_M, dg , XX__IN, YY__IN, USED_EXPO, SMALL):
    XX_for_Dif              =   list( [] )
    Result                  =   False   
    N_key_point             =   len(XX__IN)
    #accept data within the range according to the user selection Kill_1st/Kill_last
    XX_for_Dif.clear()
    Tools.YY_for_Dif.clear()
    for i in range(0, N_key_point):
        if (XX__IN[i]>=XX__IN[0]+Tools.Kill_1st/12) and (XX__IN[i]<=XX__IN[N_key_point-1]-Tools.Kill_last/12):
            XX_for_Dif.append(XX__IN[i])
            Tools.YY_for_Dif.append(YY__IN[i])
    N_key_point             =   len(XX_for_Dif) # effective number of point to tit the polynome
    if dg >= N_key_point-1:
        Listbox_infos_Add("pas assez de point clef pour le degré pôlynomial souhaité")
    else:
        if USED_EXPO:  # perform a LOG data transformation before fit
            OK  =   True    # assume success
            for i in range(0, N_key_point):
                if Tools.YY_for_Dif[i] < 1e-10:
                    OK=False # impossible to generate a log if data <=0 (here 1e-10 to respect the MATH.log function behaviour)
            if OK:
               for i in range(0, N_key_point):
                   Tools.YY_for_Dif[i]=log(Tools.YY_for_Dif[i])
            else:
                Tools.Msg (
                "Information", 
                "You can select 'expo model'\nwith non positive (or too small) values", 
                'warning'
                )
        #fit polynome degré dg dess points (XX_for_Dif, YY_for_Dif))
        # OK=True if OK alors coefs dans coef
        # elles il y a eu un problème reporté dans le string ERROR_FIT
        Tools.PCT_03  =   Tools.SCALE_Pct_dans_filtre_older_def.get()   # normalement inutile mais...
        onRadioButton_FIX_LIN_LOG_EXPO()                                # meme remarque
        OK, Tools.ERROR_FIT, Tools.coef = Poly_Fit_XX_YY_deg( dg, XX_for_Dif, Tools.YY_for_Dif, 
                        Tools.FIX_LIN_LOG_EXPO, Tools.PCT_03)
        
        if OK:  # fit success!     
            m                       =   0
            ki2_points_clef         =   0  
            for i in range(0, N_key_point):   
                x                   =   XX_for_Dif[i]
                y                   =   Tools.YY_for_Dif[i]
                m                   =   max(m, abs(y))
                s                   =   Make_Y_model(x, dg, Tools.coef)
                if USED_EXPO:   
                    Tools.YY_for_Dif[i]   =   exp(s)-exp(y)  # return back to the true values
                else:
                    Tools.YY_for_Dif[i]   =   s-y                    
                ki2_points_clef     =   ki2_points_clef + ( Tools.YY_for_Dif[i] ** 2 )
            ki2_points_clef         =   ki2_points_clef/N_key_point
            if ki2_points_clef>1e-100:
                ki2_points_clef     =   math.sqrt(ki2_points_clef)
            else:
                ki2_points_clef     =   0
            st_dev                  =   'standard deviation   σ =  ' + str(ki2_points_clef)
            Listbox_infos_Add(st_dev)
                            
            XX_fit_Draw=list([])
            YY_fit_Draw=list([])
            XX_fit_Draw.clear()
            YY_fit_Draw.clear()
             
            XX_fit2_Draw=list([])
            YY_fit2_Draw=list([])
            XX_fit2_Draw.clear()
            YY_fit2_Draw.clear()
              
            for j in range(0,251):
                x       =   XX_for_Dif[0] + (XX_for_Dif[N_key_point-1]-XX_for_Dif[0])*j/250
                XX_fit_Draw.append(x)
                Vl      =   Make_Y_model(x, dg, Tools.coef)
                if USED_EXPO:
                    Vl  =   exp(Vl)
                YY_fit_Draw.append(Vl)
            if More_M > 0:
                for j in range(0,101): # add More_M months after final date
                   x        =   XX_for_Dif[N_key_point-1] + More_M/12 *j/100
                   XX_fit2_Draw.append(x)
                   Vl       =   Make_Y_model(x, dg, Tools.coef)                
                   m        =   max(m, abs(Vl))   # better standard deviation flag positioning
                   if USED_EXPO:
                       Vl   =   exp(Vl)
                   YY_fit2_Draw.append(Vl) 
            if USED_EXPO:
                m   =   exp(m)  # for a correct standard deviation box positioning
            plt.text(XX_for_Dif[0] + (XX_for_Dif[N_key_point-1]-XX_for_Dif[0])/3,0.9*m,  st_dev, 
                     ha="center", va="center", size=16, rotation=0,
                     bbox=dict(boxstyle="round",
                                ec=(1., 0.5, 0.5), #RGB frame  0=> 0 1=> 0xFF
                                fc=(1., 0.8, 0.8), #RGB background (light red)
                              )
                     )
            Listbox_infos_Add('Coefficient du polynôme de fit :   ')
            for i in range(0, len(Tools.coef)):
                st = Tools.Form_EXP(Tools.coef[i]) 
                Listbox_infos_Add(Tools.ajuste('Coef[' + str(i) + ']   =   ' + st))
            st = '∑ c[i] * t^i    i=0..' + str(dg)
            if USED_EXPO:
                st='Fit :    Y =  expo( ' + st + ' )'
            else:
                st='Fit :    Y =  ' + st
            Listbox_infos_Add(Tools.ajuste(st))
            titre="Mean Raw data [ &  " + st + " ] :\n"
            st=''
            for i in range(0, dg+1) :
                KOF=Tools.Form_EXP(Tools.coef[i])
                st = st +  KOF + ' * t^'+ str(i) + '  '
                if (1+i) % 5 ==0:
                    st=st + '\n'
            st    =   st[:-1]     # remove '+'/'-' 1st  char
            plt.title(titre + st)
            if SMALL:
                plt.plot(XX_fit_Draw,YY_fit_Draw,'-r', label="Fit")  # (:pointillé) -: ligne , rouge)
            else:
                plt.plot(XX_fit_Draw,YY_fit_Draw,'-k', label="Fit",  linewidth=3)  # (:pointillé) -: ligne , back)
            if More_M > 0:
                if SMALL:
                    plt.plot(XX_fit2_Draw,YY_fit2_Draw,'-.g', label="Fit future")  # -. pointintillé/tiret green)
                else:
                    plt.plot(XX_fit2_Draw,YY_fit2_Draw,'-.r', label="Fit future", linewidth=3)
                # add un used missing raw data on the plot if required
                # actually displayed: plt.plot(X_MEAN_for_fit, Y_Mean_for_fit,'ob',  label='')
                    
            Result=True        
        else:
            Listbox_infos_Add(Tools.ERROR_FIT)
    return Result
    
    #FIN de Buit_matrx_

def Plot_Result_2a():
    Expo_Mode()
    USED_EXPO = Tools.Expo_Mode_==1
    onRadioButton_deg_Change()
    onRadioButton_FIX_LIN_LOG_EXPO()
    #plt.figure(figsize=(15,8))       
    
    Tools.fig77, Tools.ax77 = plt.subplots()
    Tools.fig77.subplots_adjust(right=0.78)
    
    Tools.fig77.set_size_inches(15,8)      
    
    Tools._STYLE(False)
    plt.xticks.direction = 'in'
    plt.yticks.direction ='in'
            
    plt.yticks.labelsize = 'large'
    plt.savefig.format = 'pdf'
        
    plt.xlabel('time', fontsize=15)
    plt.ylabel('data', fontsize=15)
    plt.minorticks_on()
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
        
    plt.tick_params(axis='both', which='both', direction='in')
    #print(Tools.Mean_Values__selected)
    if Tools.Mean_Values__selected:
        plt.plot(Tools.X_Split, Tools.Y_Split,'ob',  label='', color ='#FFFFFFFF')   
        Result= Buit_matrx_(Tools.More_M, Tools.Poly_deg,Tools.X_Split, 
                        Tools.Y_Split, USED_EXPO, True)
    else:   
        A = len(Tools.X_Split2)
        XMA=-1e30
        XMI=+1e30
        YMA=-1e30
        YMI=+1e30
        z= trunc(0.001 + A / Tools.NNVAL8)
        for i in range(0, Tools.NNVAL8):
            x_ = list([])
            y_= list([])
            x_.clear()
            y_.clear()
            for j in range(0, z):
                
                u       =   i + j*Tools.NNVAL8

                XMA     =   max ( XMA, Tools.X_Split2[u] )
                XMI     =   min ( XMI, Tools.X_Split2[u] )
                YMA     =   max ( YMA, Tools.Y_Split2[u] )
                YMI     =   min ( YMI, Tools.Y_Split2[u] )

                y_.append(Tools.Y_Split2[u])
                x_.append(Tools.X_Split2[u])

            Kolor=Tools.LambdaColor( float(i**2) / float(Tools.NNVAL8), -0.1,Tools.NNVAL8+0.1) 
           
            plt.plot(x_, y_,'o',  label=Tools.LAB_88[i], color = Kolor) 
            
        Result= Buit_matrx_(Tools.More_M, Tools.Poly_deg,Tools.X_Split2, 
                        Tools.Y_Split2, USED_EXPO, False)
    
        Tools.legend77 = Tools.ax77.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1))
        Tools.fig77.canvas.mpl_connect("scroll_event", Func_Mouse_Wheel_to_Scroll_Legend_Box)
        
        Tools.ax77.text(XMI - (XMA-XMI)/4, (YMA+YMI)/2, "use mouse wheel to scroll the legend", size=12, rotation=90.,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),   # RGB frame  0=>0 1=> 0xFF
                   fc=(0xa0/0xFF, 0xc6/0xFF, 0xef/0xFF),   # RGB background (light blue)
                   )
         )
            
    plt.show()
    
    Tools.Enable_disable(Tools.Button_Distibution, Result)
    
    if Result:
        Tools.Frame_SMTH.place(x=Tools.Frame_SMTH_X, y=Tools.Frame_SMTH_Y, width=Tools.Frame_SMTH_W,height=Tools.Frame_SMTH_H)
    else:
        Tools.HID(Tools.Frame_SMTH)
    #end Plot_Result_2a

def Extend(X, Y, h):
    N   =   len(X)
    XX  =   list([])
    YY  =   list([])
    dp  =   trunc(N/h)
    delta=  (X[N-1]-X[0])/h
    dt  =   delta / dp 
    #print(h, N, dp, delta, dt, X[N-1], X[0], Y[N-1], Y[0])
    # 4.5 1001 222 14.435713318920635 0.06502573567081367 25.092085032235985 -39.86862490290687 98.94625176803397 0.35360678925035377
    for nn in range(0, N +2*dp ):
        n=nn-dp
        if n <=0:
            XX.append(X[0] + n*dt)
            YY.append(0)
        elif (n>0) and (n < N):
            XX.append(X[n])
            YY.append(Y[n])
        elif n>=N:
            XX.append(X[N-1]+(n-N+1)*dt)
            YY.append(100)
    #print(XX[0], XX[N +2*dp-1])
    #-46.29730941530651 40.089515496025605
    return XX, YY

def Plot_Result_Distrib():
    Tools.PCT_01  =   Tools.SCALE_01.get()
    Tools.PCT_02  =   Tools.SCALE_02.get()
    Ratn    =   list ( [] )
    ATN     =   list ( [] )
    XTN     =   list ( [] )
    N       =   len(Tools.YY_for_Dif)
    NP      =   1000 # on génère 1000 sous niveaux lev Ymin..Y max pour définir la fonction n(y) <= lev
    Epsilon =   1E-20
    
    Tools.YY_for_Dif    =   sort(0,N-1, Tools.YY_for_Dif)  # organiser YY_for_Dif  ordre croissant
    _LOW    =   Tools.YY_for_Dif[0]-Epsilon
    _HIG    =   Tools.YY_for_Dif[N-1]+Epsilon
    k       =   0
    
    for n in range(0,NP+1):  #0..1000
        lev=_LOW+(_HIG-_LOW)*n/NP  # level . 
        while (lev > Tools.YY_for_Dif[k]) and (k < N-1): #after sort, Y is organized in crossing order
            k   +=  1  # number of occurence with Y < lev
        XTN.append(Tools.YY_for_Dif[k])
        Ratn.append(k*100/N)   # %  0%=> lev=Y min    100% lev=Ymax
        ATN.append(0) 
    
    AVERAGE =   50 # averaging  ± 5%  (50 with NP=1000)
    for n in range(0,NP+1):  #0..1000
        k   =   -AVERAGE
        U   =   0
        while True:
            U   +=  Ratn[max(0, min(NP,n+k))]
            k   +=  1
            if k==AVERAGE+1:
                ATN[n]  =   U/(1+2*AVERAGE)
                break
                 
    
    Tools._STYLE(True)
    plt.figure(figsize=(15,8))  
    
    ax              =   plt.subplot(1,1,1) # in an array Nbr_Lines X Nbr_Column   generate subplot N° index
    ax.set_ylabel('') # Y axis label
    ax.legend       =   '%'
    plt.ylim(0,100)# adjust limit max for a mor pleasent presentation
    plt.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0) # grid ( zorder=-1 < zorder(Bar) => behind bars)
    m=max(abs(XTN[0]), abs(XTN[len(XTN)-1]))
    plt.xlim(-m,m)
    #print('EEE')
    
    first_y_axis    =   plt.gca()            
    second_y_axis   =   plt.twinx() 
    
    st0='                    ξ'
    st1='П (ξ)   =    ∫       Ρ (ℓ). dℓ'
    st2='                   -∞'
    first_y_axis.set_ylabel(
        st0 + '\n' +
        st1 + '       ( % data ≤ model + ξ )\n' + 
        st2, 
        color="red", fontsize=14, 
        loc='bottom'   # in order to correctly allign st0, st1 and st2
        )
    Tools.TITR1('Distribution integration', 20)
    ax.set_xlabel('ξ  =  threshold', color="black", fontsize=16)
    L=1e-20
    first_y_axis.plot(XTN, ATN,color='#DD7722') 

    XTN, ATN = Extend(XTN, ATN, 7) 
    m=max(abs(XTN[0]), abs(XTN[len(XTN)-1]))
    plt.xlim(-m,m)
    
    if Compute_Splines(XTN, ATN, Tools.PCT_01 , Tools.Spline_X, Tools.Spline_Y, Tools.Spline_dY):
        first_y_axis.plot(Tools.Spline_X, Tools.Spline_Y, ':r') # spline sur Data
        Tools.Spline_dY   =   smooth(Tools.Spline_dY, Tools.PCT_02)  # calcul de spline compute ausse sa dérivée que l'on smooth
        if Compute_Splines(Tools.Spline_X, Tools.Spline_dY,Tools.PCT_01, Tools.Spline_X, Tools.Spline_Y, Tools.Spline_dY):  # pour "arrondir" la présentation de Spline_dY lissé  
           second_y_axis.plot(Tools.Spline_X, Tools.Spline_Y, color='Blue') # le spline de Spline_dY se retouve dans 
                                                                # Spline_Y. spline_dY contient maintenant 
                                                                # la dérivée seconde de la spline primaire 
                                                                # et n'est pas présentée ici
           for i in range(0, len(Tools.Spline_Y)):
               L = max (L, Tools.Spline_Y[i])
           second_y_axis.set_ylabel('probability density  [Ρ (ξ)]', color="blue", fontsize=14)
    second_y_axis.set_ylim(0,L*1.05)
    
    plt.show()
    #end Plot_Result_Distrib 

def Plot_Weight_Menu():
    Tools.PCT_03  =   Tools.SCALE_Pct_dans_filtre_older_def.get()
    plt.close('all')
    # Charger le fichier en memoire
                  
    Tools._STYLE(True)
    plt.figure(figsize=(15,8)) 
    plt.xticks.direction = 'in'
    plt.yticks.direction ='in'
            
    plt.yticks.labelsize = 'large'
    plt.savefig.format = 'pdf'
        
    plt.xlabel('old age as a percentage of the total range', fontsize=15)
    plt.ylabel('Weight [%]', fontsize=15)
    plt.minorticks_on()
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
        
    plt.tick_params(axis='both', which='both', direction='in')
    X = list([])
    Y=list([])
    for i in range(0,1001):
        X.append(i/10)
        Y.append(0)
    for model in range(0,4):
        for k in range(0,1001):
            Y[k] = 100*Weighting_POLY ( k/10, model, Tools.PCT_03)
        if model==0:
            st='Fix'
        elif model==1:
            st='Linear'
        elif model==2:
            st='Log'
        else:
            st='Exp'
        plt.plot(X,Y,label=st)
    plt.legend(loc="upper right")
    Tools.TITR1('Exemple with mi= ' + str(Tools.PCT_03) + '%', 20)
    plt.show()

def Update_value(value):
    XX = Tools.PCT_03
    Tools.PCT_03  =   Tools.SCALE_Pct_dans_filtre_older_def.get()
    if XX != Tools.PCT_03:
        Tools.Enable_disable(Tools.Button_Distibution, False)

def Plot_Result_3(Fenetre):
    Tools.Choix_deg_poly      = StringVar(Fenetre, "1")
    Tools.Choix_weighting_selec     = StringVar(Fenetre, "0")
    Tools.Choix_History    = StringVar(Fenetre, "0")           
    Tools.Plot_expo_selected      =   tk.IntVar()   # checked => 1    Unchecked => 0
    Tools._STYLE(True)
    Tools.Subwindow2 = tk.Toplevel(Fenetre)# Création d'une nouvelle fenêtre
    
    Tools.Subwindow2.protocol("WM_DELETE_WINDOW", on_window_close_via_X2) 
    if os.path.isfile(Tools.User_Defined_logo):
        Tools.Subwindow2.iconbitmap(Tools.User_Defined_logo) 
    Tools.Subwindow2.title("Choose a fit method")# Définition du titre
    #Tools.Subwindow2.geometry("850x650")# Dimensions de la fenêtre
    width_123   =   850
    height_123  =   650
    # récupération de la taille de l'écran
    '''
    if Tools.screen_width_123 <0: #initialisation -1  
        Tools.screen_width_123 = self.root.winfo_screenwidth()
    if Tools.screen_height_123<0:#initialisation -1
        Tools.screen_height_123 = self.root.winfo_screenheight()
        '''
    # calcul position x et y de la fenêtre
    x_coordinate_123 = Tools.screen_width_123 // 2 - width_123 // 2
    y_coordinate_123 = Tools.screen_height_123 // 2 - height_123 // 2
    Tools.Subwindow2.geometry(f"{width_123}x{height_123}+{x_coordinate_123}+{y_coordinate_123}")


    # Personnalisation de la fenêtre
    Tools.Subwindow2.configure(bg=Tools.My_Color1,relief="raised")# Changement de couleur de fond et du relief

    # Gestion des interactions avec la fenêtre
    Tools.Subwindow2.transient(Fenetre)# Place la fenêtre fille au-dessus de la fenêtre parent
    Tools.Subwindow2.grab_set()# Empêche l'utilisateur d'interagir avec la fenêtre parent
    Tools.Subwindow2.focus_set()# Donne le focus à la fenêtre fille
    
    Dico_11= { "fit order 0          "   :  "0",
               "fit order 1          "   :  "1",
               "fit order 2          "   :  "2",
               "fit order 3          "   :  "3",
               "fit order 4          "   :  "4",
               "fit order 5          "   :  "5",
               "fit order 6          "   :  "6",
               "fit order 7          "   :  "7",
               "fit order 8          "   :  "8",
               "fit order 9          "   :  "9",
               "fit order 10         "   :  "10",
               "fit order 11         "   :  "11",
               "fit order 12         "   :  "12",
               "fit order 13         "   :  "13",
               "fit order 14         "   :  "14",
               "fit order 15         "   :  "15"
             }
    a=20; b=20; ap = 65
    for (text, Dico_11_Entry) in Dico_11.items():
        Radiobutton(Tools.Subwindow2,   
                                text        =   text, 
                                variable    =   Tools.Choix_deg_poly, 
                                value       =   Dico_11_Entry,
                                font        =   ('courrier new',11),
                                command     =   onRadioButton_deg_Change,
                                bg          =   Tools.My_Color1
                                ).place(x=a, y=b)# boutons en colone
        if a==20:
            a   =   150 + ap
        elif a==150+ ap:
            a   =   280+ 2*ap
        elif a==280+ 2*ap:
            a   =   410+ 3*ap
        else:
            a   =   20
            b   +=  40
    onRadioButton_deg_Change()
            
    Tools.Plot_expo_selected.set(0)                       # default status :  checked => plot grid in plots
    Plot_expo                   =   tk.Checkbutton(Tools.Subwindow2,  
                                                   text='Exponential model', 
                                                   bg=Tools.My_Color1, fg='#000000',
                                                   variable=Tools.Plot_expo_selected, font=('courrier',12),
                                                   command=Expo_Mode) 
    
    Expo_Mode()
    Plot_expo.place(x=20,y=b+25)  

    MAX_YEARS_EXTEND    = 20
    liste_more_months_in_Model  = list([])
    
    for j in range(0, 12 * MAX_YEARS_EXTEND + 1):   # limit to MAX_YEARS_EXTEND additional years
        liste_more_months_in_Model.append('n =   ' + str(j))
    Tools.listeCombo_AditiveMonth = ttk.Combobox(
            Tools.Subwindow2, 
            values=liste_more_months_in_Model
            )
    
    lbl = Label(Tools.Subwindow2 , text = "Extend the model over 'n' month(s): select n", bg=Tools.My_Color1)
    # Placer le label sur la fenêtre
    lbl.place(x=20, y=b+95) 
    
    Tools.listeCombo_AditiveMonth.place(x=20+333, y=b + 95)     
    Tools.More_M=0
    Tools.listeCombo_AditiveMonth.current(Tools.More_M)
    
    Tools.listeCombo_AditiveMonth.bind("<<ComboboxSelected>>", action_More_M)
    action_More_M_No_Event()
    
    Button_Start_FIT =   Button(Tools.Subwindow2,  text= ' Fit data', font=('courrier',12), command=Plot_Result_2a)
   
    Tools.Button_Distibution =   Button(Tools.Subwindow2,  text= '📈 Distribution function',font=('courrier',12), command=Plot_Result_Distrib)
   
    
    lbl = Label(Tools.Subwindow2 , text = "Data to remove from fit", bg=Tools.My_Color1)    
    lbl.place(x=20, y=b+180)   # Placer le label sur la fenêtre
    
    lbl = Label(Tools.Subwindow2 , text = "n first month", bg=Tools.My_Color1)    
    lbl.place(x=220, y=b+180-20)   # Placer le label sur la fenêtre  
    
    lbl = Label(Tools.Subwindow2 , text = "n last month", bg=Tools.My_Color1)    
    lbl.place(x=220, y=b+180+20)   # Placer le label sur la fenêtre  
    
    Tools.listeCombo_Remove_1st_Month = ttk.Combobox(
            Tools.Subwindow2, 
            values=liste_more_months_in_Model
            )
    Tools.listeCombo_Remove_1st_Month.place(x=20+333, y=b+180-20)     
    Tools.Kill_1st=0
    Tools.listeCombo_Remove_1st_Month.current(Tools.Kill_1st)
    Tools.listeCombo_Remove_1st_Month.bind("<<ComboboxSelected>>", action_kill_1st)
    
    Tools.listeCombo_Remove_last_Month = ttk.Combobox(
            Tools.Subwindow2, 
            values=liste_more_months_in_Model
            )  
    action_kill_1st_No_Event()
    Tools.listeCombo_Remove_last_Month.place(x=20+333, y=b+180+20)     
    Tools.Kill_last=0
    Tools.listeCombo_Remove_last_Month.current(Tools.Kill_last) 
    Tools.listeCombo_Remove_last_Month.bind("<<ComboboxSelected>>", action_kill_last)
    LFT =   20
    a   =   850/2-LFT
    t0  =   b+250 
    action_kill_last_No_Event()
    Button_Start_FIT.place(x=LFT, y=t0,width=a)
    LFT +=  a+5
    Tools.Button_Distibution.place(x=LFT, y=t0,width=a)
    
    Tools.Frame_SMTH      =   tk.Frame(Tools.Subwindow2, borderwidth=7, pady=2, bg=Tools.My_Color2, relief='ridge')
    Tools.Frame_SMTH_X    =   LFT
    Tools.Frame_SMTH_Y    =   t0+30
    Tools.Frame_SMTH_W    =   a
    Tools.Frame_SMTH_H    =   180
    Tools.Frame_SMTH.place(x=Tools.Frame_SMTH_X, y=Tools.Frame_SMTH_Y, width=Tools.Frame_SMTH_W,height=Tools.Frame_SMTH_H)
    
    Frame_POIDS     =   tk.Frame(Tools.Subwindow2, borderwidth=7, pady=2, bg=Tools.My_Color2, relief='ridge')
    Frame_POIDS.place(x=20, y=Tools.Frame_SMTH_Y, width=Tools.Frame_SMTH_W,height=Tools.Frame_SMTH_H)
    
    Button_SHOW     =   Button(Frame_POIDS,  text = '📈 Ponderation Help ',font=('courrier',12), command=Plot_Weight_Menu)
    Button_SHOW.place(x=20, y=10, width=Tools.Frame_SMTH_W-40,height=35)
    '''
    '''
    
    Dico_20= { "Fix   "   :  '0',
               "Linear"   :  "1",
               "Log"      :  "2",
               "Exp"      :  "3"
             }
    ap = 85
    LFT=20
    for (text, Dico_20_Entry) in Dico_20.items():
        Radiobutton(Frame_POIDS,   text         =   text, 
                                  variable      =   Tools.Choix_History, 
                                value           =   Dico_20_Entry,
                                font            =   ('courrier new',11),
                                command         =   onRadioButton_FIX_LIN_LOG_EXPO,
                                bg              =   Tools.My_Color2
                                ).place(x=LFT, y=56)# boutons en colone
        LFT   += ap
    onRadioButton_FIX_LIN_LOG_EXPO()
    Frame_mi = tk.Frame(Frame_POIDS, borderwidth=2, pady=2, bg=Tools.My_Color1, relief='groove')
    Frame_mi_x  =   20 
    Frame_mi_y  =   56+28 +5
    Frame_mi_w  =   LFT 
    Frame_mi_h  =   60
    Frame_mi.place(x=Frame_mi_x, y=Frame_mi_y,width=Frame_mi_w,height=Frame_mi_h)
    lbl = Label(Frame_mi , text = "Minimum value [%]", bg=Tools.My_Color1, font=('courrier',12))    
    lbl.place(x=20, y=5)   # Placer le label sur la fenêtre

    Tools.SCALE_Pct_dans_filtre_older_def = Scale(Frame_mi, from_=0, to=100, orient=HORIZONTAL, bg=Tools.My_Color1, command = Update_value)
    Tools.SCALE_Pct_dans_filtre_older_def.set(Tools.PCT_03)
    Tools.SCALE_Pct_dans_filtre_older_def.place(x=190,  y=1,width=160)
    
    lbl         = Label(Tools.Frame_SMTH , text = "smooth % spline", bg=Tools.My_Color2)
    LFT         = 2
    dh          = 12
    t0          = -40
    # Placer le label sur la fenêtre
    lbl.place(x=LFT, y=t0+60+dh)
    lbl      = Label(Tools.Frame_SMTH , text = "smooth % distribution curve", bg=Tools.My_Color2)
    # Placer le label sur la fenêtre
    lbl.place(x=LFT, y=t0+110+dh)

    Tools.SCALE_01 = Scale(Tools.Frame_SMTH, from_=1, to=25, orient=HORIZONTAL, bg=Tools.My_Color2)
    Tools.SCALE_01.set(Tools.PCT_01)
    LFT += 200
    a   =   a-LFT-20
    Tools.SCALE_01.place(x=LFT,  y=t0+60,width=a)
    
    Tools.SCALE_02 = Scale(Tools.Frame_SMTH, from_=1, to=25, orient=HORIZONTAL, bg=Tools.My_Color2)
    Tools.SCALE_02.set(Tools.PCT_02)
    Tools.SCALE_02.place(x=LFT, y=t0+110,width=a)    
    Tools.Enable_disable(Tools.Button_Distibution, False) 
    Tools.HID(Tools.Frame_SMTH)
        
    #end Plot_Result_3

def Choose_Plot_Cnd():
    if Check_NN()>0:
        Plot_Result_3(Tools.Subwindow___)
        return
    return

TEST_VV = True

def Func_Mouse_Wheel_to_Scroll_Legend_Box(evt):
    d77     =   {"down" : 30, "up" : -30}
    if Tools.legend77     !=  None:
        if Tools.legend77.contains(evt):
            bbox    =   Tools.legend77.get_bbox_to_anchor()
            bbox    =   Bbox.from_bounds(bbox.x0, bbox.y0+d77[evt.button], bbox.width, bbox.height)
            tr      =   Tools.legend77.axes.transAxes.inverted()
            Tools.legend77.set_bbox_to_anchor(bbox.transformed(tr))
            Tools.fig77.canvas.draw_idle() # redraw fig77.canvas
    #fin de Func_Mouse_Wheel_to_Scroll_Legend_Box

def Plot_actual():
    NN= Check_NN()
    if NN > 0:
        #   PRESETAION RAW DATA SPLITTED IN A COUPLE OF SUB-RANGE
                 
        Tools._STYLE(True)
        Tools.fig77, Tools.ax77 = plt.subplots()
        Tools.fig77.subplots_adjust(right=0.78)
    
        Tools.fig77.set_size_inches(15,8)
        #plt.figure(figsize=(15,8))
        plt.xticks.direction = 'in'
        plt.yticks.direction ='in'
            
        plt.yticks.labelsize = 'large'
        plt.savefig.format = 'pdf'
        
        titre="Selected Raw data presentation"
        
        Tools.TITR1(titre, 20)
        plt.xlabel('time', fontsize=15)
        plt.ylabel('data', fontsize=15)
        plt.minorticks_on()
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
        n77=0
        plt.tick_params(axis='both', which='both', direction='in')
        YMA=-1e30
        YMI = 1e30

        for i in range(0, len(Tools.Sektor)):
            for j in range(0, len(Tools.Sect_Ticket[i])):
                if  Plot_Trace_selected[i][j].get()==1:                    
                    Kolor=Tools.LambdaColor( n77**2 / NN, 0, NN)  
                    n77 +=1
                    LAB = Tools.Sektor[i] + '/' + Tools.Sect_Ticket[i][j]
                    VV = list([])
                    VV.clear()
                    for n in range(0, Tools.N_DDT_ALL_LEN):
                        G = YVL(i, j, n, Tools.Closed_or_close_star_volume_)
                        VV.append(G)
                        YMA = max(YMA, G)
                        YMI = min(YMI, G)
        
                    Tools.ax77.plot(Tools.DDT_ALL, VV, label=LAB, color=Kolor)
                  
        Tools.ax77.plot(Tools.X_Raw, Tools.Y_Raw, label='Mean curve for analysis', color='#000000', linewidth=3)
        if Tools.Average_lev >0:
            Tools._STYLE(False)
            plt.plot(Tools.X_Split, Tools.Y_Split,'ob',  label='Points for model', color ='#FFFFFFFF')

        XMA = Tools.DDT_ALL[Tools.N_DDT_ALL_LEN-1]
        XMI = Tools.DDT_ALL[0]
        a=XMA + (XMA-XMI)/4
        b=(YMA + YMI)/2
        m=Tools.DDT_M[0]
        Catego = list([])
        PPO    = list([])
        for i in range(0,Tools.N_DDT_ALL_LEN):
            if m!=Tools.DDT_M[i]:
                m=Tools.DDT_M[i]
                Catego.append(str(Tools.DDT_Y[i] % 100) + '/' + str(m))
                PPO.append(Tools.DDT_ALL[i])

        plt.xticks(PPO, Catego, rotation ='vertical') # , fontsize=9

        Tools.legend77 = Tools.ax77.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1))
        Tools.fig77.canvas.mpl_connect("scroll_event", Func_Mouse_Wheel_to_Scroll_Legend_Box)
        Tools.ax77.text(XMI - (XMA-XMI)/4, b, "use mouse wheel to scroll the legend", size=12, rotation=90.,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),   # RGB frame  0=>0 1=> 0xFF
                   fc=(0xa0/0xFF, 0xc6/0xFF, 0xef/0xFF),   # RGB background (light blue)
                   )
         )
        plt.show()

def sel_0():
    for i in range(0, len(Tools.Sektor)):
        for j in range(0, len(Tools.Sect_Ticket[i])):
            Plot_Trace_selected[i][j].set(0)

def sel_all():
    for i in range(0, len(Tools.Sektor)):
        for j in range(0, len(Tools.Sect_Ticket[i])):
            Plot_Trace_selected[i][j].set(1)

def sel_revl():
    for i in range(0, len(Tools.Sektor)):
        for j in range(0, len(Tools.Sect_Ticket[i])):
            V=Plot_Trace_selected[i][j].get()
            Plot_Trace_selected[i][j].set(1-V)

def Distrib_Start(Fenetre):
    Tools.Plot_Closed_or_close_star_volume_selected     =   tk.IntVar()   # checked => 1    Unchecked => 0
    Tools.Plot_Mean_selected            =   tk.IntVar()   # checked => 1    Unchecked => 0
    General_ML.Preliminary_info.clear()
    Tools.Choix_Average         =   StringVar(None, "0")
    Tools.selection               =   tk.IntVar()
    Tools.Subwindow___ = tk.Toplevel(Fenetre)# Création d'une nouvelle fenêtre
    if os.path.isfile(Tools.User_Defined_logo):
        Tools.Subwindow___.iconbitmap(Tools.User_Defined_logo) 
    Tools.Subwindow___.title("Distribution analysis access")# Définition du titre
    #Tools.Subwindow___.geometry("1000x800")# Dimensions de la fenêtre

    
    width_123   =   1000
    height_123  =   800
    # récupération de la taille de l'écran
    if Tools.screen_width_123 <0: #initialisation -1  
        Tools.screen_width_123 = self.root.winfo_screenwidth()
    if Tools.screen_height_123<0:#initialisation -1
        Tools.screen_height_123 = self.root.winfo_screenheight()
    # calcul position x et y de la fenêtre
    x_coordinate_123 = Tools.screen_width_123 // 2 - width_123 // 2
    y_coordinate_123 = Tools.screen_height_123 // 2 - height_123 // 2
    Tools.Subwindow___.geometry(f"{width_123}x{height_123}+{x_coordinate_123}+{y_coordinate_123}")


    Tools.Subwindow___.protocol("WM_DELETE_WINDOW", on_window_close_via_X) 

    # Personnalisation de la fenêtre
    Tools.Subwindow___.configure(bg=Tools.My_Color2,relief="raised")# Changement de couleur de fond et du relief

    # Gestion des interactions avec la fenêtre
    Tools.Subwindow___.transient(Fenetre)# Place la fenêtre fille au-dessus de la fenêtre parent
    Tools.Subwindow___.grab_set()# Empêche l'utilisateur d'interagir avec la fenêtre parent
    
    Tools.Subwindow___.focus_set()# Donne le focus à la fenêtre fille 

    lbl = Label(Tools.Subwindow___ , text = "Select the data to introduce in the analysis",font=('Arial', 16), bg=Tools.My_Color2)
    # Placer le label sur la fenêtre
    lbl.pack(pady=15) # place(x=20, y=15) 
    
    lbl = Label(Tools.Subwindow___ , text = "(In case of multisection, average value will be analysis)", fg='#434343', 
                font=('Arial', 10), bg=Tools.My_Color2)
    # Placer le label sur la fenêtre
    lbl.pack(pady=5) # place(x=20, y=15) 

    for i in range(0, len(Tools.Sektor)):
        Seelc.append(list([]))
        Plot_Trace_selected.append(list([]))
        lbl = Label(Tools.Subwindow___ , text = Tools.Sektor[i], fg='Blue', 
                font=('Arial', 10), bg=Tools.My_Color2)
        # Placer le label sur la fenêtre
        ax = 110-40+170*i
        lbl.place(x=ax, y=130) #150*4=600+110=710
        for j in range(0, len(Tools.Sect_Ticket[i])):
            lbl = Label(Tools.Subwindow___ , text = Tools.Sect_Ticket[i][j], fg=Tools.My_Color0, 
                    font=('Arial', 10), bg=Tools.My_Color2)
            Y = 160 + 40*j
            lbl.place(x=ax-40, y=Y) #150*4=600+110=710
            Plot_Trace_selected[i].append(0)
            Seelc[i].append(0)
            Plot_Trace_selected[i][j] = tk.IntVar()
            Plot_Trace_selected[i][j].set(1) # checked default
            Seelc[i][j] =   tk.Checkbutton(Tools.Subwindow___,  text='', 
                                    fg='#000000', bg=Tools.My_Color2,
                                    variable=Plot_Trace_selected[i][j], font=('courrier',12)) 
            Seelc[i][j].place(x=ax+20, y=Y)
        
        Dico_50= { "Merge per day    "   :  '0',
                   "Merge per week   "   :  '1',
                   "Merge per month  "   :  '2',
             }
        LFT=200       
        Y+=120
        for (text, Dico_50_Entry) in Dico_50.items():
            Radiobutton(Tools.Subwindow___,   text        =   text, 
                                              variable    =   Tools.Choix_Average, 
                                              value       =   Dico_50_Entry,
                                              font        =   ('courrier new',11),
                                              command     =   onRadioButton_Day_Week_Month,
                                              bg          =   Tools.My_Color2
                                              ).place(x=LFT, y=Y)# boutons en colone
            LFT+=200
    Tools.Choix_Average.set(1)
    onRadioButton_Day_Week_Month()

    Button_Start_Calcul0    =   Button(Tools.Subwindow___,      
                                       text= '📊 Start estimation',
                                       font=('courrier',12),
                                       command=Choose_Plot_Cnd) #Plot_Result)

    Button_QUIT             =   Button(Tools.Subwindow___,                                             
                                       text= '🛑 Return',
                                       font=('courrier',12),
                                       command=End_Analyse)
    Button_plot_            =   Button(Tools.Subwindow___,                                             
                                       text= '📈 Plot actual selection',
                                       font=('courrier',12),
                                       command=Plot_actual)
    Button_Start_Calcul0.place(x=1000-250-50, w=250, y=720)
    Button_QUIT.place(x=20, y=720, w=250)
    Button_plot_.place(x=500-125, y=720, w=250)

    Button_All    =   Button(Tools.Subwindow___,      
                            text= 'Select all',
                            font=('courrier',12),
                            command=sel_all) #Plot_Result)

    Button_Rev    =   Button(Tools.Subwindow___,      
                            text= 'Reverse selection',
                            font=('courrier',12),
                            command=sel_revl) #Plot_Result)
    Button_NO    =   Button(Tools.Subwindow___,      
                            text= 'Unselect all',
                            font=('courrier',12),
                            command=sel_0) #Plot_Result)

    Button_NO.place(x=20, y=500)
    Button_Rev.place(x=300, y=500)
    Button_All.place(x=600, y=500)
    
    
            
    Tools.Plot_Closed_or_close_star_volume_selected.set(1)                       # default status :  checked => plot grid in plots
    Plot_expo_                   =   tk.Checkbutton(Tools.Subwindow___,  
                                                   text='Tag => Close data,  untag => closed data * Volume', 
                                                   bg=Tools.My_Color2, fg='#000000',
                                                   variable=Tools.Plot_Closed_or_close_star_volume_selected, font=('courrier',12),
                                                   command=Closed_or_close_star_volume) 
    
    Closed_or_close_star_volume()
    Plot_expo_.place(x=20,y=720-48)  
    
    
            
    Tools.Plot_Mean_selected.set(1)                       # default status :  checked => plot grid in plots
    Plot_Mean_                   =   tk.Checkbutton(Tools.Subwindow___,  
                                                   text='Use mean data at each selected date', 
                                                   bg=Tools.My_Color2, fg='#000000',
                                                   variable=Tools.Plot_Mean_selected, font=('courrier',12),
                                                   command=Mean_Values_Chk) 
    
    Plot_Mean_.place(x=520,y=720-48)  
    Mean_Values_Chk()

    Fenetre.update()
    
