# Tools.py (New - Simple dependency checker)
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  General_ML
from    datetime                import  datetime
import  dateutil.parser
from    tkinter                 import  messagebox
import  matplotlib.pyplot       as      plt
import  os
from    sklearn.ensemble        import  RandomForestRegressor, GradientBoostingRegressor
from    sklearn.linear_model    import  LinearRegression
from    sklearn.svm             import  SVR
from    sklearn.model_selection import  train_test_split 
from    sklearn.model_selection import  GridSearchCV
from    sklearn.metrics         import  mean_absolute_error,  mean_squared_error, r2_score

DDT_ALL                                 =   list([])
DDT_ALL_TXT                             =   list([])  
DDT_Y                                   =   list([])
DDT_M                                   =   list([])
DDT_D                                   =   list([])
X_Split                                 =   list([])
Y_Split                                 =   list([])
X_Split2                                =   list([])
Y_Split2                                =   list([])
Button_Distibution                      =   None
legend77                                =   None 
fig77                                   =   None 
ax77                                    =   None
SCALE_Pct_dans_filtre_older_def         =   None
SCALE_02                                =   None
SCALE_01                                =   None
app                                     =   None
Frame_SMTH                              =   None
Choix_deg_poly                          =   None
Choix_weighting_selec                   =   None
Choix_history                           =   None
Plot_expo_selected                      =   None
Plot_Closed_or_close_star_volume_selected=  None
Plot_Mean_selected                      =   None
listeCombo_AditiveMonth                 =   None
listeCombo_Remove_1st_Month             =   None
listeCombo_Remove_last_Month            =   None
Choix_weighting = selection             =   None 
Subwindow___                            =   None 
Subwindow2                              =   None
PLT_123                                 =   None

LAB_88                                  =   list([])
Closed_or_close_star_volume_            =   True
Mean_Values__selected                   =   True

analysis_cache_path                     =   "analysis_cache"

screen_width_123                        =   -1
screen_height_123                       =   -1

Kill_last                               =   0
FIX_LIN_LOG_EXPO                        =   0
Kill_1st                                =   0
PCT_03                                  =   35
PCT_01                                  =   2
PCT_02                                  =   7
Poly_deg                                =   1
More_M                                  =   0
Expo_Mode_                              =   0
Frame_SMTH_X                            =   0
Frame_SMTH_Y                            =   0
Frame_SMTH_W                            =   0
Frame_SMTH_H                            =   0
NNVAL8                                  =   0
ERROR_FIT                               =   ''
coef                                    =   list([])
YY_for_Dif                              =   list([])
X_Raw                                   =   list([])
Y_Raw                                   =   list([])
Spline_X                                =   list([])
Spline_Y                                =   list([])
Spline_dY                               =   list([])
Value_at_Close                          =   list([list([ list([ ])])]) # data[sector][ticket][0,1,2,...]
Volume_at_dt                            =   list([list([ list([ ])])]) # data[sector][ticket][0,1,2,...]

Sektor                                  =   list(['Technology', 'AI_Cloud', 'Finance', 'Healthcare', 'Energy', 'Consumer' ])
Sect_Ticket                             =   list([
                                                    ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'INTC', 'CSCO'], 
                                                    ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD' , 'SNOW'],
                                                    ['JPM' , 'BAC' , 'GS'   , 'MS'  , 'V'   , 'MA'  , 'PYPL', 'COF' ],
                                                    ['JNJ' , 'PFE' , 'MRK'  , 'ABT' , 'UNH' , 'LLY' , 'GILD', 'AMGN'],
                                                    ['XOM' , 'CVX' , 'COP'  , 'SLB' , 'EOG' , 'PSX' , 'VLO' , 'MPC' ],
                                                    ['AMZN', 'WMT' , 'HD'   , 'MCD' , 'NKE' , 'SBUX', 'TGT' , 'LOW' ]
                                                ]
                                                )

for i in range(0, len(Sektor)):  
    Value_at_Close.append(list([]))   # Value_at_Close[sector]
    Volume_at_dt.append(list([]))
    for j in range(0, len(Sect_Ticket[i])):
        Value_at_Close[i].append(list([])) # Value_at_Close[sector][ticket]
        Volume_at_dt[i].append(list([]))
        


def TITR1( st, sz):
    font1 = {'family':'serif','color':'blue','size':sz, 'style':'italic'}
    plt.title(st, pad=25, fontdict = font1)
    #end TITR1

def JPM123(CF, X, y, app): # here X is already scaled

    x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=5)
    #print(len(X), len(y))
    plt.figure(figsize=(15,8))
    TITR1('Data splitting  &  Models results.   Red dashed lines: Ideal situation where Ỹ(model, x_test) = Y(data, x_test) ∀ x_test', 16)
    plt.axis('off')
    xxTrain = list([])
    yyTrain =list([])
    xxTest = list([])
    yyTest =list([])
    #X_ALL  = list([])
    N_TRAIN=0
    N_TEST=0
    for k in range(0,5):
        #X_ALL.append(X[:,k])
        xxTrain.append(x_train[:,k])
        yyTrain.append(y_train)
        xxTest.append(x_test[:,k])
        yyTest.append(y_test)
        N_TRAIN += len(x_train[:,k])
        N_TEST += len(x_test[:,k])
        
    ax0  =   plt.subplot(3,2,1) # in an array Nbr_Lines X Nbr_Column   generate subplot N° index
        
    ax0.set_ylabel('Y train data', color="blue", fontsize=12) # Y axis label
    font2 = {'family':'serif','color':'#5c5c5c','size':12, 'style':'italic'}
    ax0.set_xlabel('X train data', labelpad=-35, fontdict = font2)
    font2 = {'family':'serif','color':'blue','size':12, 'style':'italic'}
   
    plt.title('Train subset  [' + str(N_TRAIN) + ' data]', pad=3, fontdict = font2)
    ax0.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
    
    ax0.scatter(xxTrain, yyTrain, alpha=0.8, color='#6666FF', label='Train')   #08 => alphas
           
    ax1  =   plt.subplot(3,2,2) # in an array Nbr_Lines X Nbr_Column   generate subplot N° index
    ax1.grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
    
    first_y_axis    =   plt.gca()            
    second_y_axis   =   plt.twinx() 

    second_y_axis.set_ylabel('Y test data', color="red", fontsize=14)

    font2 = {'family':'serif','color':'#5c5c5c','size':12, 'style':'italic'}
    #color="gray", fontsize=12
    ax1.set_xlabel('X test data', labelpad=-35, fontdict = font2)
    font2 = {'family':'serif','color':'red','size':12, 'style':'italic'}
   
    plt.title('Test subset  [' + str(N_TEST) + ' data]', pad=3, fontdict = font2)
    ax1.set_yticklabels([])    
    
    second_y_axis.scatter(xxTest, yyTest, alpha=0.8, color='#FF6666', label='Test')
    
    ax = list([])
    ax.clear()
    average_r2_best= -1e30
    for modl in range(0,4):
        gbr                 =   None
        VO_2                =   False   
        V3_4                =   False
        if modl==0:
            name            =   'random_forest'
            model           =   RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            Model_Name      =   'Random Forest - Ensemble of decision \n trees, robust to outliers'
            ''' take a lot of time without significative improvements with default R2 is already 0.99'''
            param_grid      =   {
                                'n_estimators' : [100, 200, 300],
                                'max_depth'     :   [10, 20, 30],
                                'min_samples_split': [2,5, 10],
                                'min_samples_leaf' : [1,2,4]
                                }
            gbr             =   RandomForestRegressor()
            gbr_cv          =   GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
           
        elif modl==1:
            name            =   'gradient_boosting'
            gbr             =   GradientBoostingRegressor()
            model           =   GradientBoostingRegressor(learning_rate=0.001,n_estimators=100, random_state=42, max_depth=3)
            Model_Name      =   'Gradient Boosting - Sequentially \n improves predictions'
            param_grid      =   {
                                'n_estimators' : [100, 200, 300],
                                'learning_rate' : [0.01, 0.1, 0.2],
                                'max_depth'     :   [3,4,5],
                                'min_samples_split': [2,3,4],
                                'min_samples_leaf' : [1,2,3],
                                'criterion' : ['friedman_mse']
                                }
            gbr_cv          =   GridSearchCV(gbr, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            VO_2            =   True 
        elif modl==2:
            name            =   'linear'
            model           =   LinearRegression()
            Model_Name      =   'Linear Regression - Simple \n linear relationships'
            V3_4            =   True
            '''
            Idem:
            Here we could also add a param_grid in order to scan some parameters 
            and fix the model on the best set of parameter.

            '''
        else: 
            name            =   'svr'
            model           =   SVR(kernel='rbf', C=1.0)
            Model_Name      =   'Support Vector Regression - \n Effective for complex patterns'
            VO_2            =   True  
            V3_4            =   True
            param_grid      =   {
                                'C' : [0.1,1,10,100],
                                'epsilon' : [0.01, 0.1, 1],
                                'kernel'     :   ['linear', 'poly', 'rbf'],
                                'gamma': ['scale','auto'],
                                }
            gbr             =   SVR()
            gbr_cv          =   GridSearchCV(gbr, param_grid, cv=5)
         
        ax.append(None)
        ax[modl]  =   plt.subplot(3,2,modl+3) # in an array Nbr_Lines X Nbr_Column   generate subplot N° index
        ax[modl].grid(True, axis='both', color='gray', linestyle='dashed', zorder=-1.0)
        AXX       = ax[modl]
        if VO_2: 
            ax[modl].set_yticklabels([])
            ax[modl]  = ax[modl].twinx()

        ax[modl].set_ylabel('Ỹ(X-test) \n[computed with model]', color="gray", fontsize=12) # Y axis label
        if V3_4:        
            AXX.set_xlabel('Y(X-test) [observed data]', color="gray", fontsize=12)

        font2 = {'family':'serif','color':'#111111','size':10, 'style':'italic'}
        ax[modl].set_title(Model_Name, fontdict = font2, y=1.0, pad=-25)           
            
        if gbr==None:
            model.fit(x_train,y_train)
            y_pred      =   model.predict(x_test)
            hh99        =   model
        else:
            gbr_cv.fit(x_train,y_train)
            y_pred=gbr_cv.predict(x_test)
            Best_P=gbr_cv.best_params_

            if modl==3:
                hh99    =   SVR()
            elif modl==1:
                hh99    =   GradientBoostingRegressor()
            elif modl ==0:
                hh99    =   RandomForestRegressor()

            #for k,v in Best_P.items():
            #  setattr(hh99,k,v)
            hh99.set_params(**Best_P) # report only non default values

            hh99.fit(x_train,y_train) # sans ce réappel à .fit, 
                                      # self.models[self.best_model].predict
                                      # dans
                                      # ML_Prediction.py peut planter  
            y_pred      =   hh99.predict(x_test) # normalement not utile mai
                
        ax[modl].scatter(y_test, y_pred, alpha=0.8, color='#000000', label=Model_Name)
        MI=1e+30
        MA=-1e30
        for k in range(0, len(y_pred)):
            if MI > min(y_test[k], y_pred[k]):
                MI = min(y_test[k], y_pred[k])
            if MA < max(y_test[k], y_pred[k]):
                MA = max(y_test[k], y_pred[k])
        xx=list([])
        xx.clear()
        xx.append(MI)
        xx.append(MA)
        ax[modl].plot(xx,xx, '-.r', label='ideal line') 
        MAE = mean_absolute_error(y_test, y_pred)   
        MSE = mean_squared_error(y_test, y_pred)   
        R2S = r2_score(y_test, y_pred)           
        
        if len(y_test) > 0:
            residuals = y_test - y_pred
            standard_error=0
            for i in range(0, len(y_test)):
                standard_error+=residuals[i]**2
            standard_error/=len(y_test)
            standard_error=standard_error **0.5
        else:
            standard_error=-1
        st=''
        for px in range(0, len(Model_Name)):
            if Model_Name[px] != '\n':
                st = st + Model_Name[px] 
        Display_info(st , app)    
        i=len(st)
        st= "=" * i
      
        Display_info(st , app) 
        Display_info('          mean_absolute_error     : ' + str(MAE) , app)    
        Display_info('          mean_squared_error      : ' + str(MSE) , app) 
        Display_info('          r2_score                : ' + str(R2S) , app) 
        app.ml_predictor.training_scores[name]      =   R2S
        app.ml_predictor.models[name]               =   hh99
        if standard_error >= 0:
            Display_info('          standard error σ        : ' + str(standard_error) , app) 
        else:
            Display_info('          standard error σ        : ' + 'Not available' , app) 
        
        if gbr!=None:
            Display_info('          Best model parameters   : ', app) 
            Display_info(Best_P , app) 
            
        if hasattr(hh99, 'feature_importances_'):   
           app.ml_predictor.feature_importance[name] = dict(zip(CF.columns, hh99.feature_importances_))

        if (R2S > average_r2_best): 
            average_r2_best = R2S
            Model_Name_best = ''
            for px in range(0, len(Model_Name)):
                if Model_Name[px] != '\n':
                   Model_Name_best = Model_Name_best + Model_Name[px] 
            app.ml_predictor.best_score = average_r2_best
            app.ml_predictor.best_model = name
                     
    Display_info('\n\n Best situation:\n                   "' + Model_Name_best + '"\n', app)
    if average_r2_best < 0:
        Display_info("⬛⬜⬜⬜⬜          WARNING:  Negative R² indicates poor model performance", app)
    elif average_r2_best < 0.3:
        Display_info("⬛⬛⬜⬜⬜          INFO:     Weak predictive power - consider collecting more data", app)
    elif average_r2_best < 0.6:
        Display_info("⬛⬛⬛⬜⬜          INFO:     Moderate predictive power", app)
    elif average_r2_best < 0.8:
        Display_info("⬛⬛⬛⬛⬜          INFO:     Good predictive power", app)
    else:
        Display_info("⬛⬛⬛⬛⬛          INFO:     Excellent predictive power!", app) 
    Display_info('Remark: in case of poor/insufficient results, before a definitive conclusion, we still have to optimize the models parameters', app) 
    Display_info('        using a wider range or a more accurate pesolution or including other parameters in the current models settings parameters', app)
    
    #app.progress.stop()
    #plt.show()
    return True, plt

def TITR2( st, sz):
    font1 = {'family':'serif','color':'blue','size':sz, 'style':'italic'}
    plt.title(st, pad=25, fontdict = font1)
    #end TITR2  ax

def HID(x):  # Hide x (x under control place() . other: grid_forget, pack_forget)
    if x != None: 
        try:
           x.place_forget() # hidden  (il y a une exeption si on cache un objet alors que sa fenetre mère est en cours de destuction)
        except:             # HID n'a pas de sens dans ce cas mais peut être appellé inopportumément par une suite d'events qui se succèdent
            pass
    #end HID 

def aju77(rgb):
    X=hex(rgb)
    X = X[2:] # remove 2 first char (0x)
    while len(X) < 2:
        X='0' + X
    return X
    
def RGB_(r,g,b):
    A='#' + aju77(r) + aju77(g) + aju77(b)
    return A
    #end RGB_

def LambdaColor( T, Tmin, Tmax ) :  
   if ( T < Tmin ):
      return  '#000000'
   if ( T > Tmax):
      return '#FFFFFF'
   if ( Tmax <= Tmin ) :
      return '#555555'
   Dbl = 1 / ( Tmax-Tmin)
   Dbl = Dbl * (T-Tmin)
   L = round((0x400-1) * Dbl)
 
   if ( L >= 0x400) :
       L = 0x400-1
   if ( L < 0x100): #0..0x100-1
      r=0xFF
      b=0
      g=L
      return RGB_(r,g,b)   
   if ( L < 0x200 ) : # 0x100..0x200-1      
      L -= 0x100  #0..FF
      g=0xFF
      b=0
      r= 0xFF - L
      return RGB_(r,g,b)   
   if (L < 0x300): #0x200..0x300-1
      L -= 0x200
      r=0
      g=0xFF
      b = L
      return RGB_(r,g,b)
   #0x300..0x400-1
   L-= 0x300
   r=0
   b=0xFF
   g= 0xff - L
   return RGB_(r,g,b)
   #end LambdaColor

def Hid_Plt_Attribute(hid):
    #daw the title and usinf font1 aspct. Upon hid parameter, remove or not the axis
    TITR2("Compare current market conditions with historical bubbles ", 20)
    if hid: 
        plt.axis('off')  # else default = 'on'
    #fin Hid_Plt_Attribute

def Enable_disable(X, B):
    if X != None:
        try:
            if B:
                X.config(state='normal')     # X ['state']  =   NORMAL
            else:
                X.config(state='disabled')   # X ['state']  =   DISABLED  
        except Exception as e:  
            pass  # st=f"❌ Enable_disable(X,B) : {e}" ; print(st); print(B, X)   
     #end Enable_disable 

def get_YMD(st):
    res = dateutil.parser.parse(st, fuzzy=True)
    D=res.day
    M=res.month
    Y=res.year
    return Y, M, D
          
Tools_QQUUIITT          =   True
Mon_Modele_STYLE1       =   'style1.mplstyle' 
Mon_Modele_STYLE2       =   'style2.mplstyle' 
User_Defined_logo       =   "mylogo.txt"
User_Defined_logo2      =   "MAC_MyLogo.icns"

CSV_output_fl           =   'SAVED_DATA.csv'
tempory_csv             =   'tempory_csv_file.csv'
INFO_001                =   ''

My_Color0               =   '#1936c6'
My_Color1               =   '#d8d8d8'
My_Color2               =   '#ddddff' 
My_Color3               =   '#9999cc'   
Average_lev             =   1        # 0 => par jour   1 => par semaine   2 => par moi
N_DDT_ALL_LEN           =   -1

def _STYLE( One):
    if One:  
        plt.style.use(Mon_Modele_STYLE1)
        plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']
    else:
        plt.style.use(Mon_Modele_STYLE2)
    #end STYLE

def More88(st, sep):
    return   sep * 4  + st +   sep*5

def Msg(a,b,c):
    messagebox.showinfo(title  =  a,  message  =  b,  icon  =   c )
    #end Msg

def add_missing_data(V, D, i, j):
    if len(V[i][j]) == len(D):
        return True
    if len(V[i][j]) < len(D):
        while len(V[i][j]) < len(D):
            L = V[i][j][len(V[i][j])-1]
            V[i][j].append(L)
        return True
    #ici len(V[i][j]) > len(D)
    while len(V[i][j]) > len(D):
        V[i][j].pop()
    return True
    return False

def PUSH_in_CSV(app):
    st      =   tempory_csv
    Enable_disable(app.Distrib_btn, False)
    sep=';'
    with open(st,'w') as file1: 
        file1.write('CLOSED VALUE\n')
        file1.write('=============\n')
        st='Date' + sep+sep
        for i in range(0, len(Sektor)):
            st = st + More88(Sektor[i], sep)
        file1.write(st+sep+'\n')  

        st=sep*3
        for i in range(0, len(Sektor)):
            AA=''
            for j in range(len(Sect_Ticket[i])):
                AA = AA + Sect_Ticket[i][j] + sep               
            st= st +  AA + sep
        file1.write(st+'\n')

        for ISEC in range(0, len(Sektor)):
            for JSEC in range(len(Sect_Ticket[ISEC])):
                
                if not add_missing_data(Value_at_Close, DDT_ALL, ISEC, JSEC):
                    Display_info('Inconsistent data [1]... ' + tempory_csv + ' not gererated',app)
                    file1.close()
                    General_ML.Klean(tempory_csv)
                    return    
                
                if not add_missing_data(Volume_at_dt, DDT_ALL, ISEC, JSEC):
                    Display_info('Inconsistent data [2]... ' + tempory_csv + ' not gererated',app)
                    file1.close()
                    General_ML.Klean(tempory_csv)
                    return

        for n in range(0, len(DDT_ALL)):
            st=DDT_ALL_TXT[n] + sep + str(DDT_ALL[n]) + sep*2 
            for ISEC in range(0, len(Sektor)):
                AA=''
                for JSEC in range(len(Sect_Ticket[ISEC])):
                    AA = AA + str(Value_at_Close[ISEC][JSEC][n]) + sep               
                st= st +  AA + sep
            #Tools.Value_at_Close[ISEC][JSEC].append(float(line))

            file1.write(st+'\n')  
            #end for n=0..N_lignedata
        #end with open
        
        file1.write('\n\n\nVOLUME\n')
        file1.write(      '=======\n')
        st='Date' + sep+sep
        for i in range(0, len(Sektor)):
            st = st + More88(Sektor[i], sep)
        file1.write(st+sep+'\n')  

        st=sep*3
        for i in range(0, len(Sektor)):
            AA=''
            for j in range(len(Sect_Ticket[i])):
                AA = AA + Sect_Ticket[i][j] + sep               
            st= st +  AA + sep
        file1.write(st+'\n')

        for ISEC in range(0, len(Sektor)):
            for JSEC in range(len(Sect_Ticket[ISEC])):
                              
                if not add_missing_data(Value_at_Close, DDT_ALL, ISEC, JSEC):
                    Display_info('Inconsistent data [3]... ' + tempory_csv + ' not gererated',app)
                    file1.close()
                    General_ML.Klean(tempory_csv)                       
                    return
                
                if not add_missing_data(Volume_at_dt, DDT_ALL, ISEC, JSEC):
                    Display_info('Inconsistent data [4]... ' + tempory_csv + ' not gererated',app)
                    file1.close()
                    General_ML.Klean(tempory_csv)
                    return

        for n in range(0, len(DDT_ALL)):
            st=DDT_ALL_TXT[n] + sep + str(DDT_ALL[n]) + sep*2 
            for ISEC in range(0, len(Sektor)):
                AA=''
                for JSEC in range(len(Sect_Ticket[ISEC])):
                    AA = AA + str(Volume_at_dt[ISEC][JSEC][n]) + sep               
                st= st +  AA + sep
            #Tools.Value_at_Close[ISEC][JSEC].append(float(line))

            file1.write(st+'\n')  
            #end for n=0..N_lignedata
        #end with open

    file1.close()
    Enable_disable(app.Distrib_btn, True)                   
    Display_info('Data transered in ' + tempory_csv,app)

def Jours_from_1st_January_M_d(a, m, d):
    mx=1
    H=0
    while mx<= m: 
       if mx==1:   #janvier
          H+=0
       elif mx==2:#fev >= +31j
          H+=31
       elif mx==3:#mars >= +28/29j
          H+=28
          if a % 4 ==0:
             H+=1
       elif mx==4:#avril >= +31j
          H+=31
       elif mx==5:#mai >= +30j
          H+=30
       elif mx==6:#juin >= +31j
           H+=31
       elif mx==7: #juillet >= +30j
          H+=30
       elif mx==8:#aout >= +31j
           H+=31
       elif mx==9:#sep >= +31j
           H+=31
       elif mx==10:
          H+=30
       elif mx==11:
           H+=31
       elif mx==12:
          H+=30
       mx+=1
    return H + d 
    #end Jours_from_1st_January_M_d

def Replace_Unallowed_Char_in_TXT_File(s):
    Result = ''
    i = 0
    while i < len(s)-1:
        a   =   s[i]
        i   +=  1   
        if a in ['❌','✅','📊','🚀','📈','✓','✔','🔍','⬛','⬜','∑','σ','²']:  # ces caratères ne sont pas supportés dans un save en fichier texte 
            if a =='✔':
                a   =  '   OK'
            elif a =='²':
                a   =   '2'
            elif a=='∑':
                a ='Somme '
            elif a =='σ':
                a=''
            elif a in ['⬛','⬜']:
                while True:
                    i+=1
                    if i>=len(s):
                        a=''
                        break
                    a=s[i]
                    if not (a in ['⬛','⬜']):
                        a   =   '=>  '
                        break
            else:
                a   =   '=>  '
        Result = Result + a
    Result = Result[:-1]
    return Result
    #end Replace_Unallowed_Char_in_TXT_File
    
def ajuste(s):    
    Result = ' '*30 
    return Result + s    
    #end ajuste

def Form_EXP( V):
    z=abs(V)
    if z < 1e-30:
        return '0.00'
    pw=0
    while z > 10:
        z=z/10
        pw+=1
    while z < 0.1:
        z=z*10
        pw-=1
    if pw==0:
        spw=''
    else:
        spw='E' + str(pw)                   
    Result = "{:2.6f}".format(z) + spw
    if V > 0:
       Result =  '+ ' + Result
    else:
       Result =  '- ' + Result
    return Result
    #end Form_EXP

def Get_date_time_as_a_string(Space): # PC data/time
    now = datetime.now()
    if Space:  
        return now.strftime("%Y-%m-%d  ,   %Hh %Mm %Ss")    # format to display   
    return     now.strftime("%Y-%m-%d_%H-%M-%S")            # format used in file name 
    #end Get_date_time_as_a_string

def INFO_001_():
    Result        =   '    P    R    O    G    R    A    M               S    T    A    R    T           ('   +  Get_date_time_as_a_string(True)   + ')'
    return Result

def Display_info(s, app):
    if Tools_QQUUIITT:
        print(s)
    else:
        try:
            if app !=None:
                try:
                    app.log_message(s)
                except:
                    print(s)
            else:
                print(s)
        except:
            print(s)

def Copy_filename_in_PDF(filename, cur_inp, Use_cur_inp):
    from fpdf import FPDF
    if Use_cur_inp:
        spl =   cur_inp.split('\n')
        k   =   len(spl)
    png_temp = 'image.png'
    if os.path.isfile(filename):
        pdf= FPDF(orientation = 'P', unit = 'pt', format='A4')
        pdf.add_page()
    
        if os.path.isfile(User_Defined_logo):
            from PIL import Image
            img = Image.open(User_Defined_logo)
            img.save(png_temp)
            pdf.image(png_temp, w=32, h=32)

        st = "Enhanced Multi-Sector Bubble Analysis with ML Prediction"
        pdf.set_font(family='arial', style='I', size=20) # BEFORE .cell(...)
        pdf.cell(w=0, h=50, txt='', align='C', ln=1)#w=0 => totalité width
        pdf.set_text_color(r=0x55, g=0x88, b=0xEE)
        pdf.cell(w=0, h=50, txt=st, align='C', border=1, ln=1)#w=0 => totalité width
        
        pdf.set_text_color(r=0x00, g=0x00, b=0x00)
        pdf.set_font(family='arial', style='B', size=24) # BEFORE .cell(...)
        pdf.cell(w=0, h=50, txt=' ', align='C', ln=1)#w=0 => totalité width
        if Use_cur_inp:
            for p in range(0, k):
                pdf.set_font(family='Times', style='I', size=8) # BEFORE .cell(...)
                st9 = spl[p]
                pdf.cell(w=0, h=20, txt=st9, align='L', ln=1)
        else:
            fl= open(filename, "r")
            while True:
                line    =   fl.readline()
                if not line:
                    break
                #[07:34:13] Collecting Technology
                if len(line) > 10:
                    if (line[0]=='[') and (line[3]==':') and (line[6]==':') and (line[9]==']'):
                       line = line[10:]
                pdf.set_font(family='Times', style='I', size=8) # BEFORE .cell(...)
                pdf.cell(w=0, h=20, txt=line, align='L', ln=1)

            fl.close()
        i=len(filename)
        while True:
            i-=1
            if (i==0) or(filename[i]=='.'):
                break
        if filename[i]=='.':
            fx=filename[:i]
        fx=fx + '.pdf'
        pdf.output(fx)
        General_ML.Klean(png_temp)
        return fx
    
'''   ONLY for test
def TST_PDF():
    from fpdf import FPDF
    pdf= FPDF(orientation = 'P', unit = 'pt', format='A4')
    pdf.add_page()
    
    if os.path.isfile(User_Defined_logo):
        from PIL import Image
        filename = User_Defined_logo
        img = Image.open(filename)
        img.save('image.png')
        pdf.image('image.png', w=32, h=32)
        
    pdf.set_font(family='arial', style='B', size=24) # BEFORE .cell(...)
    pdf.cell(w=0, h=50, txt='', align='C', ln=1)#w=0 => totalité width
    pdf.cell(w=0, h=50, txt='Line 1', align='C', border=1, ln=1)#w=0 => totalité width
        
    pdf.set_font(family='Times', style='B', size=14) # BEFORE .cell(...)
    pdf.cell(w=0, h=30, txt='Line 2', align='L', ln=1)#w=0 => totalité width

    pdf.set_font(family='Times', style='I', size=12) # BEFORE .cell(...)
    TXT='fcgbfgfgshf hfgd hfgdh fg sh gh fsgh fd hg fs fgsh fs h sfh gh j hdgk h hj dfgjh fs hfgs h h hfg dh dh fdgh  sfgh fgs hs h fshs'
    TXT = TXT + '  sfhg fs gh fsh sg h fsgh fs ghf sgh fh g sh fsh sfg hfs h fs h sgh sfgh'
    pdf.set_text_color(r=0x55, g=0x88, b=0xEE)
    pdf.multi_cell(w=0, h=30, txt=TXT, align='L')#w=0 => totalité width

    for n in range(0, 100):
        pdf.set_font(family='Times', style='B', size=10) # BEFORE .cell(...)
        pdf.cell(w=0, h=30, txt='Line ... ' + str(n), align='L', ln=1) #création automatique de nouvelles pages pdf si requis

    pdf.output('tst.pdf')
'''
            
def built_logo():
    Logo =  [            
              0, 0, 1, 0, 1, 0, 32, 32, 0, 0, 1, 0, 24, 0, 168, 12, 0, 0, 22, 0, 0, 0, 40, 0, 0, 
              0, 32, 0, 0, 0, 64, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 218, 61, 34, 219, 61, 32, 219, 61, 32, 219, 
              61, 31, 219, 61, 31, 219, 60, 31, 219, 60, 30, 219, 60, 30, 218, 60, 30, 218, 60, 29, 218, 59, 29, 218, 59, 
              28, 218, 59, 28, 218, 59, 28, 218, 59, 27, 218, 59, 27, 218, 58, 26, 218, 58, 26, 218, 58, 26, 218, 58, 25, 
              218, 58, 25, 218, 57, 24, 218, 57, 24, 218, 57, 24, 218, 57, 23, 217, 54, 25, 248, 218, 217, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 219, 63, 37, 254, 216, 156, 253, 218, 161, 253, 217, 158, 252, 215, 
              154, 252, 213, 150, 251, 211, 145, 250, 209, 141, 250, 207, 137, 249, 205, 132, 248, 200, 125, 247, 195, 116, 246, 189, 107, 
              245, 184, 98, 244, 179, 89, 242, 173, 80, 241, 168, 71, 240, 162, 62, 239, 157, 53, 238, 152, 44, 236, 146, 35, 235, 
              141, 26, 234, 135, 17, 233, 133, 10, 233, 143, 7, 218, 57, 23, 227, 80, 30, 248, 218, 214, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 221, 165, 254, 221, 165, 253, 219, 161, 252, 217, 158, 
              252, 216, 155, 252, 214, 152, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 127, 247, 193, 117, 246, 187, 107, 244, 
              181, 97, 243, 174, 86, 242, 168, 76, 240, 161, 66, 239, 155, 56, 238, 149, 46, 236, 142, 36, 235, 136, 25, 233, 129, 
              15, 233, 128, 8, 233, 143, 7, 218, 58, 24, 254, 161, 18, 227, 82, 29, 248, 216, 211, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 221, 165, 254, 221, 165, 253, 219, 161, 252, 217, 158, 252, 216, 155, 252, 
              214, 152, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 127, 247, 193, 117, 246, 187, 107, 244, 181, 97, 243, 174, 
              86, 242, 168, 76, 240, 161, 66, 239, 155, 56, 238, 149, 46, 236, 142, 36, 235, 136, 25, 233, 129, 15, 233, 127, 8, 
              233, 143, 7, 218, 58, 25, 254, 181, 70, 255, 167, 24, 227, 83, 29, 249, 221, 217, 255, 255, 255, 255, 255, 255, 219, 
              64, 37, 254, 216, 156, 254, 220, 165, 254, 221, 165, 253, 219, 161, 253, 217, 158, 252, 216, 155, 252, 214, 152, 251, 213, 
              149, 251, 211, 145, 250, 206, 137, 248, 200, 127, 247, 193, 117, 246, 187, 107, 245, 181, 97, 243, 174, 86, 242, 168, 76, 
              240, 161, 66, 239, 155, 56, 238, 149, 46, 236, 142, 36, 235, 136, 25, 234, 129, 15, 233, 127, 8, 233, 143, 7, 218, 
              58, 25, 255, 203, 126, 255, 187, 78, 255, 173, 30, 224, 89, 67, 255, 255, 255, 255, 255, 255, 223, 84, 62, 254, 216, 
              156, 253, 220, 165, 254, 221, 165, 253, 219, 161, 252, 217, 158, 252, 216, 155, 250, 208, 140, 250, 206, 135, 249, 203, 130, 
              248, 199, 122, 247, 194, 114, 246, 189, 105, 244, 184, 96, 243, 178, 87, 242, 173, 79, 241, 168, 70, 240, 163, 61, 239, 
              157, 52, 238, 152, 44, 236, 147, 35, 235, 142, 26, 234, 137, 17, 233, 134, 10, 233, 143, 7, 218, 59, 26, 254, 225, 
              181, 255, 208, 133, 254, 191, 81, 219, 65, 38, 255, 255, 255, 255, 255, 255, 227, 104, 88, 253, 211, 153, 253, 216, 156, 
              253, 221, 165, 253, 219, 161, 252, 217, 158, 252, 216, 155, 251, 210, 142, 247, 194, 113, 246, 190, 104, 224, 85, 43, 219, 
              62, 30, 219, 62, 29, 219, 62, 28, 219, 62, 28, 219, 61, 27, 219, 61, 27, 219, 61, 26, 218, 61, 26, 218, 60, 
              25, 218, 60, 25, 218, 60, 24, 218, 60, 24, 218, 59, 23, 218, 59, 23, 218, 57, 27, 254, 246, 234, 255, 229, 188, 
              254, 210, 132, 219, 65, 39, 255, 255, 255, 255, 255, 255, 253, 245, 245, 224, 91, 72, 252, 208, 147, 252, 214, 152, 253, 
              219, 161, 252, 217, 158, 252, 216, 155, 252, 214, 152, 250, 207, 138, 246, 191, 106, 236, 142, 81, 235, 65, 19, 249, 85, 
              4, 249, 89, 4, 249, 93, 4, 249, 96, 4, 249, 100, 4, 249, 104, 4, 249, 107, 4, 249, 111, 4, 249, 115, 4, 
              249, 118, 4, 249, 122, 4, 249, 126, 4, 249, 129, 4, 250, 140, 9, 254, 221, 172, 255, 249, 239, 254, 228, 182, 219, 
              65, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 243, 242, 224, 92, 71, 251, 206, 142, 252, 212, 148, 253, 217, 
              158, 252, 216, 155, 252, 214, 152, 251, 213, 149, 249, 205, 134, 246, 188, 99, 236, 142, 76, 241, 79, 19, 255, 100, 1, 
              255, 103, 0, 255, 108, 0, 255, 112, 0, 255, 116, 0, 255, 120, 0, 255, 124, 0, 255, 128, 0, 255, 132, 0, 255, 
              137, 0, 255, 141, 0, 255, 145, 0, 255, 149, 0, 255, 155, 3, 255, 214, 150, 255, 244, 226, 219, 65, 39, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 243, 242, 224, 91, 70, 250, 203, 135, 251, 210, 145, 252, 216, 155, 
              252, 214, 152, 251, 213, 149, 250, 211, 145, 248, 201, 126, 245, 185, 93, 236, 141, 72, 240, 88, 30, 254, 107, 6, 253, 
              104, 1, 253, 108, 1, 253, 112, 1, 253, 116, 1, 253, 120, 1, 253, 124, 1, 253, 128, 1, 253, 132, 1, 253, 136, 
              1, 253, 140, 1, 253, 144, 1, 253, 148, 1, 253, 153, 2, 253, 195, 107, 219, 65, 39, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 241, 241, 224, 89, 67, 249, 200, 129, 251, 209, 142, 252, 214, 152, 251, 
              213, 149, 250, 211, 145, 250, 206, 137, 247, 196, 117, 244, 181, 86, 235, 139, 67, 237, 95, 50, 218, 59, 39, 219, 65, 
              39, 219, 65, 39, 219, 65, 39, 219, 65, 39, 219, 65, 39, 219, 65, 39, 219, 65, 39, 219, 65, 39, 219, 66, 39, 
              219, 66, 39, 219, 66, 39, 219, 66, 39, 219, 66, 39, 219, 68, 42, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 251, 238, 238, 225, 89, 66, 249, 198, 124, 250, 208, 139, 251, 213, 149, 250, 211, 
              145, 250, 206, 137, 248, 200, 127, 246, 190, 108, 243, 178, 80, 235, 138, 63, 235, 99, 64, 235, 138, 136, 254, 253, 253, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 251, 238, 238, 224, 88, 64, 248, 195, 117, 250, 207, 137, 251, 211, 145, 250, 206, 137, 
              248, 200, 127, 247, 193, 117, 245, 184, 99, 242, 175, 73, 234, 137, 59, 235, 108, 76, 235, 142, 139, 254, 253, 253, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 251, 237, 236, 224, 88, 63, 247, 193, 112, 249, 206, 134, 250, 206, 137, 248, 200, 127, 247, 
              193, 117, 246, 187, 107, 244, 179, 89, 241, 172, 66, 233, 135, 54, 235, 116, 88, 235, 142, 141, 254, 253, 253, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 251, 235, 234, 224, 88, 61, 246, 191, 106, 248, 201, 128, 248, 200, 127, 247, 193, 117, 246, 187, 
              107, 244, 181, 97, 242, 174, 80, 240, 168, 60, 233, 134, 49, 234, 122, 99, 235, 144, 142, 254, 253, 253, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 248, 222, 220, 225, 90, 52, 246, 188, 101, 248, 198, 122, 247, 193, 117, 246, 187, 107, 244, 181, 97, 
              243, 174, 86, 241, 168, 70, 239, 165, 53, 229, 112, 43, 242, 139, 98, 236, 144, 140, 254, 252, 252, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 235, 234, 
              224, 86, 62, 246, 188, 103, 247, 194, 113, 248, 200, 126, 247, 193, 117, 246, 187, 107, 245, 181, 97, 242, 174, 84, 240, 
              168, 62, 233, 135, 49, 237, 86, 16, 254, 124, 0, 251, 150, 65, 234, 136, 131, 254, 252, 252, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 230, 229, 224, 89, 62, 247, 192, 110, 248, 
              200, 123, 249, 206, 137, 248, 200, 127, 247, 193, 117, 246, 187, 107, 244, 180, 92, 241, 172, 68, 233, 129, 52, 240, 88, 
              13, 254, 120, 1, 255, 124, 0, 254, 128, 1, 251, 136, 31, 234, 129, 117, 254, 252, 252, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 249, 223, 222, 224, 90, 62, 248, 196, 117, 249, 204, 131, 250, 211, 145, 250, 206, 
              137, 248, 200, 127, 247, 193, 117, 245, 185, 100, 242, 175, 74, 232, 123, 53, 242, 88, 10, 254, 116, 1, 254, 120, 1, 
              255, 124, 0, 254, 128, 1, 254, 132, 1, 236, 95, 31, 248, 220, 220, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 247, 213, 213, 226, 95, 64, 249, 200, 124, 250, 207, 138, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 127, 
              246, 190, 108, 243, 178, 80, 232, 115, 54, 245, 88, 8, 254, 112, 1, 254, 116, 1, 254, 120, 1, 255, 124, 0, 254, 
              128, 1, 236, 98, 39, 248, 215, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 245, 205, 204, 226, 99, 66, 
              250, 203, 131, 251, 210, 144, 252, 214, 152, 251, 213, 149, 251, 211, 145, 250, 206, 137, 247, 194, 115, 243, 181, 86, 231, 
              106, 52, 247, 89, 5, 255, 108, 0, 255, 112, 0, 255, 116, 0, 255, 120, 0, 254, 123, 0, 235, 96, 50, 250, 226, 
              223, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 243, 195, 195, 228, 104, 69, 251, 207, 138, 252, 213, 150, 252, 
              216, 155, 252, 214, 152, 251, 213, 149, 250, 211, 145, 248, 198, 121, 244, 182, 91, 231, 97, 50, 249, 89, 4, 254, 103, 
              1, 255, 108, 0, 254, 112, 1, 254, 116, 1, 253, 117, 1, 233, 99, 65, 252, 236, 234, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 242, 185, 185, 229, 108, 74, 252, 210, 145, 252, 216, 156, 252, 217, 158, 252, 216, 155, 252, 214, 
              152, 251, 213, 148, 248, 201, 124, 244, 182, 96, 231, 89, 46, 251, 89, 2, 254, 99, 1, 254, 103, 1, 255, 108, 0, 
              254, 112, 1, 252, 111, 1, 232, 105, 82, 252, 242, 241, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 242, 184, 
              184, 230, 114, 78, 253, 213, 152, 253, 219, 161, 253, 219, 161, 252, 217, 158, 252, 216, 155, 251, 214, 151, 249, 202, 127, 
              246, 191, 105, 222, 74, 39, 219, 56, 26, 219, 57, 25, 219, 57, 25, 219, 57, 25, 219, 57, 25, 219, 57, 25, 218, 
              56, 25, 218, 60, 31, 218, 61, 32, 218, 61, 32, 218, 61, 31, 218, 61, 31, 218, 60, 31, 218, 60, 31, 218, 61, 
              34, 244, 193, 186, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 223, 84, 62, 254, 216, 156, 
              253, 220, 164, 254, 221, 165, 253, 219, 161, 253, 217, 158, 252, 216, 155, 251, 210, 144, 249, 202, 128, 248, 200, 123, 247, 
              196, 116, 246, 191, 108, 245, 187, 100, 244, 182, 91, 243, 177, 83, 242, 173, 75, 241, 168, 67, 240, 163, 59, 238, 159, 
              51, 237, 154, 43, 236, 149, 34, 235, 145, 26, 234, 140, 18, 233, 142, 12, 231, 137, 9, 219, 57, 23, 220, 67, 33, 
              248, 215, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 221, 165, 254, 
              221, 165, 253, 219, 161, 252, 217, 158, 252, 216, 155, 252, 214, 152, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 
              127, 247, 193, 117, 246, 187, 107, 244, 181, 97, 243, 174, 86, 242, 168, 76, 240, 161, 66, 239, 155, 56, 238, 149, 46, 
              236, 142, 36, 235, 136, 25, 233, 129, 15, 233, 137, 11, 231, 137, 9, 219, 58, 23, 254, 153, 1, 229, 87, 19, 238, 
              154, 138, 255, 255, 255, 255, 255, 255, 255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 221, 165, 254, 221, 165, 253, 219, 
              161, 252, 217, 158, 252, 216, 155, 252, 214, 152, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 127, 247, 193, 117, 
              246, 187, 107, 244, 181, 97, 243, 174, 86, 242, 168, 76, 240, 161, 66, 239, 155, 56, 238, 149, 46, 236, 142, 36, 235, 
              136, 25, 233, 129, 15, 233, 137, 11, 231, 137, 9, 219, 59, 24, 254, 161, 20, 255, 163, 15, 222, 69, 26, 254, 254, 
              254, 255, 255, 255, 255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 221, 165, 254, 221, 165, 253, 219, 161, 252, 217, 158, 
              252, 216, 155, 252, 214, 152, 251, 213, 149, 250, 211, 145, 250, 206, 137, 248, 200, 127, 247, 193, 117, 246, 187, 107, 244, 
              181, 97, 243, 174, 86, 242, 168, 76, 240, 161, 66, 239, 155, 56, 238, 149, 46, 236, 142, 36, 235, 136, 25, 233, 129, 
              15, 233, 137, 11, 231, 137, 9, 219, 60, 28, 254, 185, 78, 255, 185, 72, 219, 59, 26, 254, 254, 254, 255, 255, 255, 
              255, 255, 255, 219, 64, 37, 254, 216, 156, 253, 220, 164, 253, 220, 163, 253, 218, 159, 252, 216, 156, 252, 214, 152, 251, 
              213, 149, 251, 211, 145, 250, 209, 141, 249, 204, 134, 248, 198, 124, 247, 192, 114, 245, 186, 104, 244, 180, 95, 243, 174, 
              85, 242, 168, 75, 240, 161, 65, 239, 155, 55, 238, 149, 46, 236, 143, 36, 235, 137, 26, 234, 131, 16, 233, 138, 11, 
              231, 137, 9, 219, 62, 31, 255, 208, 138, 255, 208, 131, 219, 59, 29, 254, 254, 254, 255, 255, 255, 255, 255, 255, 218, 
              61, 34, 219, 63, 34, 219, 63, 33, 219, 63, 33, 219, 63, 32, 219, 62, 32, 219, 62, 31, 219, 62, 31, 219, 62, 
              30, 219, 61, 30, 219, 61, 30, 219, 61, 29, 219, 61, 29, 219, 61, 28, 219, 60, 28, 218, 60, 27, 218, 60, 27, 
              218, 60, 26, 218, 59, 26, 218, 59, 25, 218, 59, 25, 218, 59, 25, 218, 59, 24, 218, 58, 24, 218, 58, 23, 219, 
              62, 33, 254, 232, 198, 255, 230, 191, 219, 60, 31, 254, 254, 254, 255, 255, 255, 255, 255, 255, 252, 241, 242, 238, 162, 
              166, 225, 67, 50, 240, 45, 11, 252, 59, 1, 252, 63, 1, 252, 67, 1, 252, 71, 1, 252, 75, 1, 252, 79, 1, 
              252, 83, 1, 252, 87, 1, 252, 91, 1, 252, 95, 1, 252, 99, 1, 252, 103, 1, 252, 107, 1, 252, 111, 1, 252, 
              115, 1, 252, 119, 1, 252, 123, 1, 252, 127, 1, 252, 131, 1, 252, 135, 1, 252, 139, 1, 254, 156, 20, 254, 204, 
              128, 255, 244, 228, 219, 61, 34, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
              250, 224, 224, 228, 98, 82, 236, 78, 58, 235, 63, 37, 235, 66, 37, 235, 68, 37, 235, 70, 37, 235, 72, 37, 235, 
              74, 37, 235, 76, 37, 235, 78, 37, 235, 80, 37, 235, 82, 37, 235, 84, 37, 235, 86, 37, 235, 88, 37, 235, 90, 
              37, 235, 92, 37, 235, 94, 37, 235, 96, 37, 235, 98, 37, 235, 101, 37, 235, 102, 37, 235, 104, 37, 235, 111, 50, 
              220, 67, 43, 254, 254, 254, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
           ] 

    byte_array = bytearray(Logo)
    dst = open(User_Defined_logo, "wb")
    try:	        
        dst.write(byte_array)    
    except IOError:	    
        pass    
    finally:
        dst.close()

    # TST_PDF() just for test  TXT  => Logo +  PDF

    #fin de built_logo

def built_styl():
    style_info = [            
        'xtick.minor.visible : True',
        'xtick.direction : in',
        'xtick.labelsize : large',
        'ytick.minor.visible : True',
        'ytick.direction : in',
        'ytick.labelsize : large',
        'axes.labelsize : xx-large',
        'axes.labelpad : 10',
        'figure.subplot.bottom: 0.15',
        'figure.subplot.left: 0.15',
        'errorbar.capsize : 2 ',
        'scatter.edgecolors: black',
        'lines.markeredgecolor : black',
        'legend.edgecolor : 0',
        'savefig.format : pdf',
        # the last line is only present in Mon_Modele_STYLE2
        "axes.prop_cycle : cycler('marker', ['o','o','s','s'])+cycler(markerfacecolor=['w','k','w','k'])+cycler(linestyle=['None','None','None','None'])"
        ]    
   
    Mon_Modele_STYLE= Mon_Modele_STYLE1
    N_element=len(style_info)-1 # remove last table line
    with open(Mon_Modele_STYLE,'w') as file1: 
        for i in range(0, N_element):
            file1.write(style_info[i] + '\n')
        file1.close()
   
    Mon_Modele_STYLE= Mon_Modele_STYLE2
    N_element += 1   #include all tabel lines
    with open(Mon_Modele_STYLE,'w') as file1: 
        for i in range(0, N_element):
            file1.write(style_info[i] + '\n')
        file1.close()

''' jut to test .pop()
TTT=list(['1','2','3','4','5','6'])
print(TTT, len(TTT))=> ['1','2','3','4','5','6'] 6
TTT.pop()
TTT.pop()
print(TTT, len(TTT)) => ['1','2','3','4'] 4
'''
