"""
Historical Comparator - Compare current sectors with historical bubbles
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-


from    math                 import  * 
import  math

KKst_1  =   1/log(1001/20)
KKst_2  =   1/ (exp(1000/443)-1)

def Weighting_POLY ( pct, model, mi_pct):
    i= 1000-10*pct
    mi=mi_pct/100
    if model==0:  # fixe
        Result=1
    elif model==1: #linear 0%  older => 100& weight
        Result=(i/1000)*(1-mi) + mi  # mi..1
    elif model==2: # Log
        Result=mi + log(1+i/20)*(1-mi)*KKst_1
    else:
        Result=mi+(1-mi)*(exp(i/443)-1)*KKst_2
    return Result

def sort(l, r, Y): # re-organize Y in ascending order
    i                       =   l
    j                       =   r
    pivot                   =   Y[floor((l+r) /2)]
    while ( i <=j ):     
        while Y[i]  <  pivot:
            i               +=  1
        while pivot  < Y[j]:
            j               -=  1
        if i <= j: # swap i and j elements
            swap            =   Y[i]
            Y[i]            =   Y[j]
            Y[j]            =   swap
            i               +=  1
            j               -=  1
    if l < j:
        Y= sort(l,j, Y)  # Be carfull! This routine calls itself => in some case, could generate stack overflow
    if i < r:
        Y= sort(i,r, Y) # same remark
    return Y
    #end sort

a_mat   = list([  list([])  ])
b_mat   = list([]) 

def solu(dg, coef): # function solu : boolean;
    i   =   dg
    while ( i>=0):   #for i:=dg downto 0 do
        u   =   b_mat[i]
        if i < dg:            
            for j in range (i+1, dg+1): #j:=i+1 to dg do            
                u   -= a_mat[i][j]*coef[j];
                #fin de for j:=i+1 to dg do 
            #fin de if i < dg
        if abs(a_mat[i][i]) < 1e-200:
            return False   #   sort de la routine solu avec False 
        coef[i]     =   u/a_mat[i][i]            
        i  -= 1
        #fin de while i >=0
    return True  #sort de la routine solu avec True
    #fin de solu

def permut( i , dg):
    arot_    = list([] )   
    #for j:=i to dg do arot[j]:=a_mat[i,j];
    for j  in range (0,dg+1):
        arot_.append(0.0)
    for j in range (i, dg+1):
        arot_[j]=a_mat[i][j]
        #Fin #for j:=i to dg do
    brot=b_mat[i] 
    for j in range (i,dg): #j:=i to dg-1      
        for k in range (i,dg+1): #  k:=i to dg do 
            a_mat[j][k]=a_mat[j+1][k]
            #Fin for k:=i to dg do 
        b_mat[j]=b_mat[j+1]
        #FIN for j:=i to dg-1
    for k in range (i,dg+1): #for  k:=i to dg do
        a_mat[dg][k]=arot_[k]
        #FIN #for  k:=i to dg do
    b_mat[dg]=brot
    #fin permut
    
def triangle ( dg):  
    pivot     = list([]) 
    for j in range(0, dg+1):
        pivot.append(0.0)

    for i in range(0,dg): # for i:=0 to dg-1 do
        ktest=0
        while True: 
            if ktest > dg:
                return False   #sort de la routine triangle avec FALSE. on a fini tous les permutation; aucune n'est acceptable 
            if abs(a_mat[i][i]) < 1e-30:            
                permut( i , dg)
                ktest   +=  1  # compteur d'itération pour permut
            else:
                break #sort de while True => la permutation actuelle donne un pivot=a_mat[i][i] !=0
            
        for j in range(i+1, dg+1): #for j:=i+1 to dg do            
            pivot[j]=a_mat[i][j]/a_mat[i][i]
            #fin #for j:=i+1 to dg do
        for j in range(i+1,dg+1):  #=i+1 to dg do
            for k in range(i, dg+1): #k:=i to dg do
                a_mat[j][k]=a_mat[j][k]-pivot[j]*a_mat[i][k]
                #fin for k:=i to dg do
            b_mat[j]=b_mat[j]-pivot[j]*b_mat[i]
            #fin for j=io+1 to dg
        #fin de if permut_allowed
    #END for i:=0 to dg-1 do
    return True #sot de la routine triangle avec True
    #Fin Triangle


def WWGHT(k, N_key_point, XXp, weight_metod, PCT_03):
    H=(XXp[N_key_point-1]-XXp[k])/max(1e-30, XXp[N_key_point-1]-XXp[0])
    H=min(1,max(0,H))
    return Weighting_POLY ( H*100, weight_metod, PCT_03)

def Poly_Fit_XX_YY_deg( dg, XXp, YYp, weight_metod, PCT_03):            
    coef = list([])
    b_mat.clear() 
    a_mat.clear()
    coef.clear()
    for j in range (0, dg + 1):  
        b_mat.append(0.0)
        coef.append(0.0)
        a_mat.append(list([])); 
        a_mat[j].clear()
        for k in range (0, dg + 1):
            a_mat[j].append(0.0)    
    N_key_point=len(XXp)   
    for i in range(0, dg + 1):
        
        for k in range(0, N_key_point):
            WW= WWGHT(k, N_key_point, XXp, weight_metod, PCT_03)
            b_mat[i]=b_mat[i] + YYp[k] * WW * XXp[k] ** i #   somme k=1..ndata    Y[k] x[k]^î 

        for j in range (0, dg + 1):                  
            for k in range (0, N_key_point):
                WW= WWGHT(k, N_key_point, XXp, weight_metod, PCT_03)
                a_mat[i][j] = a_mat[i][j] + WW * (XXp[k] **  (i+j))

    ERROR_FIT=''
    OK = triangle( dg)
    if OK:
        OK=solu(dg, coef)
        if not OK:            
           ERROR_FIT ="No solution found"
    else:
        ERROR_FIT= "triangularisation failed"
    return OK, ERROR_FIT, coef
    #fin de Poly_Fit_XX_YY_deg

def Make_Y_model(x, dg, coef): 
    Result=0  
    for j in range (0,dg+1):#for j:=0 to dg do
        Result = Result + coef[j] * (x ** j)   #inc(Power_ent(_x^,j)*coef[j];
    return Result

    
'''
   ==========================================
        end Polynomial fit     section       
   =========================================
'''


def smooth(IN, A):
    OUT = list([])
    SM=0
    l=len(IN)
    AVERAGE = max(3,floor(l*A/100))
    for n in range(0, l):
        k   =   -AVERAGE
        U   =   0
        z   =0
        while True:
            U += IN[max(0, min(l-1,n+k))]
            k+=1
            if k==AVERAGE+1:
                OUT.append(U)
                SM += OUT[n]
                break            
    for n in range(0, l):
        OUT[n]=OUT[n]*100/SM
    return OUT
    
def Hamming(u, k, extn):
    P       =   k * pi/extn
    P       =   u + (1-u)*cos(P)
    return P


def SplineValue( t,  Y0,  Y1,  M0,  M1 ):
    #cubic polynome with Y(t=0)=Y0, Y(t=1)=Y1, Y'(t=0)=M0, Y'(t=1)=M1 
    t2 = t ** 2
    t3 = t * t2
    Spl = Y0 * (2*t3  - 3*t2 + 1 ) +  M0 * (t3 - 2*t2 + t ) +  Y1 * (-2*t3 + 3*t2 ) +  M1 * (t3 - t2)
    d_Spl = Y0 * (6*t2  - 6*t  ) +  M0 * (3*t2 - 4*t + 1 ) +  Y1 * (-6*t2 + 6*t ) +  M1 * (3*t2 - 2*t)
    return  Spl, d_Spl

def get_M1(a, X, Y, mx):
    b = min(a+2, mx)
    dt1 = X[b]- X[a]

    if abs(dt1) >= 1e-100: #{ Show Message("SMALL DX1"); continue; }            
       M1 = 1/dt1
    else:
       M1  =   0
    return M1 * ( Y[b]- Y[a])

def Compute_Splines(XX_in, YY_in, pct, Spline_X, Spline_Y, Spline_dY):   
    XXp     =   list([])
    YYp     =   list([])
    li      =   len(XX_in)
    extn    =   round(li * pct/100)
    HK      =   1/ (1+2*extn)
    i       =   0
    OX      =   -1e30
    ip      =   max(1, floor(extn/4))
    while True:
        IE  = floor(i*ip)
        if IE >=li:
            break
        x   =   XX_in[IE]
        y   =   YY_in[IE]
        for k in range(1, extn+1):
            HAM     =   Hamming(0.5, k, extn) # Hamming filter coefficient  (0,54 => Hanning)
            x       +=  XX_in[min(li-1,IE+k)]
            x       +=  XX_in[max(0,IE-k)]
            y       +=  YY_in[min(li-1,IE+k)]*HAM
            y       +=  YY_in[max(0,IE-k)]*HAM
        x *= HK
        if x-OX > 1e-10:
            XXp.append(x)
            YYp.append(y/extn)
            OX = x
        i +=1
    ndata_in_file=len(XXp)
    if ndata_in_file < 5:
        return False
    N_key_point = ndata_in_file
    Spline_X.clear()
    Spline_Y.clear()
    Spline_dY.clear()
    M1 = 0
    dt0 = XXp[1]- XXp[0]
    M0 = 1/dt0 
    M0 = M0 * ( YYp[1]- YYp[0]) * dt0
    _Max_pts = ndata_in_file
    _Max_pts = math.trunc(ndata_in_file/(N_key_point-1));
    Scal = _Max_pts
    Scal = 1 / Scal
    MM =_Max_pts
    MM_s=0
    for i in range (0, N_key_point-1):
        MM_s=MM_s+MM
    decal_due_to_round=ndata_in_file-MM_s
    MM_0=MM
    for i in range (0, N_key_point-1):#for i:=0 to PK-1 do
        MM=MM_0
        if i < decal_due_to_round:   
            MM = MM + 1
        M1  = get_M1(i, XXp, YYp, N_key_point-1)
        
        for p in range (0, 25): # 25 Arbitrary value. It's enough to get a nice plot                
            rho = p/24
            x = XXp[i] + rho * (XXp[i+1]-XXp[i]);
            y, dy = SplineValue(rho, YYp[i], YYp[i+1],M0 ,M1 );
            Spline_X.append(x)
            Spline_Y.append(y)
            Spline_dY.append(dy)
            #end for p in range (0,25)
        M0  =   M1 
        #end loop for i in range (0, PK)
    return True
    #fin de Compute_Splines

'''=============================='''
'''   END  SPLINE     section    '''
'''==============================''' 
