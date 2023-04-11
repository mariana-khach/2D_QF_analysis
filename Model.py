#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:21:52 2023

@author: marianak
"""

import numpy as np
import scipy.interpolate as SI_INT
# normalized distribution functions
from numba_stats import truncnorm


class BP:
    def __init__(self,n,interval=[0,1]):
        self.size=n
        self.int=interval
        self.norm=(interval[1]-interval[0])/(n+1)
        coef=(n+1)*[[0]]
        b_pol=[]
        for i in range(n+1):
            coef[i]=[1]
            b_pol.append(SI_INT.BPoly(coef,self.int))
            coef[i]=[0]
        self.bpol=b_pol
    def __getitem__(self,n):
        return self.bpol[n]
    
    
    
    
    
class BLin:
    def __init__(self,xr,yr,z):
        """
        b-linear interpolation in a rectangular are defined by 4 corners
        
        # The 4 corners are :

        P1 = S.Matrix([x1, y1, z1])
        
        P2 = S.Matrix([x2, y1, z2])
        
        P3 = S.Matrix([x2, y2, z3])
        
        P4 = S.Matrix([x1, y2, z4])

        Parameters
        ----------
        xr : size 2, float array
            limits of the x-range.
        yr : size 2, float array
            limits of the y-range.
        z : size 4, parameter array
            the z values for the 4 corners: z1, z2, z3, z4
            these are fit parameters

        Returns
        -------

        """
        # these are the xy ranges 
        self.xrange=xr
        self.yrange=yr
      

    def Get_value(self,x,y,z1,z2,z3,z4):
                
        """
        calculate interpolated value
            z = (x*y*(z1 - z2 + z3 - z4) + 
                 x*(-y1*z3 + y1*z4 - y2*z1 + y2*z2) + 
                 x1*y1*z3 - x1*y2*z2 - x2*y1*z4 + x2*y2*z1 + 
                 y*(x1*z2 - x1*z3 - x2*z1 + x2*z4))/
                ((x1 - x2)*(y1 - y2))

        """
        x1 = self.xrange[0]
        y1 = self.yrange[0]
        
        x2 = self.xrange[1]
        y2 = self.yrange[1]
        
        
        zval= (x*y*(z1 - z2 + z3 - z4) + 
             x*(-y1*z3 + y1*z4 - y2*z1 + y2*z2) + 
             x1*y1*z3 - x1*y2*z2 - x2*y1*z4 + x2*y2*z1 + 
             y*(x1*z2 - x1*z3 - x2*z1 + x2*z4)) / ( (x1 - x2)*(y1 - y2) )

        self.norm = (x1 - x2)*(y1 - y2)*(z1 + z2 + z3 + z4)/4
        
        return zval
    
    def Get_value_pdf(self,x,y,z1,z2,z3,z4):
        
        
        x1 = self.xrange[0]
        y1 = self.yrange[0]

        x2 = self.xrange[1]
        y2 = self.yrange[1]

        self.norm = (x1 - x2)*(y1 - y2)*(z1 + z2 + z3 + z4)/4

        cdf = (x1 - x)*(y1 - y)*(x1*y1*z1 + x1*y1*z2 + x1*y1*z3 + x1*y1*z4 - \
                                     2*x1*y2*z1 - 2*x1*y2*z2 + x1*y*z1 + x1*y*z2 - \
                                     x1*y*z3 - x1*y*z4 - 2*x2*y1*z1 - 2*x2*y1*z4 + \
                                     4*x2*y2*z1 - 2*x2*y*z1 + 2*x2*y*z4 + x*y1*z1 - \
                                     x*y1*z2 - x*y1*z3 + x*y1*z4 - 2*x*y2*z1 + \
                                     2*x*y2*z2 + x*y*z1 - x*y*z2 + 
                                     x*y*z3 - x*y*z4)/ \
                                    (4*(x1 - x2)*(y1 - y2))
        return cdf/self.norm














class analysis:
    
    def __init__(self,N_om,N_etap,int_om,int_etap,int_zx,int_zy,z_par):
        
        self.Nom=N_om
        self.Netap=N_etap
        self.intom=int_om
        self.intetap=int_etap
        self.intzx=int_zx
        self.intzy=int_zy
        self.zpar=z_par
        
        
    #For details check https://iminuit.readthedocs.io/en/stable/notebooks/cost_functions.html
    # Dependence on etaprime mass is denoted by y and dependence on omega mass denoted by x    
    def Omega_PDF(self,x,y,cy0,cy1,cy2,x0,sig_x):
        
        # Bernstein polynomial for omega amplitude
        BP_omega = BP(self.Nom, self.intom)      
        
        Amp_omega=(cy0*BP_omega[0](y)+cy1*BP_omega[1](y)+cy2*BP_omega[2](y))/BP_omega.norm
        Dist_omega=truncnorm.pdf(x,self.intetap[0], self.intetap[1],x0,sig_x)
        return Amp_omega*Dist_omega





    def Eta_PDF(self,x,y,cye0,cye1,cye2,x0e,sig_xe):
        
        # Bernstein polynomial for omega amplitude
        BP_omega = BP(self.Nom, self.intom)   
        
        Amp_eta=(cye0*BP_omega[0](y)+cye1*BP_omega[1](y)+cye2*BP_omega[2](y))/BP_omega.norm
        Dist_eta=truncnorm.pdf(x,self.intetap[0], self.intetap[1],x0e,sig_xe)
        return Amp_eta*Dist_eta



    def Truncnorm_sum(self,x,xmin,xmax,x0,sig,co_A,co_sig):
        
        return (truncnorm.pdf(x,xmin,xmax,x0,sig)+co_A*truncnorm.pdf(x,xmin,xmax,x0,co_sig*sig))/(1+co_A)



    def Etprime_PDF(self,x,y,cx0,cx1,cx2,cx3,y0,sig_y,co_A,co_sig_y):
        
        # Bernstein polynomial for etap amplitude
        BP_etap = BP(self.Netap, self.intetap)    
        
        Amp_etap=(cx0*BP_etap[1](x)+cx1*BP_etap[3](x)+cx2*BP_etap[5](x)+cx3*BP_etap[6](x))/BP_etap.norm
        Dist_etap=self.Truncnorm_sum(y,self.intom[0],self.intom[1],y0,sig_y,co_A,co_sig_y)
        return Amp_etap*Dist_etap
    
    
    def Bckg_PDF(self,x,y,z1,z2,z3,z4):
        
        # bi-linear background
        Bckg = BLin(self.intzx, self.intzy, self.zpar)
        
        return Bckg.Get_value(x,y,z1,z2,z3,z4)/Bckg.norm
    
    
    
    #Total function
    def Total_F_PDF(self,xy, cy0,cy1,cy2,x0,sig_x,\
                cye0,cye1,cye2,x0e,sig_xe,\
                cx0,cx1,cx2,cx3,y0,sig_y,co_A,co_sig_y,\
                z1,z2,z3,z4 ):
        
        x,y=xy
        norm=cy0+cy1+cy2+cye0+cye1+cye2+cx0+cx1+cx2+cx3+1
        
        tot= np.array([self.Omega_PDF(x,y,cy0,cy1,cy2,x0,sig_x)/norm,\
                self.Eta_PDF(x,y,cye0,cye1,cye2,x0e,sig_xe)/norm,\
                self.Etprime_PDF(x,y,cx0,cx1,cx2,cx3,y0,sig_y,co_A,co_sig_y)/norm,\
                self.Bckg_PDF(x,y,z1,z2,z3,z4)/norm])
        
        return tot




    #Logarithm of total function    
    def Total_logF_PDF(self,xy, cy0,cy1,cy2,x0,sig_x,\
                cye0,cye1,cye2,x0e,sig_xe,\
                cx0,cx1,cx2,cx3,y0,sig_y,co_A,co_sig_y,\
                z1,z2,z3,z4):
        
        total= self.Total_F_PDF(xy, cy0,cy1,cy2,x0,sig_x,\
                cye0,cye1,cye2,x0e,sig_xe,\
                cx0,cx1,cx2,cx3,y0,sig_y,co_A,co_sig_y,\
                z1,z2,z3,z4 )
    
        return np.log(np.sum(total,axis=0))


