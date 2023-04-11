#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:37:06 2023

@author: marianak
"""


import numpy as np
#Lab Tools package from Werner Boeglin, FIU
import LT.box as B

import scipy.spatial.distance as SPD

#Importing Minuit object
from iminuit import Minuit
#Cost function for Minuit minimization
import iminuit.cost as cost

import math
import copy as C
import pickle

#Importing my modules 
import File_read_write as FRW
import Model

import vector

DegtoRad=math.pi/180.


#%%

# load historgram

#h = B.histo2d(file = 'Metaprime_Momega_totalMC_rebin_diffrange_new.data')
h = B.histo2d(file = 'Metaprime_Momega_omegaMC_rebin_origrange.data')


# bin centers
xb = h.x_bin_center
yb = h.y_bin_center

# bin edges
xbe = h.x_bins
ybe = h.y_bins

# labels

l_omega =  r'$M_{\pi^{+},\pi^{-},\pi^{0}}$'
l_etap = r'$M_{\pi^{+},\pi^{-},\eta}$'
l_title = ''





file='2017_widecoherent_good_events_signal_8komega_20ksignal.npz'
data_dict=FRW.NPZ_file(file)
Momega=data_dict['mpippimpi0']
Metap=data_dict['metap']

#Selecting events in x and y range of the above histogram
sel_Momega=B.in_between(xbe.min(),xbe.max(),Momega)
sel_Metap=B.in_between(ybe.min(), ybe.max(),Metap)

sel_Momega_Metap=sel_Momega & sel_Metap



#Selecting the remaining variables
M_omega=Momega[sel_Momega_Metap]
M_etap=Metap[sel_Momega_Metap]

CosT_hx = data_dict['cost_hx'][sel_Momega_Metap]
Phi_hx = data_dict['phi_hx'] [sel_Momega_Metap]* DegtoRad

Pol = data_dict['pol'][sel_Momega_Metap]

Px_p = data_dict['px_pr'][sel_Momega_Metap]
Px_etapr = data_dict['px_etapr'][sel_Momega_Metap]
Px_pi0 = data_dict['px_pi0'][sel_Momega_Metap]

Py_p = data_dict['py_pr'][sel_Momega_Metap]
Py_etapr = data_dict['py_etapr'][sel_Momega_Metap]
Py_pi0 = data_dict['py_pi0'][sel_Momega_Metap]

Pz_p = data_dict['pz_pr'][sel_Momega_Metap]
Pz_etapr = data_dict['pz_etapr'][sel_Momega_Metap]
Pz_pi0 = data_dict['pz_pi0'][sel_Momega_Metap]

E_p = data_dict['e_pr'][sel_Momega_Metap]
E_etapr = data_dict['e_etapr'][sel_Momega_Metap]
E_pi0 = data_dict['e_pi0'][sel_Momega_Metap]

Px_beam = data_dict['px_beam'][sel_Momega_Metap]
Py_beam = data_dict['py_beam'][sel_Momega_Metap]
Pz_beam = data_dict['pz_beam'][sel_Momega_Metap]
E_beam = data_dict['e_beam'][sel_Momega_Metap]




#%%
#Fit etaprime with sume of two Gaussians and eta and omega with single Gaussian each
# fit 2 gaussians
#Limits for fit paramters
no_bound = [-np.inf, np.inf]
is_pos =   [1e-6, np.inf]
is_prob = [1e-6, 1.]


# initial values
IV = {}


# omega peak
IV['x0'] = [0.780, [0.775, 0.785]]
IV['sig_x'] = [0.0129, [0.008, 0.02]]

# omega amplitude as a function of etap mass 
IV['cy0'] = [0.07, is_pos]
IV['cy1'] = [0.1, is_pos]
IV['cy2'] = [0.1, is_pos]


# eta peak
IV['x0e'] = [0.555, [0.545, 0.565]]
IV['sig_xe'] = [0.0143, [0.005, 0.015]]

# eta amplitude as a function of etap mass 
IV['cye0'] = [0., is_pos]
IV['cye1'] = [0., is_pos]
IV['cye2'] = [0., is_pos]

# etap peak
IV['y0'] = [0.957, [0.955, 0.965]]
IV['sig_y'] = [0.658, [0.006, 0.015]]
IV['co_A'] = [0.5, [0,0.5]]
IV['co_sig_y'] = [0.52, [0,50]]
 

# etap Amplitude as a function of omega mass
IV['cx0'] = [0., is_pos]
IV['cx1'] = [10., is_pos]
IV['cx2'] = [10., is_pos]
IV['cx3'] = [10., is_pos]


# backgrpound paramters z-value at the edges
xr = [xbe.min(), xbe.max()]
yr = [ybe.min(), ybe.max()]

IV['z1'] = [0.9480, is_pos]
IV['z2'] = [1.404, is_pos]
IV['z3'] = [2.2843, is_pos]
IV['z4'] = [2.813, is_pos]

ZP = [IV['z1'][0], IV['z2'][0], IV['z3'][0], IV['z4'][0]]


# fractions for extended unbinned likelihood fit
IV['o']    = [25, is_pos]
IV['e']  = [0., is_pos]
IV['ep'] = [25, is_pos]
IV['b']  = [25, is_pos]

#Interval for Bernstein polynomials
interv_x = [xbe.min(), xbe.max()]
interv_y = [ybe.min(), ybe.max()]

      

# Bernstein polynomial for etap amplitude as a function of omega
# Bernstein polynomial for omega amplitude as a function of etaprime
# bi-linear background
             
fit_F=Model.analysis(2,8, interv_y, interv_x,xr, yr, ZP)


#List of total fit parameters
Fit_par=['cy0','cy1','cy2','x0','sig_x',\
            'cye0','cye1','cye2','x0e','sig_xe',\
            'cx0','cx1','cx2','cx3','y0','sig_y','co_A','co_sig_y',\
            'z1','z2','z3','z4']

#%%
#Prepare and execute unbinned likelihood fit

#Preparin arrays of variables we use to choose nearest neighbours of given event to fit
CosP_hx=np.cos(Phi_hx)
SinP_hx=np.sin(Phi_hx)
CosT_hx_pair=np.array([CosT_hx,CosT_hx]).T
Phi_hx_pair=np.array([CosP_hx,SinP_hx]).T
print('Here 1')

#Calculate distance between variable to be able to select close neighbours later on
dCosT_hx_pair=(SPD.pdist(CosT_hx_pair))/math.sqrt(2.)   #need to divide since we calcualte sqrt(2*dist^2) and want dist
dPhi_hx_pair=SPD.pdist(Phi_hx_pair)
print('Here 2')





#Takes few minutes
#Convert 1D array to squared form array, where diagonal elements are 0 , and each of the elements v_ij correspond to distance between i and j
# Divide by normalization 
CosT_norm=math.sqrt(4.0)
Phi_norm=math.sqrt(8.0)
dCosT_hx=SPD.squareform(dCosT_hx_pair)/CosT_norm
dPhi_hx=SPD.squareform(dPhi_hx_pair)/Phi_norm
print('Here 3')





# load final parameters from initial fit as initial parameters
with open('final_pars.pkl','rb') as pickle_file:
    m_init = pickle.load(pickle_file)
    
init_values = [m_init[k].value for k in Fit_par]
#init_values = [IV[k][0] for k in Fit_par]



#Arrays for storing q values
qf_omega = np.zeros_like(M_etap)
qf_etap = np.zeros_like(M_etap)
qf_eta = np.zeros_like(M_etap)
qf_bkg = np.zeros_like(M_etap)



#%%
#Number of nearest neighbours to chose
N_near=1500
#Number of neighbours to fit
N_fit=24000
N_fit = min(N_fit, M_etap.shape[0] )
#If want to plot fit results
do_plot=False



#Looping through events, peaking nearest neighbours and fitting
for ind, M in enumerate(M_etap[:N_fit]):
    
    print(f'Fit number {ind} out of {N_fit}')
    #Combine distances of diff. variables
    #Pick nearest neighbours by taking array of their indexes
    ind_neigh=np.argsort(np.sqrt(dCosT_hx[ind]**2+dPhi_hx[ind]**2))[:N_near]
    
    
    #Multivariate fit
    unbinned_NLL = cost.UnbinnedNLL((M_omega[ind_neigh], M_etap[ind_neigh]), fit_F.Total_logF_PDF, log = True)
    
    
    # initialize values
    Mi = Minuit(unbinned_NLL,  *init_values)
    
    # set the bounds
    for k in Fit_par:
        Mi.limits[k] = [m_init[k].lower_limit, m_init[k].upper_limit]
        #Mi.limits[k] = IV[k][1]

        
    #fix some of the values
    Mi.fixed["cye0"]= True
    Mi.fixed["cye1"]= True
    Mi.fixed["cye2"]= True
    Mi.fixed['x0'] = True
    Mi.fixed['sig_x'] = True
    Mi.fixed['x0e'] = True
    Mi.fixed['sig_xe'] = True
    Mi.fixed['y0'] = True
    Mi.fixed['sig_y'] = True
    Mi.fixed['co_sig_y'] = True

    #Execute fit
    Mi.migrad()

    
    

    #Get fit parameters
    final_pars = [Mi.params[k].value for k in Fit_par]
    
    
    # make histogram of exp.data
    h_exp = B.histo2d(M_omega[ind_neigh],M_etap[ind_neigh], bins = [h.nbins_x, h.nbins_y], \
                  range = [[xbe.min(), xbe.max()],[ybe.min(),ybe.max()]], \
                 xlabel = l_omega, ylabel = l_etap, title = '')
    #B.pl.figure(); h_exp.plot()     

    # calculate the fitted values
    xx,yy = np.meshgrid(h_exp.x_bin_center, h_exp.y_bin_center)
    S_log_pdf = fit_F.Total_logF_PDF((xx, yy), *final_pars)
    # need to integrate PDF and normalize to number of events instead 1
    S_pdf = np.exp(S_log_pdf)*h_exp.x_bin_width*h_exp.y_bin_width*M_omega[ind_neigh].shape[0]
    
    h_likefit = C.copy(h_exp)
    h_likefit.bin_content = S_pdf.T
    h_likefit.bin_error = np.sqrt(h_likefit.bin_content)
    #B.pl.figure();B.pl.close('all')
    #h_likefit.plot()


    if do_plot:
        # compare etap projection
        B.pl.figure(); 
        o_range = (0.6, 0.65)
        h_exp.project_y(range = o_range).plot_exp()
        h_likefit.project_y(range = o_range).plot(filled = False, color = 'r')
        
        B.pl.figure()
        e_range = (0.96,0.97)
        h_exp.project_x(range = e_range).plot_exp()
        h_likefit.project_x(range = e_range).plot(filled = False, color = 'r')
        
       
    # calculate q-factors
    all_weights = fit_F.Total_F_PDF((M_omega[ind], M_etap[ind]), *final_pars)
    signal_weights = np.sum(all_weights, axis = 0)
    fracs = all_weights/signal_weights 
    q_omega = fracs[0]
    q_eta = fracs[1]
    q_etap = fracs[2]
    q_bkg = fracs[3]
    
    # print results
    #print(f'fracs = {fracs}, signal_weights = {signal_weights}')
    # save the q-factors
    qf_omega[ind]= q_omega
    qf_etap[ind] = q_etap
    qf_eta[ind] = q_eta
    qf_bkg[ind] = q_bkg



#%% make 4 vectors


# eta prime
p4_etap = vector.array({'x':Px_etapr, 'y':Py_etapr, 'z':Pz_etapr, 'E':E_etapr})

# pi0
p4_pi0 = vector.array({'x':Px_pi0, 'y':Py_pi0, 'z':Pz_pi0, 'E':E_pi0})
 
            
# calculate the 4 vector of the eta' pi0 system

p4_p1 = p4_etap + p4_pi0



#Making histograms
h_omega = B.histo(p4_p1.M, range = (1.,2.), bins = 25, weights = qf_omega, title= r'$M_{\omega}$')
h_eta = B.histo(p4_p1.M, range = (1.,2.), bins = 25, weights = qf_eta, title= r'$M_{\eta}$')
h_etap = B.histo(p4_p1.M, range = (1.,2.), bins = 25, weights = qf_etap, title= r'$M_{\eta^{\prime}}$')
h_bkg = B.histo(p4_p1.M, range = (1.,2.), bins = 25, weights = qf_bkg, title= r'$M_{bkg}$')
h_all = B.histo(p4_p1.M, range = (1.,2.), bins = 25, title= r'$M_{all}$')



axis_range=[[0.4,1.5],[0.9,1.02]]   

h2_Metaprime_Momega_orig=B.histo2d(M_omega, \
                                  M_etap, bins=(100,100), range=axis_range ,\
                                      title="No weight", \
                                          xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                              ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                              )

    
h2_Metaprime_Momega_omega=B.histo2d(M_omega[:N_fit] , \
                                      M_etap[:N_fit] , bins=(100,100), range=axis_range ,\
                                          title=r"$\omega$", \
                                              xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                                  ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                                  , weights = qf_omega[:N_fit])
    
    
h2_Metaprime_Momega_eta=B.histo2d(M_omega[:N_fit] , \
                                          M_etap[:N_fit] , bins=(100,100), range=axis_range ,\
                                              title=r"$\eta$", \
                                                  xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                                      ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                                      , weights = qf_eta[:N_fit])
    
    
h2_Metaprime_Momega_etaprime=B.histo2d(M_omega[:N_fit] , \
                                              M_etap[:N_fit] , bins=(100,100), range=axis_range ,\
                                                  title=r"$\eta^{\prime}$", \
                                                      xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                                          ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                                          , weights = qf_etap[:N_fit])
    
   
h2_Metaprime_Momega_bkgd=B.histo2d(M_omega[:N_fit] , \
                                              M_etap[:N_fit] , bins=(100,100), range=axis_range ,\
                                                  title="Background", \
                                                      xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                                          ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                                          , weights = qf_bkg[:N_fit])     
     
     
    
    
B.pl.figure()
h2_Metaprime_Momega_orig.plot()       
B.pl.savefig("Plots/Metaprime_Momega_orig.pdf")   

B.pl.figure()
h2_Metaprime_Momega_bkgd.plot()       
B.pl.savefig("Plots/Metaprime_Momega_bkgd.pdf")   

B.pl.figure()
h2_Metaprime_Momega_etaprime.plot()       
B.pl.savefig("Plots/Metaprime_Momega_etaprime.pdf")   

B.pl.figure()
h2_Metaprime_Momega_eta.plot()      
B.pl.savefig("Plots/Metaprime_Momega_eta.pdf")   

B.pl.figure()
h2_Metaprime_Momega_omega.plot()       
B.pl.savefig("Plots/Metaprime_Momega_omega.pdf")   
    


B.pl.figure()
h_omega.plot()
B.pl.savefig("Plots/h_omega.pdf")

B.pl.figure()
h_eta.plot()
B.pl.savefig("Plots/h_eta.pdf")

B.pl.figure()
h_etap.plot()
B.pl.savefig("Plots/h_etap.pdf")

B.pl.figure()
h_bkg.plot()
B.pl.savefig("Plots/h_bkg.pdf")

B.pl.figure()
h_all.plot()
B.pl.savefig("Plots/h_all.pdf")





#%%
#Saving data in .root file, separated by polarization angle

# polarization selections

sel_amo = Pol == -1
sel_0 = Pol == 0
sel_90 = Pol == 90
sel_45 = Pol == 45
sel_135 = Pol == 135

sel_all = Pol > - 5



Pol_dict = {'0':sel_0, '45':sel_45, '90':sel_90, '135':sel_135, 'amo':sel_amo, 'all':sel_all}


# sel_pol is the polarization angle, can be 0, 45, 90 135
# possible vlaues sel_amo | sel_0 | sel_90 | sel_45 | sel_135

loop_pol=[key for key in Pol_dict.keys()]


#Loop through different polarization orientations and save corresponding  "kin" tree to a .root file
for polarization in loop_pol[:4]:
    


    # select the correponding values
    sel_pol = Pol_dict[polarization]
    
    # selected polarizztions
    Pol_s = Pol[sel_pol]
    
    Px_p_s = Px_p[sel_pol]
    Px_etapr_s = Px_etapr[sel_pol]
    Px_pi0_s = Px_pi0[sel_pol]
    Py_p_s = Py_p[sel_pol]
    Py_etapr_s = Py_etapr[sel_pol]
    Py_pi0_s = Py_pi0[sel_pol]
    Pz_p_s = Pz_p[sel_pol]
    Pz_etapr_s = Pz_etapr[sel_pol]
    Pz_pi0_s = Pz_pi0[sel_pol]
    
    E_p_s = E_p[sel_pol]
    E_etapr_s = E_etapr[sel_pol]
    E_pi0_s = E_pi0[sel_pol]
    
    Px_beam_s = Px_beam[sel_pol]
    Py_beam_s = Py_beam[sel_pol]
    Pz_beam_s = Pz_beam[sel_pol]
    E_beam_s = E_beam[sel_pol]
    
    # selected q_factors
    QF_etap_s = qf_etap[sel_pol]
    





    
    # make output dictionary
    
    d_pwa = {}
    
    d_pwa['px_pr'] = Px_p_s
    d_pwa['px_etapr'] = Px_etapr_s
    d_pwa['px_pi0'] = Px_pi0_s
    
    
    d_pwa['py_pr'] = Py_p_s
    d_pwa['py_etapr'] = Py_etapr_s
    d_pwa['py_pi0'] = Py_pi0_s
    
    
    d_pwa['pz_pr'] = Pz_p_s
    d_pwa['pz_etapr'] = Pz_etapr_s
    d_pwa['pz_pi0'] = Pz_pi0_s
    
    d_pwa['e_pr'] = E_p_s
    d_pwa['e_etapr'] = E_etapr_s
    d_pwa['e_pi0'] = E_pi0_s
    
    d_pwa['px_beam'] = Px_beam_s
    d_pwa['py_beam'] = Py_beam_s
    d_pwa['pz_beam'] = Pz_beam_s
    d_pwa['e_beam'] = E_beam_s
    
    d_pwa['pol'] = Pol_s
    d_pwa['qf'] = QF_etap_s
    
    
    
    
    
    
    save_root_tree=FRW.Write_kin_tree(d_pwa,tree_name="kin",out_file="data/pwa/kintree_"+polarization+".root")
    save_root_tree.save_kindict_toroot()
 







