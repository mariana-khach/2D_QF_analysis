#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:20:42 2023

@author: marianak
"""

import numpy as np
import uproot as ur








class NPZ_file:
    
    
    def __init__(self,filename):
        self.file=filename
        self.data=np.load(self.file)
        self.keylist=self.data.keys()
        
    def __getitem__(self,keyname):
        return self.data[keyname]
    
    
    
    
    
    
    
    
    
    
class Write_kin_tree:
    

    def __init__(self,dict,tree_name="kin",out_file="kintree.root"):
        self.d=dict
        self.tree=tree_name
        self.file=out_file
        

 
    def create_kin_dict(self):
        
        
        # setup kin tree px,py,pz,E array branches corresponding to proton, etaprime, pi0 
        
        Px = np.array([self.d['px_pr'], self.d['px_etapr'] , self.d['px_pi0']]).T
        Py = np.array([self.d['py_pr'], self.d['py_etapr'] , self.d['py_pi0']]).T
        Pz = np.array([self.d['pz_pr'], self.d['pz_etapr'] , self.d['pz_pi0']]).T
        E = np.array([self.d['e_pr'], self.d['e_etapr'], self.d['e_pi0']]).T
        
        
        # prepare the tree dictionary
        d_kin = {}
        
        d_kin['E_FinalState'] = E.astype(np.float32)
        d_kin['Px_FinalState'] = Px.astype(np.float32)
        d_kin['Py_FinalState'] = Py.astype(np.float32)
        d_kin['Pz_FinalState'] = Pz.astype(np.float32)
        
        d_kin['E_Beam'] = self.d['e_beam'].astype(np.float32)
        d_kin['Px_Beam'] = self.d['px_beam'].astype(np.float32)
        d_kin['Py_Beam'] = self.d['py_beam'].astype(np.float32)
        d_kin['Pz_Beam'] = self.d['pz_beam'].astype(np.float32)
        
        # number of final state particles (always 3 in this case)
        n_fin = np.repeat(3, Px.shape[0])
        d_kin['NumFinalState'] = n_fin.astype(np.int32)
        
        # polarization
        d_kin['Polarization_Angle'] = self.d['pol'].astype(np.float32) 
        
        d_kin['Weight'] = self.d['qf'].astype(np.float32)
    
        return d_kin
    
    
    
    
    def save_kindict_toroot(self):
        
        file = ur.recreate(self.file)
        file[self.tree] = self.create_kin_dict()
        file.close()
    
    
    
  

    
    
    
    
    
    
    

    
   