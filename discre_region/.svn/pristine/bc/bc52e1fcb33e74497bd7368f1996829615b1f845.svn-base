__author__ = 'zeboli1'

import numpy as np
import lattice_info as LI
from scipy import linalg as LA

'''
This part is only suitable for 2D projected FCC lattice, and vacancy mediated single vacancy diffusion
It is designed specifically for our system and it can not be transfered to other system
'''
class jump_matrix_info:
    #including the neighboring relationship, energy calculation and jump_matrix setup
    def __init__(self,site_coord,Burg,Boltz_KT):
        self.sites = np.array(site_coord)
        self.num_sites = len(site_coord)
        #magnitude of the burger's vector and boltzmann factor
        self.Burg = Burg
        self.kBT = Boltz_KT
        #neighboring relationship
        self.nei_matrix = np.zeros((self.num_sites,self.num_sites),dtype=np.int)
        self.nei_list = [[] for i in xrange(self.num_sites)]
        #pre_computed volumetric strain values
        self.vol_strain = np.zeros((self.num_sites))
        #solute and vacancy site energies and two types of jump frequency matrix
        #In the jump frequency matrix, M[i][j] denote the jump frequency of vacancy moving from i to j
        self.vaca_site_energy = np.zeros((self.num_sites))
        self.solu_site_energy = np.zeros((self.num_sites))
        self.jump_selfdiff = np.zeros((self.num_sites,self.num_sites))
        self.jump_exchange = np.zeros((self.num_sites,self.num_sites))
        #transition matrix that satisfies: dC/dt = M dot C (The second index is the initial sites)
        self.tran_matrix_vacancy = np.zeros((self.num_sites,self.num_sites))
        self.tran_matrix_solute = np.zeros((self.num_sites,self.num_sites))

    #Judge 2D-fcc nearest neighboring type
    def judge_neighbor_2Dfcc(self,position,x_space,y_space):
        if((abs(position[0]) < 3./2.*x_space) & (abs(position[1]) < 1./2.*y_space)):
            return 2
        elif((abs(position[0]) < 3./2.*x_space) & (abs(position[1]) < 3./2.*y_space)):
            return 1
        elif((abs(position[0]) < 5./2.*x_space) & (abs(position[1]) < 1./2.*y_space)):
            return 1
        else:
            return 0

    #setup 2D-fcc lattice neighboring matrix
    def setup_neighboring_2Dfcc(self,x_space,yspace,site_list = []):
        if(site_list == []):
            site_list = self.sites
        n_sites = min(len(site_list),self.num_sites)
        for i in range(n_sites):
            for j in range(i+1,n_sites):
                self.nei_matrix[i][j] = self.nei_matrix[j][i] = self.judge_neighbor_2Dfcc(site_list[i]-site_list[j],x_space,yspace)
                if(self.nei_matrix[i][j]>=1):
                    self.nei_list[i].append(j)
                    self.nei_list[j].append(i)

    #formula to compute volumetric strain for 2Dfcc
    def volmetric_strain_2Dfcc(self,position,Burg = -1.0,r_min = -1.0):
        if(Burg <= 0.):
            Burg = self.Burg
        if(r_min <= 0.):
            r_min = Burg/100.
        r = LA.norm(position)
        Cta = np.arctan2(position[1],position[0])
        if(r <= r_min):
            r = r_min
        return -Burg*np.sin(Cta)/4./r/np.pi

    #compute the volumetric strain for all the 2D fcc lattice sites
    def setup_volstrain_2Dfcc(self,site_list = []):
        if(site_list == []):
            site_list = self.sites
        n_site = min(len(site_list),self.num_sites)
        for i in range(n_site):
            self.vol_strain[i] = self.volmetric_strain_2Dfcc(site_list[i])

    #compute the vacancy or solute energy at a certain site (the site-energy + the interaction from neighbors# )
    #here we only consider about the vacancy-solute interaction
    def compute_site_energy(self,ind,alpha,inter_act,inter_conc):
        self_ener = alpha*self.vol_strain[ind]
        inter_ener = 0.0
        for j in self.nei_list[ind]:
            inter_ener += inter_act*inter_conc[j]*self.nei_matrix[ind][j]

        ener = self_ener + inter_ener
        return ener

    #setup energy matrix:
    def compute_energy_forall(self,alpha_V,alpha_S,inter_act,vaca_conc,solu_conc):
        for i in range(self.num_sites):
            self.vaca_site_energy[i] = self.compute_site_energy(i,alpha_V,inter_act,solu_conc)
            self.solu_site_energy[i] = self.compute_site_energy(i,alpha_S,inter_act,vaca_conc)

    #compute the jump frequencies:
    def compute_jump_frequencies(self,init_ener,fina_ener,barr_heig,attm_freq):
        sadd_ener = max(init_ener,fina_ener,(fina_ener+init_ener)/2.+barr_heig)
        barrier = sadd_ener - init_ener
        return attm_freq*np.exp(-barrier/self.kBT)

    #compute jump frequency matrix for all the sites
    def compute_freq_forall(self,barr_heig_list,attm_freq):
        for i in range(self.num_sites):
            for j in self.nei_list[i]:
                ener_i = self.vaca_site_energy[i]
                ener_j = self.vaca_site_energy[j]
                self.jump_selfdiff[i][j] = self.compute_jump_frequencies(ener_i,ener_j,barr_heig_list[0],attm_freq)*self.nei_matrix[i][j]
                ener_i = self.vaca_site_energy[i] + self.solu_site_energy[j]
                ener_j = self.vaca_site_energy[j] + self.solu_site_energy[i]
                self.jump_exchange[i][j] = self.compute_jump_frequencies(ener_i,ener_j,barr_heig_list[1],attm_freq)*self.nei_matrix[i][j]

    #compute transition matrices:
    def setup_tran_matrix(self,vaca_conc,solu_conc):
        for i in range(self.num_sites):
            self.tran_matrix_vacancy[i][i] = 0.
            self.tran_matrix_solute[i][i] = 0.
            for j in range(self.num_sites):
                if( self.nei_matrix[i][j] == 0):
                    continue
                self.tran_matrix_vacancy[j][i] = self.jump_selfdiff[i][j]*(1. - vaca_conc[j] - solu_conc[j]) + self.jump_exchange[i][j]*solu_conc[j]
                self.tran_matrix_vacancy[i][i] += -self.tran_matrix_vacancy[j][i]
                self.tran_matrix_solute[j][i] = self.jump_exchange[j][i]*vaca_conc[j]
                self.tran_matrix_solute[i][i] += -self.tran_matrix_solute[j][i]



if __name__ == "__main__":
    print 1,2,3







