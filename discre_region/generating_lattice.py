__author__ = 'zeboli1'

import numpy as np
import lattice_info as LI
from scipy import linalg as LA

class Projected_sites:
    #setup sites for the projected 2D fcc lattices

    def __init__(self,threading,burger,normal,latt_para,Burg,poi_ratio,struct = 1):
        #we will constructed 3D and 2D sites for the system in the dislocation coordinate system
        #struct = 1: for fcc
        self.latt_info = LI.latt_struct(struct)
        self.lattice_parameter = latt_para
        #dislocation information
        self.threading = np.array(threading)/LA.norm(threading)
        self.burger = np.array(burger)/LA.norm(burger)
        self.normal = np.array(normal)/LA.norm(normal)
        self.burg_mag = Burg
        self.poisson_ratio = poi_ratio
        #sites slots
        self.coord_3D = []
        self.distorted_3D = []
        self.coord_2D = []
        self.distorted_2D = []

    def gene_3D_perfect(self,left,right,bottom,top,back,front):
        #generate 3_D perfict lattice
        for i in range(left,right):
            for j in range(bottom,top):
                for k in range(back,front):
                    for index in range(self.latt_info.num_basis):
                        self.coord_3D.append((np.array([i,j,k]) + np.array(self.latt_info.basis[index]))*self.lattice_parameter)

    def judge_exist(self,site,site_list = [],error=1.E-8):
        #check whether a site is already in the list
        if(site_list == []):
            site_list = self.coord_2D
        n_site = len(site_list)
        for i in range(n_site):
            displace = np.array(site) - np.array(site_list[i])
            if(np.dot(displace,displace) < error):
                return True
        return False

    def gene_2D_prefect(self,Radius,center,sites = [],error=1.E-8):
        #generate 2_D projected sites in dislocation system (perfect)
        if(sites == []):
            sites = self.coord_3D
        n_site = len(sites)
        for i in range(n_site):
            x = np.dot(sites[i],self.burger) - center[0]
            y = np.dot(sites[i],self.normal) - center[1]
            site_coord = np.array([x,y])
            if((self.judge_exist(site_coord,error=error)==False) & (LA.norm(site_coord)<Radius)):
                self.coord_2D.append(site_coord)


    '''
    The following are formula to compute x,y_displacments around an edge dislocation.
    Derived from the classical elasticity theory
    '''
    def x_displacment(self,Cta,r,Burg=-1.,poi_ratio=-1.):
        #x_displacement near an edge dislocation
        if(Burg<0):
            Burg = self.burg_mag
        if(poi_ratio<0):
            poi_ratio = self.poisson_ratio
        ux = (Burg/2./np.pi)*(Cta+(np.cos(Cta)*np.sin(Cta))/2./(1.-poi_ratio))
        return(ux)

    def y_displacment(self,Cta,r,Burg=-1.,poi_ratio=-1.):
        #y_displacement near an edge dislocation
        if(Burg<0):
            Burg = self.burg_mag
        if(poi_ratio<0):
            poi_ratio = self.poisson_ratio
        uy = (Burg/2./np.pi)*((2.*poi_ratio-1.)*np.log(r/Burg)+(np.sin(Cta))**2)/2./(1.-poi_ratio)
        return(uy)

    def distorting(self,site,Burg=-1.,poi_ratio=-1.,error=1.E-8):
        #distort one site near and edge dislocation
        if(np.dot(site,site) <= error):
            return np.array([0.,0.])
        ref_site = site
        site_old = np.array([0.,0.])
        while(np.dot(site-site_old,site-site_old) > error):
            site_old = site
            r = LA.norm(site)
            Cta = np.arctan2(site[1],site[0])
            displacement = np.array([self.x_displacment(Cta,r,Burg=Burg,poi_ratio=poi_ratio),
                                     self.y_displacment(Cta,r,Burg=Burg,poi_ratio=poi_ratio)])
            site = ref_site + displacement
        return site

    def gene_3D_distorted(self,center,Radius,site_list = [],Burg=-1.,poi_ratio=-1.,error = 1.E-8):
        #generate distorted 3D lattice sites
        if(site_list == []):
            site_list = self.coord_3D
        n_site = len(site_list)
        for i in range(n_site):
            x = np.dot(site_list[i],self.burger) - center[0]
            y = np.dot(site_list[i],self.normal) - center[1]
            z = np.dot(site_list[i],self.threading)
            posi_2D = np.array([x,y])
            posi_2D = self.distorting(posi_2D,Burg=Burg,poi_ratio=poi_ratio,error=error)
            posi_3D = posi_2D[0]*self.burger + posi_2D[1]*self.normal + z*self.threading
            if((self.judge_exist(posi_3D,site_list=self.distorted_3D)==False) & (LA.norm(posi_2D)<Radius)):
                self.distorted_3D.append(posi_3D)


    def gene_2D_distorted(self,Radius,site_list = [],Burg=-1.,poi_ratio=-1.,error = 1.E-8):
        #generate distorted 2D lattice sites
        if(site_list == []):
            site_list = self.coord_2D
        n_site = len(site_list)
        for i in range(n_site):
            distor_coord = self.distorting(site_list[i],Burg=Burg,poi_ratio=poi_ratio,error=error)
            if((self.judge_exist(distor_coord,site_list=self.distorted_2D)==False) & (LA.norm(distor_coord)<Radius)):
                self.distorted_2D.append(distor_coord)

    def sorting_sites(self,site_list = []):
        #sort the distorted 2D-sites
        if(site_list == []):
            site_list = self.distorted_2D
        def getKey(site):
            return np.dot(site,site)
        self.distorted_2D = sorted(site_list,key=getKey)

    def symmetrize_sites(self,site_list=[],error = 1.E-6):
        #symmetrize sites about y-axies
        if(site_list == []):
            site_list = self.distorted_2D
        n_sites = len(site_list)
        for i in range(1,n_sites):
            if((abs(site_list[i][0] + site_list[i-1][0]) <= error) & (abs(site_list[i][1] - site_list[i-1][1]) <= error)):
                x = 1./2.*(site_list[i][0] - site_list[i-1][0])
                y = 1./2.*(site_list[i][1] + site_list[i-1][1])
                site_list[i] = np.array([x,y])
                site_list[i-1] = np.array([-x,y])

    def output_data(self,fp,site_list = [],dimension = 2):
        #output 2D distorted sites coordinates
        if(site_list == []):
            site_list = self.distorted_2D
        n_site = len(site_list)
        for i in range(n_site):
            if(dimension == 2):
                fp.write("%.12f"%site_list[i][0] + ' ' + "%.12f"%site_list[i][1])
            elif(dimension == 3):
                fp.write("%.12f"%site_list[i][0] + ' ' + "%.12f"%site_list[i][1] + ' ' + "%.12f"%site_list[i][2])
            if(i != (n_site-1)):
                fp.write('\n')


if __name__ == "__main__":
    lattice = Projected_sites(LI.t_vec,LI.b_vec,LI.n_vec,LI.a,LI.burg,LI.nu_Poisson)
    lattice.gene_3D_perfect(-8,8,-8,8,-2,2)
    center = np.array([0.,LI.a/2./np.sqrt(3)])
    lattice.gene_2D_prefect(LI.R_buff,center)
    lattice.gene_2D_distorted(LI.R_in)
    print lattice.distorted_2D
    print len(lattice.distorted_2D)
    lattice.sorting_sites()
    print lattice.distorted_2D
    print len(lattice.distorted_2D)
    lattice.symmetrize_sites()
    print lattice.distorted_2D
    print len(lattice.distorted_2D)
    fp = open('Inputs/sites_coord.txt','w')
    lattice.output_data(fp)
    fp.close()

