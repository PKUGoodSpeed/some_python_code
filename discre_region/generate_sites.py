__author__ = 'zeboli1'

import numpy as np
import lattice_info as li
from scipy import linalg as LA

class lattice_2Dsites:
    """creating lattice sites"""

    def __init__(self,latt_const=li.a,type=1,x_vec=li.b_vec,y_vec=li.n_vec,error = 1.E-4):
        self.perfect = []
        self.n_perfect = 0
        self.distort = []
        self.n_distort = 0
        structure = li.latt_struct(type=type)
        structure.create_twoD_neighbors(x_vec,y_vec)
        self.n_neib = structure.num_twoDnb
        self.neib_list = np.array(structure.twoD_neighbor)*latt_const
        x_min = latt_const
        y_min = latt_const
        for i in range(self.n_neib):
            if(abs(self.neib_list[i][0])<x_min and abs(self.neib_list[i][0])>error):
                x_min = abs(self.neib_list[i][0])
            if(abs(self.neib_list[i][1])<y_min and abs(self.neib_list[i][1])>error):
                y_min = abs(self.neib_list[i][1])
        self.x_displ = x_min
        self.y_displ = y_min

    #check whether a point has been added or not
    def contain(self,point,list=[],error=1.E-8):
        if(list == []):
            list = self.perfect
        n = len(list)
        point = np.array(point)
        for i in range(n):
            if( np.dot(list[i]-point,list[i]-point) < error ):
                return True
        return False


    #adding two-D perfect sites
    def create_perfect(self,point,radius,error=1.E-8):
        point = np.array(point)
        if(self.contain(point,error=error) == True or np.dot(point,point)>radius**2):
            return
        self.perfect.append(point)
        self.n_perfect += 1
        #after adding a certain site, adding its neighboring sites
        for i in range(self.n_neib):
            new = point + np.array(self.neib_list[i])
            self.create_perfect(new,radius=radius,error=error)
        return


    '''
    The following are formula to compute x,y_displacments around an edge dislocation.
    Derived from the classical elasticity theory
    '''
    def x_displacment(self,Cta,r,Burg,poi_ratio):
        #x_displacement near an edge dislocation
        ux = (Burg/2./np.pi)*(Cta+(np.cos(Cta)*np.sin(Cta))/2./(1.-poi_ratio))
        return(ux)

    def y_displacment(self,Cta,r,Burg,poi_ratio):
        #y_displacement near an edge dislocation
        uy = (Burg/2./np.pi)*((2.*poi_ratio-1.)*np.log(r/Burg)+(np.sin(Cta))**2)/2./(1.-poi_ratio)
        return(uy)

    def distorting(self,site,Burg,poi_ratio,error=1.E-8):
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


    ##generating 2D distorted sites from perfect site
    def gene_2D_distorted(self,radius,Burg,poi_ratio,error = 1.E-8):
        #generate distorted 2D lattice sites
        for i in range(self.n_perfect):
            distor_site = self.distorting(self.perfect[i],Burg=Burg,poi_ratio=poi_ratio,error=error)
            if((self.contain(distor_site,list=self.distort,error=error)==False) & (LA.norm(distor_site)<radius)):
                self.distort.append(distor_site)
                self.n_distort += 1

    def sorting_sites(self,site_list = []):
        #sort the distorted 2D-sites by distance for the origin
        if(site_list == []):
            site_list = self.distort
        def getKey(site):
            return np.dot(site,site)
        self.distort = sorted(site_list,key=getKey)

    def symmetrize_sites(self,site_list=[],error = 1.E-6):
        #symmetrize sites about y-axies
        if(site_list == []):
            site_list = self.distort
        n_sites = len(site_list)
        for i in range(1,n_sites):
            if((abs(site_list[i][0] + site_list[i-1][0]) <= error) & (abs(site_list[i][1] - site_list[i-1][1]) <= error)):
                x = 1./2.*(site_list[i][0] - site_list[i-1][0])
                y = 1./2.*(site_list[i][1] + site_list[i-1][1])
                site_list[i] = np.array([x,y])
                site_list[i-1] = np.array([-x,y])



if __name__ == '__main__':
    A = lattice_2Dsites()
    center = np.array([0.,li.a/2./np.sqrt(3)])
    print A.n_neib
    print A.neib_list
    print A.x_displ
    print A.y_displ
    A.create_perfect(center,li.R_in*1.2)
    print A.n_perfect
    print A.perfect
    A.gene_2D_distorted(li.R_in,li.burg,li.nu_Poisson)
    A.sorting_sites()
    A.symmetrize_sites()
    print A.n_distort
    print A.distort