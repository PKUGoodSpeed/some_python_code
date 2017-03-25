__author__ = 'zeboli1'

import numpy as np
import scipy
from scipy import linalg as LA
from scipy.spatial import Voronoi


'''
*****************
Generating 2D voronoi information
'''
class voro_info_2D:
    def __init__(self,sites):
        self.sites = np.array(sites)
        self.num_sites = len(sites)
        self.vor = Voronoi(sites)
        self.nei_list = [[] for i in xrange(self.num_sites)]
        self.ver_list = [[] for i in xrange(self.num_sites)]
        self.ridge_length = np.zeros((self.num_sites,self.num_sites))
        self.cell = np.zeros((self.num_sites))

    #checking whether two sites are neighbors or not
    def judge_neighbor(self,i,j,edges=[]):
        if(edges == []):
            edges = self.vor.ridge_points
        n_edges = len(edges)
        for k in range(n_edges):
            if(np.array_equal([i,j],edges[k]) or np.array_equal([j,i],edges[k])):
                return True
        return False

    #creating neighbor list
    def creating_neighbor(self, edges = []):
        n_sites = self.num_sites
        if(edges == []):
            edges = self.vor.ridge_points
        n_edges = len(edges)
        for k in range(n_edges):
            if((edges[k][0]<n_sites) & (edges[k][1]<n_sites)):
                self.nei_list[edges[k][0]].append(edges[k][1])
                self.nei_list[edges[k][1]].append(edges[k][0])

    #determine whether a certain point belongs to a certain cell
    def judge_attach(self,i,point,site_list = [],error=1E-8):
        if(site_list == []):
            site_list = self.sites
        n_sites = len(site_list)
        distance0_sq = np.dot(site_list[i]-point,site_list[i]-point)
        for j in range(n_sites):
            if(j == i):
                continue
            distance1_sq = np.dot(site_list[j]-point,site_list[j]-point)
            if(distance1_sq < distance0_sq - error):
                return False
        return True

    #construct the vertices attachment list
    def attach_vertices(self,vert_list = [],error = 1E-8):
        if(vert_list == []):
            vert_list = self.vor.vertices
        n_vertices = len(vert_list)
        for i in range(self.num_sites):
            for k in range(n_vertices):
                if(self.judge_attach(i,vert_list[k],error=error)):
                    self.ver_list[i].append(k)

    #compute the ridge length
    def compute_ridge(self,i,j,vert_list = [],default_length = 0.):
        if(j not in self.nei_list[i]):
            return 0.0
        if(vert_list == []):
            vert_list = self.vor.vertices
        n_verices = len(vert_list)
        common = []
        for k in self.ver_list[i]:
            if((k<n_verices) & (k in self.ver_list[j])):
                common.append(k)
        if(len(common)<=1):
            return default_length
        elif(len(common)>=3):
            return default_length
        else:
            return LA.norm(vert_list[common[0]]-vert_list[common[1]])

    #construct the ridges
    def ridge_network(self,vert_list = [],default_length = 0.):
        for i in range(self.num_sites):
            for j in range(i+1,self.num_sites):
                self.ridge_length[i][j] = self.ridge_length[j][i] = self.compute_ridge(i,j,vert_list,default_length)

    #compute cell area
    def compute_cells(self):
        for i in range(self.num_sites):
            self.cell[i] = 0.
            for j in self.nei_list[i]:
                distance = LA.norm(self.sites[i]-self.sites[j])
                self.cell[i] += 1./4.*distance*self.ridge_length[i][j]



class voro_info_3D:
    def __init__(self,sites):
        self.sites = np.array(sites)
        self.num_sites = len(sites)
        self.vor = Voronoi(sites)
        self.nei_list = [[] for i in xrange(self.num_sites)]
        self.ver_list = [[] for i in xrange(self.num_sites)]
        self.ridge_area = np.zeros((self.num_sites,self.num_sites))
        self.cell = np.zeros((self.num_sites))

    #creating neighbor list
    #scipy.spacial.voronoi do not show all the neighbors for 3D cases
    def creating_neighbor(self,site_list = [],error = 1E-8):
        if(site_list == []):
            site_list = self.sites
        N_sites = len(site_list)
        for i in range(N_sites):
            for j in range(i+1,N_sites):
                midpoint = 0.5*(site_list[i] + site_list[j])
                #computing neighbor relationship
                switch = 1
                distancesq_i = np.dot(site_list[i] - midpoint,site_list[i] - midpoint)
                for k in range(N_sites):
                    distancesq_k = np.dot(site_list[k]-midpoint,site_list[k]-midpoint)
                    if( (k!=i) & (k!=j) & (distancesq_k < distancesq_i + error)):
                        switch = 0
                        break
                if(switch==1):
                    self.nei_list[i].append(j)
                    self.nei_list[j].append(i)

    #determine whether a certain point belongs to a certain cell
    def judge_attach(self,i,point,site_list = [],error=1E-8):
        if(site_list == []):
            site_list = self.sites
        n_sites = len(site_list)
        distance0_sq = np.dot(site_list[i]-point,site_list[i]-point)
        for j in range(n_sites):
            if(j == i):
                continue
            distance1_sq = np.dot(site_list[j]-point,site_list[j]-point)
            if(distance1_sq < distance0_sq - error):
                return False
        return True

    #construct the vertices attachment list
    def attach_vertices(self,vert_list = [],error = 1E-8):
        if(vert_list == []):
            vert_list = self.vor.vertices
        n_vertices = len(vert_list)
        for i in range(self.num_sites):
            for k in range(n_vertices):
                if(self.judge_attach(i,vert_list[k],error=error)):
                    self.ver_list[i].append(k)

    #compute perpendicular basis
    def get_horizonBasis(self,vec,threshold=1./np.sqrt(3)):
        vec_0 = vec/LA.norm(vec)
        if(abs(vec_0[0])>threshold):
            vec_1 = np.array([-vec_0[1],vec_0[0],0.0])
        else:
            vec_1 = np.array([0.0,-vec_0[2],vec_0[1]])
        x_base = vec_1/LA.norm(vec_1)
        y_base = np.cross(x_base,vec_0)
        return x_base,y_base

    #compute angle in projected 2D Polar coordinate system
    def angle(self,position,origin,x_basis,y_basis):
        x = np.dot(position-origin,x_basis)
        y = np.dot(position-origin,y_basis)
        return np.arctan2(y,x)

    #compute the polygon areas
    #We need the origin be located within the polygon. All the vertices as well as origin are in the same plane
    #The above conditions are automatically satisfied for the voronoi case
    def poly_area(self,ind_list,origin,x_basis,y_basis,vert_list = [],default_area = 0.):
        n_ind = len(ind_list)
        if(n_ind <= 2):
            return default_area
        if(vert_list == []):
            vert_list = self.vor.vertices
        def getKey(ind):
            return self.angle(vert_list[ind],origin,x_basis,y_basis)
        srted_ind = sorted(ind_list,key=getKey)
        area = 0
        for i in range(n_ind-1):
            area += 1./2.*LA.norm(np.cross(vert_list[srted_ind[i]]-origin,vert_list[srted_ind[i+1]]-origin))
        area += 1./2.*LA.norm(np.cross(vert_list[srted_ind[n_ind-1]]-origin,vert_list[srted_ind[0]]-origin))
        return abs(area)


    #compute the ridge area
    def compute_ridgearea(self,i,j,vert_list = [],default_area = 0.):
        if(j not in self.nei_list[i]):
            return 0.0
        if(vert_list == []):
            vert_list = self.vor.vertices
        n_verices = len(vert_list)
        common = []
        for k in self.ver_list[i]:
            if((k<n_verices) & (k in self.ver_list[j])):
                common.append(k)
        basis = self.get_horizonBasis(self.sites[j]-self.sites[i])
        origin = 1./2.*(self.sites[i]+self.sites[j])
        return self.poly_area(common,origin,basis[0],basis[1],default_area=default_area)

    #construct the ridges
    def ridge_network(self,vert_list = [],default_area = 0.):
        for i in range(self.num_sites):
            for j in range(i+1,self.num_sites):
                self.ridge_area[i][j] = self.ridge_area[j][i] = self.compute_ridgearea(i,j,vert_list,default_area)

    #compute cell area
    def compute_cells(self):
        for i in range(self.num_sites):
            self.cell[i] = 0.
            for j in self.nei_list[i]:
                distance = LA.norm(self.sites[i]-self.sites[j])
                self.cell[i] += 1./6.*distance*self.ridge_area[i][j]






