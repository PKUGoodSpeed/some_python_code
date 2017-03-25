__author__ = 'zeboli1'
'''
*****************
output file
Two voronoi files:
1, voronoi_information (relationship between two sites i and j)
Format:
i   j   geo_neighbor[i][j]  interface[i][j]  vector[i][j]_x  vector[i][j]_y

2, voronoi_area (voronoi area for each site)
Format:
i   area[i]
*****************
'''

import numpy as np
from scipy import linalg as LA


'''
*****************
Generating 2D voronoi information
'''
class voronoi_2D:
    def __init__(self,site_coordinates):
        self.sites = np.array(site_coordinates)
        self.num_sites = len(site_coordinates)
        self.mid_pt = np.zeros((self.num_sites,self.num_sites,2))
        self.vector = np.zeros((self.num_sites,self.num_sites,2))
        self.geo_neighbor = np.zeros((self.num_sites,self.num_sites),dtype=np.int)
        self.neigh_list = [[] for i in xrange(self.num_sites)]
        self.inter_bound = np.zeros((self.num_sites,self.num_sites))
        self.cell = np.zeros((self.num_sites))
    #compute middle points, vectors and neighboring information
    def network_struct(self):
        sites = self.sites
        N_sites = self.num_sites
        for i in range(N_sites):
            self.mid_pt[i][i] = sites[i]
            for j in range(i+1,N_sites):
                self.mid_pt[i][j] = self.mid_pt[j][i] = 0.5*(sites[i] + sites[j])
                v = sites[j] - sites[i]
                self.vector[i][j] = v/LA.norm(v)
                self.vector[j][i] = -self.vector[i][j]
                #computing neighbor relationship
                self.geo_neighbor[i][j] = 1
                distance_i = np.dot(sites[j] - self.mid_pt[i][j],sites[j] - self.mid_pt[i][j])
                for k in range(N_sites):
                    distance_k = np.dot(sites[k]-self.mid_pt[i][j],sites[k]-self.mid_pt[i][j])
                    if( (k!=i) & (k!=j) & (distance_k <= distance_i)):
                        self.geo_neighbor[i][j] = 0
                        break
                self.geo_neighbor[j][i] = self.geo_neighbor[i][j]
                if(self.geo_neighbor[i][j]==1):
                    self.neigh_list[i].append(j)
                    self.neigh_list[j].append(i)
    #This function is to compute the minimum length square from a certain point
    #to any sites from neigh_list[i] and neigh_list[j]
    def min_lengthsq(self,i,j,point,Lmax):
        rsq_min = Lmax**2
        for k in self.neigh_list[i]:
            rsq_temp = np.dot(point - self.sites[k],point - self.sites[k])
            if((k!=j)&(k!=i)&(rsq_temp<rsq_min)):
                rsq_min = rsq_temp
        for k in self.neigh_list[j]:
            rsq_temp = np.dot(point - self.sites[k],point - self.sites[k])
            if((k!=j)&(k!=i)&(rsq_temp<rsq_min)):
                rsq_min = rsq_temp
        return rsq_min
    #compute the length of the inter boundary between site i and site j
    def searching_length(self,i,j,Lmax,error,hj):
        if(self.geo_neighbor[i][j]==0):
            return 0.0
        v0 = self.mid_pt[i][j]
        v1 = self.sites[i]
        direction = np.array([self.vector[i][j][1],-self.vector[i][j][0]])
        if(hj<0):
            direction = -direction
        a = 0.0
        b = Lmax
        while((b-a)>error):
            c = 0.5*(a+b)
            v_p = v0 + c*direction
            rsq0 = np.dot(v_p-v1,v_p-v1)
            rsq1 = self.min_lengthsq(i,j,v_p,2*Lmax)
            if(rsq1>rsq0):
                a = c
            else:
                b = c
        return a
    ##setting up voronoi interface information
    def calc_interlengths(self,Lmax,error):
        for i in range(self.num_sites):
            for j in self.neigh_list[i]:
                self.inter_bound[i][j] = self.searching_length(i,j,Lmax,error,1) + self.searching_length(i,j,Lmax,error,-1)
                self.inter_bound[j][i] = self.inter_bound[i][j]
    #compute cell area
    def compute_cell(self):
        for i in range(self.num_sites):
            self.cell[i] = 0.0
            for j in self.neigh_list[i]:
                r = 0.5*np.sqrt(np.dot(self.sites[i]-self.sites[j],self.sites[i]-self.sites[j]))
                self.cell[i] += 0.5*r*self.inter_bound[i][j]
    #output data
    def output_information(self,fp1,fp2):
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                fp1.write("%d"%i + ' ' + "%d"%j + ' ' + "%d"%self.geo_neighbor[i][j] + ' ')
                fp1.write("%.12f"%self.inter_bound[i][j] + ' ')
                fp1.write("%.12f"%self.vector[i][j][0] + ' ' + "%.12f"%self.vector[i][j][1])
                if((i!=self.num_sites-1)|(j!=self.num_sites-1)):
                    fp1.write('\n')
        for i in range(self.num_sites):
            fp2.write("%d"%i + ' ' "%.12f"%self.cell[i])
            if(i!=self.num_sites-1):
                fp2.write('\n')

'''
*****************
Generating 3D voronoi information
'''
class voronoi_3D:
    def __init__(self,site_coordinates):
        self.sites = np.array(site_coordinates)
        self.num_sites = len(site_coordinates)
        self.mid_pt = np.zeros((self.num_sites,self.num_sites,3))
        self.vector = np.zeros((self.num_sites,self.num_sites,3))
        self.geo_neighbor = np.zeros((self.num_sites,self.num_sites),dtype=np.int)
        self.neigh_list = [[] for i in xrange(self.num_sites)]
        self.interface = np.zeros((self.num_sites,self.num_sites))

    #computer middle points vectors and neighbors
    def network_struct(self):
        for i in range(self.num_sites):
            self.mid_pt[i][i] = self.sites[i]
            for j in range(i+1,self.num_sites):
                self.mid_pt[i][j] = self.mid_pt[j][i] = 0.5*(self.sites[i] + self.sites[j])
                v = self.sites[j] - self.sites[i]
                self.vector[i][j] = v/np.sqrt(np.dot(v,v))
                self.vector[j][i] = -self.vector[i][j]
                #computing neighboring relationship
                self.geo_neighbor[i][j] = 1
                distance_i = np.dot(self.sites[i] - self.mid_pt[i][j],self.sites[i] - self.mid_pt[i][j])
                for k in range(self.num_sites):
                    distance_k = np.dot(self.sites[k] - self.mid_pt[i][j],self.sites[k] - self.mid_pt[i][j])
                    if( (k!=i) & (k!=j) & (distance_k <= distance_i)):
                        self.geo_neighbor[i][j] = 0
                        break
                self.geo_neighbor[j][i] = self.geo_neighbor[i][j]
                if(self.geo_neighbor[i][j]==1):
                    self.neigh_list[i].append(j)
                    self.neigh_list[j].append(i)
    #This function is to compute the minimum length square from a certain point
    #to any sites from neigh_list[i] and neigh_list[j]
    def min_lengthsq(self,i,j,point,Lmax):
        rsq_min = Lmax**2
        for k in self.neigh_list[i]:
            rsq_temp = np.dot(point - self.sites[k],point - self.sites[k])
            if((k!=j)&(k!=i)&(rsq_temp<rsq_min)):
                rsq_min = rsq_temp
        for k in self.neigh_list[j]:
            rsq_temp = np.dot(point - self.sites[k],point - self.sites[k])
            if((k!=j)&(k!=i)&(rsq_temp<rsq_min)):
                rsq_min = rsq_temp
        return rsq_min

    #compute perpendicular basis
    def get_horizonBasis(self,vec,threshold):
        vec_0 = vec/LA.norm(vec)
        if(abs(vec_0[0])>threshold):
            vec_1 = np.array([-vec_0[1],vec_0[0],0.0])
        else:
            vec_1 = np.array([0.0,-vec_0[2],vec_0[1]])
        x_base = vec_1/LA.norm(vec_1)
        y_base = np.cross(x_base,vec_0)
        return x_base,y_base
    #searching for the largest length along the "direction" from the i,j mid-point toward the interface boundary
    def searching_length(self,i,j,direction,Lmax,error):
        p_m = self.mid_pt[i][j]
        p_i = self.sites[i]
        direction = direction/LA.norm(direction)
        a = 0.0
        b = Lmax
        while((b-a)>error):
            c = (a+b)/2.
            p_f = p_m + c*direction
            r0sq = np.dot(p_f-p_i,p_f-p_i)
            r1sq = self.min_lengthsq(i,j,p_f,2.*Lmax)
            if(r1sq>r0sq):
                a = c
            else:
                b = c
        return a
    #compute the area of i and j interface
    def compute_area(self,i,j,threshold,N_cta,Lmax,error):
        if(self.geo_neighbor[i][j]==0):
            return 0.0
        basis = self.get_horizonBasis(self.vector[i][j],threshold)
        d_Cta = 2.0*np.pi/N_cta
        Cta = 0.5*d_Cta
        area = 0.0
        while(Cta<2.0*np.pi):
            direction = basis[0]*np.cos(Cta) + basis[1]*np.sin(Cta)
            r = self.searching_length(i,j,direction,Lmax,error)
            area += 0.5*d_Cta*r**2
            Cta += d_Cta
        return area
    ##setting up voronoi interface information
    def calc_interfaces(self,threshold,N_cta,Lmax,error):
        for i in range(self.num_sites):
            for j in self.neigh_list[i]:
                self.interface[i][j] = self.compute_area(i,j,threshold,N_cta,Lmax,error)
                self.interface[j][i] = self.interface[i][j]
    #output data
    def output_data(self,fp,interface):
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                fp.write("%d"%i + ' ' + "%d"%j + ' ' + "%d"%self.geo_neighbor[i][j] + ' ')
                fp.write("%.12f"%interface[i][j] + ' ')
                fp.write("%.12f"%self.vector[i][j][0] + ' ' + "%.12f"%self.vector[i][j][1] + ' ' + "%.12f"%self.vector[i][j][2])
                if((i!=self.num_sites-1)|(j!=self.num_sites-1)):
                    fp.write('\n')

