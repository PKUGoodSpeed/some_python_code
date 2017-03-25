__author__ = 'zeboli1'

import unittest
import numpy as np
import linear_interp as LIP
import matplotlib.pyplot as plt




b = 2.5
kT = 0.08617
alpha = 6.37
radi = 6.67
n_mesh = 50

"""
*****************
Creating testing case for the lattice generating process
"""

def test_func1(x,y):
    return 1.0*x+2.0*y

def test_func2(x,y):
    return -b*y/4./np.pi/(x**2+y**2)

def test_func3(x,y):
    return np.exp(-alpha*test_func2(x,y)/kT)

class rbf_interpolation_test(unittest.TestCase):
    """Set of tests to check class behavior"""
    def setUp(self):
        """Ni fcc lattice"""
        #input sites coordinates from an input file
        #inner region
        fp = open('Inputs/Inner_sites.txt','r')
        sites = []
        for line in fp:
            position = map(float,line.split(' '))
            if(np.dot(position,position)>radi*radi/4.):
                sites.append(position)
        fp.close()
        #Buffer region
        fp = open('Inputs/Buffer_sites.txt','r')
        for line in fp:
            sites.append(map(float,line.split(' ')))
        fp.close()
        self.sites = np.array(sites)
        self.n_site = len(sites)
        #testing functions
        self.function_list = []
        self.function_list.append(test_func1)
        self.function_list.append(test_func2)
        self.function_list.append(test_func3)
        self.label = ["test#1","test#2","test#3"]
        #setup linear interpolation scheme
        self.strategy = LIP.linear_inter_scheme(self.sites)
        self.strategy.create_1Dmesh(radi,n_grid=n_mesh)
        self.types = ['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate']
        self.show = ['mul','inv','gau','lin','cub','qui','thi']
        self.epsilon = 1.5
        #create boundary mesh
        self.bsites = self.strategy.grid1D
        self.n_bsites = self.strategy.n_grid1D

    def test_relative_error(self):
        #output the scatter plots of the relative errors
        n_case = 3
        n_tp = len(self.types)
        for case in range(n_case):
            values = self.function_list[case](self.sites[:,0],self.sites[:,1])
            analy_bvalues = self.function_list[case](self.bsites[:,0],self.bsites[:,1])
            x_list = []
            for i in range(n_tp):
                x_list.append(i+0.5)
            y_list = []
            for tp in range(n_tp):
                self.strategy.gene_rbf_func(values,type = self.types[tp], epsilon = self.epsilon)
                self.strategy.compute_boundary()
                interp_bvalues = self.strategy.bound_value
                rela_error = np.dot(analy_bvalues-interp_bvalues,analy_bvalues-interp_bvalues)/np.dot(analy_bvalues,analy_bvalues)
                y_list.append(rela_error)
            y_range = 1.2*max(y_list)
            plt.scatter(x_list, y_list,s=100, alpha=0.3)
            plt.title(self.label[case],rotation=0,family = 'serif',size = 24)
            plt.xticks(x_list,self.show,rotation=45,family = 'serif',size = 17)
            plt.ylabel("Relative errors",rotation='vertical',family = 'serif',size = 24)
            plt.ylim((0.,y_range))
            plt.show()

    def test_perturbation_responses(self):
        case = 1
        delta = 1E-6
        n_tp = len(self.types)
        values = self.function_list[case](self.sites[:,0],self.sites[:,1])
        x_lable = []
        for i in range(13):
            x_lable.append(1.0*i)
        for k in range(n_tp):
            self.strategy.gene_rbf_func(values,type = self.types[k], epsilon = self.epsilon)
            self.strategy.compute_boundary()
            ini_bvalues = self.strategy.bound_value
            x_list = []
            y_list = []
            for i in range(self.n_site):
                values[i] += delta
                self.strategy.gene_rbf_func(values,type = self.types[k], epsilon = self.epsilon)
                self.strategy.compute_boundary()
                fin_bvalues = self.strategy.bound_value
                values[i] -= delta
                for j in range(self.n_bsites):
                    dist = np.sqrt(np.dot(self.sites[i]-self.bsites[j],self.sites[i]-self.bsites[j]))
                    slope = (fin_bvalues[j]-ini_bvalues[j])/delta
                    x_list.append(dist)
                    y_list.append(slope)
            plt.scatter(x_list, y_list,s=17, alpha=0.5)
            plt.title(self.types[k],rotation=0,family = 'serif',size = 24)
            plt.ylabel("responses to a perturbation",rotation='vertical',family = 'serif',size = 24)
            plt.yticks(family = 'serif',size = 17)
            plt.xlabel("distance",rotation=0,family = 'serif',size = 24)
            plt.xticks(family = 'serif',size = 17)
            plt.xlim(0.,17.)
            plt.show()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(rbf_interpolation_test)
    unittest.TextTestRunner(verbosity=2).run(suite)
