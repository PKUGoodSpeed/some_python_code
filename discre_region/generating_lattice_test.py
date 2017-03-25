__author__ = 'zeboli1'

import unittest
import numpy as np
from scipy import linalg as LA
import generating_lattice as GL
import lattice_info as LI


"""
*****************
Creating testing case for the lattice generating process
"""

def min_dis(ind,sites,max_Length = 100.):
    d_min = max_Length
    for i in range(len(sites)):
        if (i==ind):
            continue
        d_temp = LA.norm(sites[ind] - sites[i])
        if(d_temp < d_min):
            d_min = d_temp
    return d_min


class lettice_generator_test(unittest.TestCase):
    """Set of tests to check class behavior"""


    def setUp(self):
        """Ni fcc lattice"""
        self.lattice = GL.Projected_sites(LI.t_vec,LI.b_vec,LI.n_vec,LI.a,LI.burg,LI.nu_Poisson)

    #test the 3D perfect generation is correct
    def test_3D_perfect_lattice(self):
        N_x = 2*np.random.random_integers(1,5)
        N_y = 2*np.random.random_integers(1,5)
        N_z = 2*np.random.random_integers(1,5)
        n_site = 4*N_x*N_y*N_z
        self.lattice.gene_3D_perfect(-N_x/2,N_x/2,-N_y/2,N_y/2,-N_z/2,N_z/2)
        self.assertEqual(np.shape(self.lattice.coord_3D), (n_site,3))
        for i in range(n_site):
            self.assertAlmostEqual(min_dis(i,self.lattice.coord_3D),self.lattice.lattice_parameter/np.sqrt(2))

    #test the judging existance function is running correctly
    def test_judging_existance(self):
        N_x = 2*np.random.random_integers(1,5)
        N_y = 2*np.random.random_integers(1,5)
        N_z = 2*np.random.random_integers(1,5)
        n_site = 4*N_x*N_y*N_z
        self.lattice.gene_3D_perfect(-N_x/2,N_x/2,-N_y/2,N_y/2,-N_z/2,N_z/2)
        N_cases = 17
        for k in range(N_cases):
            x = np.random.random_integers(-N_x,N_x)
            y = np.random.random_integers(-N_y,N_y)
            z = np.random.random_integers(-N_z,N_z)
            ind = np.random.random_integers(0,self.lattice.latt_info.num_basis-1)
            site = (np.array([x,y,z]) + self.lattice.latt_info.basis[ind])*self.lattice.lattice_parameter
            if((x>=-N_x/2) & (x<N_x/2) & (y>=-N_y/2) & (y<N_y/2) & (z>=-N_z/2) & (z<N_z/2)):
                self.assertTrue(self.lattice.judge_exist(site,self.lattice.coord_3D))
            else:
                self.assertFalse(self.lattice.judge_exist(site,self.lattice.coord_3D))

    #test the 2D-perfect lattice is generated correctly
    def test_2D_perfect_lattice(self):
        N_x = 2*np.random.random_integers(2,5)
        N_y = 2*np.random.random_integers(2,5)
        N_z = 2*np.random.random_integers(2,5)
        n_site = 4*N_x*N_y*N_z
        self.lattice.gene_3D_perfect(-N_x/2,N_x/2,-N_y/2,N_y/2,-N_z/2,N_z/2)
        center = np.array([0.,self.lattice.lattice_parameter/2./np.sqrt(3)])
        self.lattice.gene_2D_prefect(LI.R_buff,center)
        point = -center
        self.assertTrue(self.lattice.judge_exist(point))
        point = -center + np.array([0.,self.lattice.lattice_parameter/np.sqrt(3)])
        self.assertTrue(self.lattice.judge_exist(point))
        point = -center + np.array([0.,-self.lattice.lattice_parameter/np.sqrt(3)])
        self.assertTrue(self.lattice.judge_exist(point))
        point = -center + np.array([self.lattice.lattice_parameter/2./np.sqrt(2),0.])
        self.assertTrue(self.lattice.judge_exist(point))
        point = -center + np.array([-self.lattice.lattice_parameter/2./np.sqrt(2),0.])
        self.assertTrue(self.lattice.judge_exist(point))
        for i in range(len(self.lattice.coord_2D)):
            self.assertTrue(LA.norm(self.lattice.coord_2D[i]) <= LI.R_buff)

    #test the 2D distorted lattice is generated correctly (compare with the C++ code results)
    #The site coordinates generated from C++ code are sorted and stored in test_sites_2D.txt
    def test_2D_distorted_lattice(self):
        fp = open('test_sites_2D.txt','r')
        Cd_in = []
        for line in fp:
            Cd_in.append(np.array(map(float,line.split(' '))))
        fp.close()
        self.lattice.gene_3D_perfect(-8,8,-8,8,-2,2)
        center = np.array([0.,self.lattice.lattice_parameter/2./np.sqrt(3)])
        self.lattice.gene_2D_prefect(LI.R_buff,center)
        self.lattice.gene_2D_distorted(LI.R_in)
        self.lattice.sorting_sites()
        self.lattice.symmetrize_sites()
        np.testing.assert_array_almost_equal(Cd_in,self.lattice.distorted_2D)














if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(lettice_generator_test)
    unittest.TextTestRunner(verbosity=2).run(suite)