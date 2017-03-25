__author__ = 'zeboli1'
import unittest
import numpy as np
from scipy import linalg as LA
import jump_matrix as JM
import generating_lattice as GL
import lattice_info as LI
import Parameter as PR


"""
*****************
Creating testing case for the jump matrix calculation process
"""

class jump_matrix_test(unittest.TestCase):
    """Set of tests to check class behavior"""


    def setUp(self):
        """Ni fcc lattice"""
        self.lattice = GL.Projected_sites(LI.t_vec,LI.b_vec,LI.n_vec,LI.a,LI.burg,LI.nu_Poisson)
        self.lattice.gene_3D_perfect(-8,8,-8,8,-2,2)
        center = np.array([0.,self.lattice.lattice_parameter/2./np.sqrt(3)])
        self.lattice.gene_2D_prefect(LI.R_buff,center)
        self.lattice.gene_2D_distorted(LI.R_in)
        self.lattice.sorting_sites()
        self.lattice.symmetrize_sites()
        self.jump_matrix = JM.jump_matrix_info(self.lattice.distorted_2D,self.lattice.burg_mag,PR.KB*PR.Temp)

    #test the class is generated is correct
    def test_initilization(self):
        n_site = 56
        self.assertEqual(self.jump_matrix.num_sites, n_site)
        self.assertEqual(np.shape(self.jump_matrix.sites), (n_site,2))
        self.assertEqual(np.shape(self.jump_matrix.nei_matrix), (n_site,n_site))
        self.assertEqual(np.shape(self.jump_matrix.vol_strain), (n_site,))
        self.assertEqual(np.shape(self.jump_matrix.vaca_site_energy), (n_site,))
        self.assertEqual(np.shape(self.jump_matrix.solu_site_energy), (n_site,))
        self.assertEqual(np.shape(self.jump_matrix.jump_exchange), (n_site,n_site))
        self.assertEqual(np.shape(self.jump_matrix.jump_selfdiff), (n_site,n_site))
        self.assertEqual(np.shape(self.jump_matrix.tran_matrix_solute), (n_site,n_site))
        self.assertEqual(np.shape(self.jump_matrix.tran_matrix_vacancy), (n_site,n_site))



    #check the neighbor is generated correctly
    #here we just check the 6 central sites around the dislocation core
    """
    (4)      (3)    (2)      (5)

                (core)

           (0)         (1)
    """
    def test_neighbor_judging(self):
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[0]-self.jump_matrix.sites[1],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          2)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[2]-self.jump_matrix.sites[3],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          2)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[0]-self.jump_matrix.sites[3],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[1]-self.jump_matrix.sites[3],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[2]-self.jump_matrix.sites[0],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[2]-self.jump_matrix.sites[1],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[2]-self.jump_matrix.sites[4],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[3]-self.jump_matrix.sites[5],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[4]-self.jump_matrix.sites[0],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[5]-self.jump_matrix.sites[1],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          1)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[4]-self.jump_matrix.sites[5],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          0)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[4]-self.jump_matrix.sites[1],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          0)
        self.assertEqual(self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[5]-self.jump_matrix.sites[0],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)),
                          0)


    #Make sure the neighboring relationship is set correctly
    def test_neighboring_setup(self):
        self.jump_matrix.setup_neighboring_2Dfcc(self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3))
        for i in range(self.jump_matrix.num_sites):
            #print i,self.jump_matrix.nei_list[i]
            for j in range(self.jump_matrix.num_sites):
                if( i == j ):
                    self.assertEqual(self.jump_matrix.nei_matrix[i][j], 0)
                    self.assertFalse(i in self.jump_matrix.nei_list[i])
                else:
                    self.assertEqual(self.jump_matrix.nei_matrix[i][j], self.jump_matrix.judge_neighbor_2Dfcc(self.jump_matrix.sites[i]-self.jump_matrix.sites[j],
                                                                self.lattice.lattice_parameter/2./np.sqrt(2),self.lattice.lattice_parameter/np.sqrt(3)))
                    if(self.jump_matrix.nei_matrix[i][j] == 0):
                        self.assertFalse(i in self.jump_matrix.nei_list[j])
                        self.assertFalse(j in self.jump_matrix.nei_list[i])
                    else:
                        self.assertTrue(i in self.jump_matrix.nei_list[j])
                        self.assertTrue(j in self.jump_matrix.nei_list[i])




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(jump_matrix_test)
    unittest.TextTestRunner(verbosity=2).run(suite)