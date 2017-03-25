__author__ = 'zeboli1'
'''
test for 2D and 3D Voronoi diagram
'''

import unittest
import numpy as np
import voronoi as vr


"2D square test"
class voronoi_2Dsquare_test(unittest.TestCase):
    """Set of tests to check class behavior for 2D-square lattice"""


    def setUp(self):
        self.Lattice_sites = np.zeros((12,2))
        for i in range(3):
            for j in range(4):
                self.Lattice_sites[i*4+j][0] = i
                self.Lattice_sites[i*4+j][1] = j
        self.voronoi = vr.voronoi_2D(self.Lattice_sites)

    def test_dimensionCount(self):
        """Check that Initialization is correct"""
        self.assertEqual(self.voronoi.num_sites, 12)
        self.assertEqual(np.shape(self.voronoi.sites), (12,2))
        self.assertEqual(np.shape(self.voronoi.mid_pt), (12,12,2))
        self.assertEqual(np.shape(self.voronoi.vector), (12,12,2))
        self.assertEqual(np.shape(self.voronoi.inter_bound), (12,12))
        self.assertEqual(np.shape(self.voronoi.geo_neighbor), (12,12))
        self.assertEqual(np.shape(self.voronoi.cell), (12,))

    def test_midPoints(self):
        """Check the middle points calculation is correct"""
        self.voronoi.network_struct()
        self.assertTrue((self.voronoi.mid_pt[5][6]==[1.0,1.5]).all())
        self.assertTrue((self.voronoi.mid_pt[5][9]==[1.5,1.0]).all())
        self.assertTrue((self.voronoi.mid_pt[5][10]==[1.5,1.5]).all())
        self.assertTrue((self.voronoi.mid_pt[5][5]==self.Lattice_sites[5]).all())

    def test_pointingVector(self):
        """Check the vector pointing from i to j is correct"""
        self.voronoi.network_struct()
        self.assertTrue((self.voronoi.vector[5][6]==[0.0,1.0]).all())
        self.assertTrue((self.voronoi.vector[5][10]==[1./np.sqrt(2),1./np.sqrt(2)]).all())
        self.assertTrue((self.voronoi.vector[5][3]==[-1./np.sqrt(5),2./np.sqrt(5)]).all())
        self.assertTrue((self.voronoi.vector[5][5]==[0.,0.]).all())

    def test_neighborRelationship(self):
        """Check the geometrical neighbor matrix is correct"""
        self.voronoi.network_struct()
        self.assertEqual(self.voronoi.geo_neighbor[5][6], 1)
        self.assertEqual(self.voronoi.geo_neighbor[5][10], 0)
        self.assertEqual(self.voronoi.geo_neighbor[5][11], 0)

    def test_neighborList(self):
        """Check the neighbor list is correct"""
        self.voronoi.network_struct()
        self.assertEqual(len(self.voronoi.neigh_list[5]), 4)
        self.assertEqual(len(self.voronoi.neigh_list[2]), 3)
        self.assertEqual(len(self.voronoi.neigh_list[3]), 2)
        self.assertTrue(self.voronoi.neigh_list[5]==[1, 4, 6, 9])
        self.assertTrue(self.voronoi.neigh_list[2]==[1, 3, 6])
        self.assertTrue(self.voronoi.neigh_list[3]==[2, 7])
        #self.assertTrue((self.voronoi.vector[5][5]==[0.,0.]).all())

    def test_minLength(self):
        """Check the function min_length is working correctly"""
        self.voronoi.network_struct()
        self.assertEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[8],20), 1.0)
        self.assertEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[5],20), 1.0)
        self.assertEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[9],20), 0.0)
        self.assertEqual(self.voronoi.min_lengthsq(5,6,[3,0],20), 2)

    def test_serchingLength(self):
        """Check the function searching_length is working correctly"""
        self.voronoi.network_struct()
        self.assertAlmostEqual(self.voronoi.searching_length(5,6,20,1E-8,1), 0.5)
        self.assertAlmostEqual(self.voronoi.searching_length(5,6,20,1E-8,-1), 0.5)
        self.assertAlmostEqual(self.voronoi.searching_length(9,10,20,1E-8,1), 20)
        self.assertAlmostEqual(self.voronoi.searching_length(9,10,20,1E-8,-1), 0.5)
        self.assertAlmostEqual(self.voronoi.searching_length(5,8,20,1E-8,1), 0.0)
        self.assertAlmostEqual(self.voronoi.searching_length(5,8,20,1E-8,-1), 0.0)

    def test_interBoundary(self):
        """Check the interface calculation is correct"""
        self.voronoi.network_struct()
        self.voronoi.calc_interlengths(20,1E-8)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][6], 1.0)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][8], 0.0)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][1], 1.0)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][9], 1.0)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][4], 1.0)

    def test_cellCalculation(self):
        """Check the cell of each site is computed correctly"""
        self.voronoi.network_struct()
        self.voronoi.calc_interlengths(20,1E-8)
        self.voronoi.compute_cell()
        self.assertAlmostEqual(self.voronoi.cell[5], 1.0)
        self.assertAlmostEqual(self.voronoi.cell[6], 1.0)

"2D hex test"
class voronoi_2Dhex_test(unittest.TestCase):
    """Set of tests to check class behavior for 2-D hex lattice"""


    def setUp(self):
        self.Lattice_sites = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[0.5,np.sqrt(3)/2.],[1.5,np.sqrt(3)/2.]
                                       ,[2.5,np.sqrt(3)/2.],[3.5,np.sqrt(3)/2.],[0.,np.sqrt(3)],[1.,np.sqrt(3)]
                                       ,[2.,np.sqrt(3)],[3.,np.sqrt(3)]])
        #print self.Lattice_sites
        self.voronoi = vr.voronoi_2D(self.Lattice_sites)

    def test_dimensionCount(self):
        """Check that Initialization is correct"""
        self.assertEqual(self.voronoi.num_sites, 12)
        self.assertEqual(np.shape(self.voronoi.sites), (12,2))
        self.assertEqual(np.shape(self.voronoi.mid_pt), (12,12,2))
        self.assertEqual(np.shape(self.voronoi.vector), (12,12,2))
        self.assertEqual(np.shape(self.voronoi.inter_bound), (12,12))
        self.assertEqual(np.shape(self.voronoi.geo_neighbor), (12,12))
        self.assertEqual(np.shape(self.voronoi.cell), (12,))

    def test_midPoints(self):
        """Check the middle points calculation is correct"""
        self.voronoi.network_struct()
        np.testing.assert_array_almost_equal(self.voronoi.mid_pt[5][6], [2.0,np.sqrt(3)/2.])
        np.testing.assert_array_almost_equal(self.voronoi.mid_pt[5][9], [1.25,0.75*np.sqrt(3)])
        np.testing.assert_array_almost_equal(self.voronoi.mid_pt[5][6], self.voronoi.mid_pt[3][9])
        np.testing.assert_array_almost_equal(self.voronoi.mid_pt[5][5], self.Lattice_sites[5])

    def test_pointingVector(self):
        """Check the vector pointing from i to j is correct"""
        self.voronoi.network_struct()
        np.testing.assert_array_almost_equal(self.voronoi.vector[5][6], [1.,0.])
        np.testing.assert_array_almost_equal(self.voronoi.vector[5][10], [1./2.,np.sqrt(3)/2.])
        np.testing.assert_array_almost_equal(self.voronoi.vector[5][3], [np.sqrt(3)/2.,-1./2.])
        np.testing.assert_array_almost_equal(self.voronoi.vector[5][5], [0.,0.])

    def test_neighborRelationship(self):
        """Check the geometrical neighbor matrix is correct"""
        self.voronoi.network_struct()
        self.assertEqual(self.voronoi.geo_neighbor[5][6], 1)
        self.assertEqual(self.voronoi.geo_neighbor[5][10], 1)
        self.assertEqual(self.voronoi.geo_neighbor[5][11], 0)

    def test_neighborList(self):
        """Check the neighbor list is correct"""
        self.voronoi.network_struct()
        self.assertEqual(len(self.voronoi.neigh_list[5]), 6)
        self.assertEqual(len(self.voronoi.neigh_list[2]), 4)
        self.assertEqual(len(self.voronoi.neigh_list[3]), 3)
        self.assertEqual(len(self.voronoi.neigh_list[7]), 3)
        self.assertTrue(self.voronoi.neigh_list[5]==[1, 2, 4, 6, 9, 10])
        self.assertTrue(self.voronoi.neigh_list[2]==[1, 3, 5, 6])
        self.assertTrue(self.voronoi.neigh_list[3]==[2, 6, 7])
        self.assertTrue(self.voronoi.neigh_list[7]==[3, 6, 11])

    def test_minLength(self):
        """Check the function min_length is working correctly"""
        self.voronoi.network_struct()
        self.assertAlmostEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[8],20), 1.0)
        self.assertAlmostEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[5],20), 1.0)
        self.assertAlmostEqual(self.voronoi.min_lengthsq(5,6,self.Lattice_sites[9],20), 0.0)
        self.assertAlmostEqual(self.voronoi.min_lengthsq(5,6,[2,-1],20), 1.0)

    def test_serchingLength(self):
        """Check the function searching_length is working correctly"""
        self.voronoi.network_struct()
        self.assertAlmostEqual(self.voronoi.searching_length(5,6,20,1E-8,1), np.sqrt(3)/6.)
        self.assertAlmostEqual(self.voronoi.searching_length(5,6,20,1E-8,-1), np.sqrt(3)/6.)
        self.assertAlmostEqual(self.voronoi.searching_length(9,10,20,1E-8,1), np.sqrt(3)/6.)
        self.assertAlmostEqual(self.voronoi.searching_length(9,10,20,1E-8,-1), 20.)
        self.assertAlmostEqual(self.voronoi.searching_length(5,8,20,1E-8,1), 0.)
        self.assertAlmostEqual(self.voronoi.searching_length(5,8,20,1E-8,-1), 0.)

    def test_interBoundary(self):
        """Check the interface calculation is correct"""
        self.voronoi.network_struct()
        self.voronoi.calc_interlengths(20,1E-8)
        self.assertAlmostEqual(self.voronoi.inter_bound[5][8], 0.)
        for i in self.voronoi.neigh_list[5]:
            self.assertAlmostEqual(self.voronoi.inter_bound[5][i], np.sqrt(3)/3.)

    def test_cellCalculation(self):
        """Check the cell of each site is computed correctly"""
        self.voronoi.network_struct()
        self.voronoi.calc_interlengths(20,1E-8)
        self.voronoi.compute_cell()
        self.assertAlmostEqual(self.voronoi.cell[5], np.sqrt(3)/2.)
        self.assertAlmostEqual(self.voronoi.cell[6], np.sqrt(3)/2.)


"3D square test"
class voronoi_3Dsquare_test(unittest.TestCase):
    """Set of tests to check class behavior for 2D-square lattice"""
    def setUp(self):
        self.Lattice_sites = np.zeros((27,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Lattice_sites[i*9+j*3+k][0] = i
                    self.Lattice_sites[i*9+j*3+k][1] = j
                    self.Lattice_sites[i*9+j*3+k][2] = k
        self.voronoi = vr.voronoi_3D(self.Lattice_sites)

    def test_dimensionCount(self):
        """Check that Initialization is correct"""
        self.assertEqual(self.voronoi.num_sites, 27)
        self.assertEqual(np.shape(self.voronoi.sites), (27,3))
        self.assertEqual(np.shape(self.voronoi.mid_pt), (27,27,3))
        self.assertEqual(np.shape(self.voronoi.vector), (27,27,3))
        self.assertEqual(np.shape(self.voronoi.interface), (27,27))
        self.assertEqual(np.shape(self.voronoi.geo_neighbor), (27,27))

    def test_midPoints(self):
        """Check the middle points calculation is correct"""
        j = 13
        self.voronoi.network_struct()
        for i in range(27):
            np.testing.assert_array_almost_equal(self.voronoi.mid_pt[i][j], 1./2.*(self.Lattice_sites[i]+self.Lattice_sites[j]))

    def test_pointingVector(self):
        """Check the vector pointing from j to i is correct"""
        j = 13
        self.voronoi.network_struct()
        for i in range(27):
            dist_ji = np.sqrt(np.dot(self.Lattice_sites[i]-self.Lattice_sites[j],self.Lattice_sites[i]-self.Lattice_sites[j]))
            np.testing.assert_array_almost_equal(dist_ji*self.voronoi.vector[j][i],self.Lattice_sites[i]-self.Lattice_sites[j])

    def test_neighborRelationship(self):
        """Check the geometrical neighbor matrix is correct"""
        j = 13
        neighbors = [4, 10, 12, 14, 16, 22]
        non_neighbors = []
        for i in range(27):
            if i not in neighbors:
                non_neighbors.append(i)
        self.voronoi.network_struct()
        for i in neighbors:
            self.assertEqual(self.voronoi.geo_neighbor[i][j], 1)
        for i in non_neighbors:
            self.assertEqual(self.voronoi.geo_neighbor[i][j], 0)

    def test_neighborList(self):
        """Check the neighbor list is correct"""
        j = 13
        neighbors = [4, 10, 12, 14, 16, 22]
        self.voronoi.network_struct()
        self.assertEqual(len(self.voronoi.neigh_list[j]), 6)
        self.assertTrue(self.voronoi.neigh_list[j]==neighbors)
        j = 0
        neighbors = [1, 3, 9]
        self.assertEqual(len(self.voronoi.neigh_list[j]), 3)
        self.assertTrue(self.voronoi.neigh_list[j]==neighbors)
        j = 1
        neighbors = [0, 2, 4, 10]
        self.assertEqual(len(self.voronoi.neigh_list[j]), 4)
        self.assertTrue(self.voronoi.neigh_list[j]==neighbors)
        j = 4
        neighbors = [1, 3, 5, 7, 13]
        self.assertEqual(len(self.voronoi.neigh_list[j]), 5)
        self.assertTrue(self.voronoi.neigh_list[j]==neighbors)

    def test_minLength(self):
        """Check the function min_length is working correctly"""
        i = 13
        j = 4
        self.voronoi.network_struct()
        for k in self.voronoi.neigh_list[i]:
            if(k!=4):
                self.assertAlmostEqual(self.voronoi.min_lengthsq(i,j,self.Lattice_sites[k],20), 0.0)
            else:
                self.assertAlmostEqual(self.voronoi.min_lengthsq(i,j,self.Lattice_sites[k],20), 1.0)
        self.assertAlmostEqual(self.voronoi.min_lengthsq(i,j,self.Lattice_sites[i],20), 1.0)
        self.assertAlmostEqual(self.voronoi.min_lengthsq(i,j,[-1, 1, 1],20), 2.0)

    def test_horizonBasis(self):
        """Check the Horizontal plane basis calculation is correct"""
        j = 13
        self.voronoi.network_struct()
        basis = np.zeros((3,3))
        for i in range(27):
            if(i==j):
                continue
            basis[0] = self.voronoi.vector[j][i]
            hori_base = self.voronoi.get_horizonBasis(basis[0],np.sqrt(3)/3.)
            basis[1] = hori_base[0]
            basis[2] = hori_base[1]
            for ii in range(3):
                for jj in range(ii,3):
                    self.assertAlmostEqual(np.dot(basis[ii],basis[jj]),(1.0 if(ii==jj) else 0.0))

    def test_searchingLength(self):
        """Check the function searching_length function is working correctly"""
        j = 13
        self.voronoi.network_struct()
        i = 4
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([0., 1., 0.]),20.,1E-8),1./2.)
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([0., 0., 1.]),20.,1E-8),1./2.)
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([0., 1., 1.]),20.,1E-8),1./np.sqrt(2))
        i = 0
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([0., 1., -1.]),20.,1E-8),0.)
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([-1., 0., 1.]),20.,1E-8),0.)
        i = 1
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([0., 0., 1.]),20.,1E-8),0.)
        self.assertAlmostEqual(self.voronoi.searching_length(i,j,np.array([-1., 1., 0.]),20.,1E-8),0.)

    def test_commonfaceCalculation(self):
        """Check the area calculation is correct"""
        j = 13
        self.voronoi.network_struct()
        for i in range(27):
            if(i in self.voronoi.neigh_list[j]):
                self.assertAlmostEqual(self.voronoi.compute_area(i,j,np.sqrt(3)/3.,100,20.,1.E-8), 1.0, delta = 2.E-3)
            else:
                self.assertAlmostEqual(self.voronoi.compute_area(i,j,np.sqrt(3)/3.,100,20.,1.E-8), 0.0, delta = 2.E-3)

    def test_interfaceCalculation(self):
        self.voronoi.network_struct()
        self.voronoi.calc_interfaces(1./np.sqrt(3),100,20.,1E-8)
        j = 13
        for i in range(27):
            if(i in self.voronoi.neigh_list[j]):
                self.assertAlmostEqual(self.voronoi.interface[i][j], 1.0, delta = 2.E-3)
            else:
                self.assertAlmostEqual(self.voronoi.interface[i][j], 0.0, delta = 2.E-3)




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(voronoi_2Dhex_test)
    unittest.TextTestRunner(verbosity=2).run(suite)
