__author__ = 'zeboli1'

import unittest
import numpy as np
from scipy import linalg as LA
import voronoi2 as vr


"2D square test for voronoi2.py"
class voronoi_2Dsquare_test(unittest.TestCase):
    """Set of tests to check class behavior for 2D-square lattice"""


    def setUp(self):
        """setup 2D 3X4 lattice"""
        self.Lattice_sites = np.zeros((12,2))
        for i in range(3):
            for j in range(4):
                self.Lattice_sites[i*4+j][0] = i
                self.Lattice_sites[i*4+j][1] = j
        self.voronoi = vr.voro_info_2D(self.Lattice_sites)

    def test_dimensionCount(self):
        """Check that Initialization (dimensions of arrays) is correct"""
        self.assertEqual(self.voronoi.num_sites, 12)
        self.assertEqual(np.shape(self.voronoi.sites), (12,2))
        vor_info = self.voronoi.vor
        vertices = vor_info.vertices
        self.assertEqual(np.shape(vertices), (6,2))
        edges = vor_info.ridge_points
        self.assertEqual(np.shape(edges), (17,2))

    def test_neighborJudge(self):
        """Check that the neighbor judgements are correct"""
        sites = self.Lattice_sites
        n_sites = len(sites)
        edges = self.voronoi.vor.ridge_points
        for i in range(n_sites):
            for j in range(i+1,n_sites):
                if(np.dot(sites[i]-sites[j],sites[i]-sites[j]) == 1.):
                    self.assertEqual(self.voronoi.judge_neighbor(i,j,edges),True)
                else:
                    self.assertEqual(self.voronoi.judge_neighbor(i,j,edges),False)

    def test_neighborList(self):
        """Check the neighbor list is correct"""
        self.voronoi.creating_neighbor()
        self.assertEqual(len(self.voronoi.nei_list[5]), 4)
        self.assertEqual(len(self.voronoi.nei_list[2]), 3)
        self.assertEqual(len(self.voronoi.nei_list[3]), 2)
        self.assertTrue(set(self.voronoi.nei_list[5])==set([1, 4, 6, 9]))
        self.assertTrue(set(self.voronoi.nei_list[2])==set([1, 3, 6]))
        self.assertTrue(set(self.voronoi.nei_list[3])==set([2, 7]))

    def test_attachJudge(self):
        """Check the attaching judge is correct"""
        i = 5
        self.assertEqual(self.voronoi.judge_attach(i,[0.5,0.5]),True)
        self.assertEqual(self.voronoi.judge_attach(i,[1.5,0.5]),True)
        self.assertEqual(self.voronoi.judge_attach(i,[0.5,1.5]),True)
        self.assertEqual(self.voronoi.judge_attach(i,[1.5,1.5]),True)
        self.assertEqual(self.voronoi.judge_attach(i,[0.5,2.5]),False)
        self.assertEqual(self.voronoi.judge_attach(i,[1.5,2.5]),False)
        self.assertEqual(self.voronoi.judge_attach(i,[0.6,0.6]),True)
        self.assertEqual(self.voronoi.judge_attach(i,[0.4,0.4]),False)

    def test_vertex_list(self):
        """Check the attached vertices is correct"""
        vert_list = self.voronoi.vor.vertices
        n_vert = len(vert_list)
        self.voronoi.attach_vertices(vert_list)
        for i in range(self.voronoi.num_sites):
            for k in self.voronoi.ver_list[i]:
                if(k>=n_vert):
                    continue
                distancesq = np.dot(self.voronoi.sites[i]-vert_list[k],self.voronoi.sites[i]-vert_list[k])
                self.assertEqual(distancesq,1./2.)

    def test_ridgeComputing(self):
        """Chech the ridge lengthes are computed correctly"""
        self.voronoi.creating_neighbor()
        self.voronoi.attach_vertices()
        for i in range(self.voronoi.num_sites):
            for j in self.voronoi.nei_list[i]:
                self.assertEqual(self.voronoi.compute_ridge(i,j,default_length=1.),1.)

    def test_ridgeMatrix(self):
        """Check the ridgematrix is constructed correctly"""
        self.voronoi.creating_neighbor()
        self.voronoi.attach_vertices()
        self.voronoi.ridge_network(default_length=1.)
        for i in range(self.voronoi.num_sites):
            for j in range(i+1,self.voronoi.num_sites):
                if(self.voronoi.judge_neighbor(i,j)):
                    self.assertEqual(self.voronoi.ridge_length[i][j],1.)
                else:
                    self.assertEqual(self.voronoi.ridge_length[i][j],0.)

    def test_cellCalculation(self):
        """Check the cell of each site is computed correctly"""
        self.voronoi.creating_neighbor()
        self.voronoi.attach_vertices()
        self.voronoi.ridge_network(default_length=1.)
        self.voronoi.compute_cells()
        self.assertAlmostEqual(self.voronoi.cell[5], 1.0)
        self.assertAlmostEqual(self.voronoi.cell[6], 1.0)



"3D square test for voronoi2.py"
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
        self.voronoi = vr.voro_info_3D(self.Lattice_sites)

    def test_dimensionCount(self):
        """Check that Initialization (dimensions of arrays) is correct"""
        self.assertEqual(self.voronoi.num_sites, 27)
        self.assertEqual(np.shape(self.voronoi.sites), (27,3))
        vor_info = self.voronoi.vor
        vertices = vor_info.vertices
        self.assertEqual(np.shape(vertices), (8,3))
        #The neighbor relationship computed from the scipy is not complete
        #edges = vor_info.ridge_points
        #self.assertEqual(np.shape(edges), (54,2))

    def test_neighborList(self):
        """Check the neighbor list is correct"""
        self.voronoi.creating_neighbor()
        for i in range(self.voronoi.num_sites):
            for j in range(i+1,self.voronoi.num_sites):
                distancesq = np.dot(self.Lattice_sites[i]-self.Lattice_sites[j],self.Lattice_sites[i]-self.Lattice_sites[j])
                if(distancesq==1.):
                    self.assertTrue(j in self.voronoi.nei_list[i])
                    self.assertTrue(i in self.voronoi.nei_list[j])
                else:
                    self.assertTrue(j not in self.voronoi.nei_list[i])
                    self.assertTrue(i not in self.voronoi.nei_list[j])

    def test_attachJudge(self):
        """Check the attaching judge is correct"""
        j = 13
        n_pts = 52
        for i in range(n_pts):
            point = np.array([0.5+np.random.rand(),0.5+np.random.rand(),0.5+np.random.rand()])
            self.assertTrue(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([-0.5+np.random.rand(),0.5+np.random.rand(),0.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([1.5+np.random.rand(),0.5+np.random.rand(),0.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([0.5+np.random.rand(),1.5+np.random.rand(),0.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([0.5+np.random.rand(),-0.5+np.random.rand(),0.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([0.5+np.random.rand(),0.5+np.random.rand(),-0.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([0.5+np.random.rand(),0.5+np.random.rand(),1.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))
        for i in range(n_pts):
            point = np.array([1.5+np.random.rand(),1.5+np.random.rand(),1.5+np.random.rand()])
            self.assertFalse(self.voronoi.judge_attach(j,point))

    def test_vertex_list(self):
        """Check the attached vertices is correct"""
        vert_list = self.voronoi.vor.vertices
        n_vert = len(vert_list)
        self.voronoi.attach_vertices(vert_list)
        for i in range(self.voronoi.num_sites):
            for k in self.voronoi.ver_list[i]:
                if(k>=n_vert):
                    continue
                distancesq = np.dot(self.voronoi.sites[i]-vert_list[k],self.voronoi.sites[i]-vert_list[k])
                self.assertEqual(distancesq,3./4.)

    def test_horizonBasis(self):
        """Check the Horizontal plane basis calculation is correct"""
        n_testcase = 100
        basis = np.zeros((3,3))
        for i in range(n_testcase):
            vec = np.array([-0.5+np.random.rand(),-0.5+np.random.rand(),-0.5+np.random.rand()])
            basis[0] = vec/LA.norm(vec)
            hori_base = self.voronoi.get_horizonBasis(basis[0],np.sqrt(3)/3.)
            basis[1] = hori_base[0]
            basis[2] = hori_base[1]
            for ii in range(3):
                for jj in range(ii,3):
                    self.assertAlmostEqual(np.dot(basis[ii],basis[jj]),(1.0 if(ii==jj) else 0.0))

    def test_angleCalculation(self):
        """Check the Polar coordinates (agnle) calculation is correct"""
        n_testcase = 100
        Range = 100
        basis = np.zeros((3,3))
        for i in range(n_testcase):
            vec = np.array([-0.5+np.random.rand(),-0.5+np.random.rand(),-0.5+np.random.rand()])
            basis[0] = vec/LA.norm(vec)
            hori_base = self.voronoi.get_horizonBasis(basis[0],np.sqrt(3)/3.)
            basis[1] = hori_base[0]
            basis[2] = hori_base[1]
            Cta = (np.random.rand()-0.5)*2.*np.pi
            origin = np.array([Range*(np.random.rand()-0.5),Range*(np.random.rand()-0.5),Range*(np.random.rand()-0.5)])
            r = Range*np.random.rand()
            position = origin + r*np.cos(Cta)*basis[1] + r*np.sin(Cta)*basis[2] + Range*(np.random.rand()-0.5)*basis[0]
            self.assertAlmostEqual(self.voronoi.angle(position,origin,basis[1],basis[2]),Cta)

    def test_polyCalculation(self):
        """check the polygon calculation is correct"""
        origin = [0.0,0.0,0.0]
        x_base = [1.0,0.0,0.0]
        y_base = [0.0,1.0,0.0]
        vert_list = np.array([[1.0,0.0,0.0],
                              [1.0,1.0,0.0],
                              [0.0,1.0,0.0],
                              [-1.0,1.0,0.0],
                              [-1.0,0.0,0.0],
                              [-1.0,-1.0,0.0],
                              [0.0,-1.0,0.0],
                              [1.0,-1.0,0.0],])
        ind_list = [7,4,5,6,2,3,1,0]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),4.0)
        ind_list = [1,2]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),0.0)
        ind_list = [4,2,0,6]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),2.0)
        '''
        #These two cases suggests that the origin should be located within the polygon
        ind_list = [4,3,2,5]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),1.0)
        ind_list = [2,3,1]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),0.0)
        '''
        ind_list = [0,2,5]
        self.assertAlmostEqual(self.voronoi.poly_area(ind_list,origin,x_base,y_base,vert_list),1.5)

    def test_areaCalculation(self):
        """Check the ridge area calculation is correct"""
        j = 13
        self.voronoi.creating_neighbor()
        self.voronoi.attach_vertices()
        for i in range(self.voronoi.num_sites):
            if(i in self.voronoi.nei_list[j]):
                self.assertAlmostEqual(self.voronoi.compute_ridgearea(i,j),1.0)
            else:
                self.assertAlmostEqual(self.voronoi.compute_ridgearea(i,j),0.0)

    def test_cellCalculation(self):
        """Check the cell of each site is computed correctly"""
        self.voronoi.creating_neighbor()
        self.voronoi.attach_vertices()
        self.voronoi.ridge_network(default_area=1.)
        self.voronoi.compute_cells()
        self.assertAlmostEqual(self.voronoi.cell[13], 1.0)



    '''




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


    '''







if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(voronoi_3Dsquare_test)
    unittest.TextTestRunner(verbosity=2).run(suite)