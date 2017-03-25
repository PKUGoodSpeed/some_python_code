__author__ = 'zeboli1'

import unittest
import numpy as np
from scipy import linalg as LA
import evolution_schemes as ES


"""
*****************
Creating testing cases for the evolution schemes
"""

class symm_matrix_test(unittest.TestCase):
    """For time-independent matrix"""

    def setUp(self):
        """set matrix"""
        self.unit = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
        self.matrix = np.array([[-4., 1., 1.],[1., -4., 2.],[1., 2., -4.]])
        val,vec = LA.eig(self.matrix)
        idx = val.argsort()[::-1]
        val = val[idx]
        vec = vec[:,idx]
        self.value = np.real(val)
        self.vectors = np.real(vec)
        self.time_duration = 100.

    #test the class is generated is correct
    #test whether the eigen information is computed correctly
    def test_initilization(self):
        n_dim = 3
        for i in range(n_dim):
            np.testing.assert_array_almost_equal(np.dot(self.matrix,self.vectors.transpose()[i]),self.value[i]*self.vectors.transpose()[i])
        np.testing.assert_array_almost_equal(np.dot(self.vectors.transpose(),self.vectors),self.unit)
        #test decomposition
        n_cases = 17
        for j in range(n_cases):
            vec = np.random.random((n_dim,))
            coeff = np.dot(self.vectors.transpose(),vec)
            vec_rec = np.dot(self.vectors,coeff)
            np.testing.assert_array_almost_equal(vec,vec_rec)

    #test_finite_difference scheme
    def test_fd_steps(self):
        n_cases = 17
        n_dim = 3
        delta_t = 0.1/max(abs(self.value))
        print delta_t
        for k in range(n_cases):
            var = np.random.random((3,))
            srcs = np.array(np.random.random((3,)))
            fd_scheme = ES.finite_diff_evo(var)
            self.assertEqual(fd_scheme.dimension,n_dim)
            coeff = np.dot(self.vectors.transpose(),var)
            src = np.dot(self.vectors.transpose(),srcs)
            coeff = coeff*np.exp(self.value*self.time_duration)
            for i in range(3):
                if(abs(self.value[i])<1.E-17):
                    src[i] = src[i]*self.time_duration
                else:
                    src[i] = src[i]*(np.exp(self.value[i]*self.time_duration) - 1.)/self.value[i]
            coeff = coeff + src
            var_analytical = np.dot(self.vectors,coeff)
            time = 0.
            while(time+delta_t < self.time_duration):
                fd_scheme.evol_step(self.matrix,delta_t,source=srcs)
                time += delta_t
            fd_scheme.evol_step(self.matrix,self.time_duration-time,source=srcs)
            np.testing.assert_array_almost_equal(fd_scheme.var,var_analytical)
    #test adaptive time step calculation
    def test_fd_adaptive(self):
        n_cases = 17
        n_dim = 3
        kappa = 1.E-2
        delta_t = 0.1/max(abs(self.value))
        for k in range(n_cases):
            var = np.random.random((3,))
            srcs = np.array(np.random.random((3,)))*0.0
            fd_scheme = ES.finite_diff_evo(var)
            self.assertEqual(fd_scheme.dimension,n_dim)
            coeff = np.dot(self.vectors.transpose(),var)
            src = np.dot(self.vectors.transpose(),srcs)
            coeff = coeff*np.exp(self.value*self.time_duration)
            for i in range(3):
                if(abs(self.value[i])<1.E-17):
                    src[i] = src[i]*self.time_duration
                else:
                    src[i] = src[i]*(np.exp(self.value[i]*self.time_duration) - 1.)/self.value[i]
            coeff = coeff + src
            var_analytical = np.dot(self.vectors,coeff)
            time = 0.
            while(time+delta_t < self.time_duration):
                adapt = fd_scheme.evol_step(self.matrix,delta_t,source=srcs)
                time += delta_t
                delta_t = fd_scheme.adaptive_step(kappa,adapt)
            fd_scheme.evol_step(self.matrix,self.time_duration-time,source=srcs)
            np.testing.assert_array_almost_equal(fd_scheme.var,var_analytical)

    #test the function for getting eigenvalue information
    def test_eigen_info(self):
        var = np.random.random((3,))
        eg_scheme = ES.eigen_decom_evo(var)
        values,right_vec,left_vec = eg_scheme.get_eigen_info(self.matrix)
        np.testing.assert_array_almost_equal(values,self.value)
        np.testing.assert_array_almost_equal(right_vec,self.vectors)
        np.testing.assert_array_almost_equal(left_vec,self.vectors)

    #test eigen-decomposition evolution with adaptive_steps
    def test_eg_steps(self):
        n_cases = 17
        n_dim = 3
        kappa = 1.7
        delta_t = 1.7/max(abs(self.value))
        for k in range(n_cases):
            var = np.random.random((3,))
            eg_scheme = ES.eigen_decom_evo(var)
            self.assertEqual(eg_scheme.dimension,n_dim)
            coeff = np.dot(self.vectors.transpose(),var)
            coeff = coeff*np.exp(self.value*self.time_duration)
            var_analytical = np.dot(self.vectors,coeff)
            time = 0.
            while(time+delta_t < self.time_duration):
                adapt = eg_scheme.evol_step(self.matrix,delta_t,eg_scheme.var)
                eg_scheme.update()
                time += delta_t
                delta_t = eg_scheme.adaptive_step(kappa,adapt)
            eg_scheme.evol_step(self.matrix,self.time_duration-time,eg_scheme.var)
            eg_scheme.update()
            np.testing.assert_array_almost_equal(eg_scheme.var,var_analytical)


class vacancy_transport_test(unittest.TestCase):
    """For time-dependent matrix"""

    def setUp(self):
        """set matrix"""
        self.dim = 3
        self.init_conc = np.array([1.E-1,1.E-1,1.E-1])
        self.factor = 1.
        self.time_duration = 50./self.factor

    #compute matrix
    def comput_matrix(self,w_01,w_12,vec):
        w_02 = w_01*w_12
        w_10 = self.factor/w_01
        w_20 = self.factor/w_02
        w_21 = self.factor/w_12
        matrix = np.array([[-w_01*(1.-vec[1])-w_02*(1.-vec[2]), w_10*(1.-vec[0]), w_20*(1.-vec[0])],
                           [w_01*(1.-vec[1]), -w_10*(1.-vec[0])-w_12*(1.-vec[2]), w_21*(1.-vec[1])],
                           [w_02*(1.-vec[2]), w_12*(1.-vec[2]), -w_20*(1.-vec[0])-w_21*(1.-vec[1])]])
        return matrix

    #test the eigen_information calculation is correct
    def test_eigen_info(self):
        eg_evol = ES.eigen_decom_evo(self.init_conc)
        n_cases = 17
        for k in range(n_cases):
            w_01 = np.random.rand()
            w_12 = np.random.rand()
            matrix = self.comput_matrix(w_01,w_12,self.init_conc)
            unit = np.identity(3)
            val,right_vec,left_vec = eg_evol.get_eigen_info(matrix,n_dim=3)
            np.testing.assert_array_almost_equal(np.dot(left_vec.transpose(),right_vec).real,unit)

    #using finite difference scheme to test the eigen_decomposition scheme
    def test_eg_decomp_evol(self):
        n_cases = 17
        kappa = 0.1
        for k in range(n_cases):
            eg_evol = ES.eigen_decom_evo(self.init_conc)
            fd_evol = ES.finite_diff_evo(self.init_conc)
            w_01 = np.random.rand()
            w_12 = np.random.rand()
            srcs = np.array([1.E-3,0.E-3,-1.E-3])
            var = np.array([0.,0.,0.])

            #using finite difference scheme
            matrix = self.comput_matrix(w_01,w_12,fd_evol.var)
            val,vec = LA.eig(matrix)
            delta_t = 0.1/max(abs(val))
            time = 0.
            while(time+delta_t < self.time_duration):
                matrix = self.comput_matrix(w_01,w_12,fd_evol.var)
                fd_evol.evol_step(matrix,delta_t,source=srcs)
                time += delta_t
            matrix = self.comput_matrix(w_01,w_12,fd_evol.var)
            fd_evol.evol_step(matrix,self.time_duration-time,source=srcs)

            #using eigen_decomposition scheme
            delta_t *= 10
            time = 0.
            while(time+delta_t < self.time_duration):
                matrix = self.comput_matrix(w_01,w_12,fd_evol.var)
                adapt = eg_evol.evol_step(matrix,delta_t,eg_evol.var,source=srcs)
                eg_evol.update()
                time += delta_t
                delta_t = eg_evol.adaptive_step(kappa,adapt)
            matrix = self.comput_matrix(w_01,w_12,fd_evol.var)
            eg_evol.evol_step(matrix,self.time_duration-time,eg_evol.var,source=srcs)
            eg_evol.update()
            np.testing.assert_array_almost_equal(eg_evol.var,fd_evol.var)







if __name__ == '__main__':
    '''
    print "time-independent transition matrix"
    suite = unittest.TestLoader().loadTestsFromTestCase(symm_matrix_test)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print "\n\n\nIlove Hyojung kim forever!!\n"
    '''



    print "simple vacancy transport problem"
    suite = unittest.TestLoader().loadTestsFromTestCase(vacancy_transport_test)
    unittest.TextTestRunner(verbosity=2).run(suite)
