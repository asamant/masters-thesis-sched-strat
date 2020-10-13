#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:02:31 2018

original author: @ggleizer
modified by @asamant to include only relevant parts.

"""
import numpy as np
from numpy import random
import control as ct
import warnings
import scipy.linalg as la
from scipy import integrate
from etcutil import normalize_state_to_lyapunov
import logging
import cvxpy as cvx

ct.use_numpy_matrix(flag=False)

_MAX_TRIES_TIGHT_LYAPUNOV = 10
_ASSERT_SMALL_NUMBER = 1e-6
_LMIS_SMALL_IDENTITY_FACTOR = 1e-6

''' linearetc.py '''
''' Main building blocks for building your ETC implementation'''


'''
    AUXILIARY
'''


def is_positive_definite(A):
    return (la.eig(A)[0] > 0).all()


def is_positive_semidefinite(A):
    return (la.eig(A)[0] >= 0).all()


'''
    SYSTEM DEFINITION
'''

class Plant:
    nx = 0
    ny = 0
    nu = 0
    nw = 0


class LinearPlant(Plant):
    A = np.array([])
    B = np.array([])
    C = np.array([])
    D = np.array([])
    E = np.array([])

    def __init__(self, A, B, C, D=None, E=None):
        # Retrieve A
        A = np.array(A)
        if len(A.shape) == 0:
            A = np.array([[A]])
        nx1, nx2 = A.shape
        assert nx1 == nx2, 'Matrix A needs to be square'
        self.nx = nx1
        self.A = A

        # Retrieve B
        try:
            nx, nu = B.shape
            self.B = B
        except ValueError:  # B is a vector
            nx = B.shape[0]
            nu = 1
            self.B = B.reshape(nx, nu)
        assert self.nx == nx, 'B should have the same number of rows as A'
        self.nu = nu

        # Retrieve C
        try:
            ny, nx = C.shape
            self.C = C
        except ValueError:
            nx = C.shape[0]
            ny = 1
            self.C = C.reshape(ny, nx)
        assert self.nx == nx, 'C should have the same number of columns as A'
        self.ny = ny

        # Retreive D
        if D is None:
            self.D = np.zeros((ny, nu))
        else:
            D = np.array(D)
            assert len(D.shape) in [0, 2], \
                'D cannot be a one-dimensional array'
            try:
                ny, nu = D.shape
                self.D = D
            except ValueError:
                ny = 1
                nu = 1
                self.D = np.array([[D]])
            assert self.ny == ny, 'D must have the same number of rows as C'
            assert self.nu == nu, 'D must have the same number of columns as B'

        # Retreive E
        if E is None:
            self.E = np.zeros((nx, 0))
            self.nw = 0
        else:
            try:
                nx, nw = E.shape
                self.E = E
            except ValueError:  # B is a vector
                nx = E.shape[0]
                nw = 1
                self.E = E.reshape(nx, nw)
            assert self.nx == nx, 'E should have the same number of rows as A'
            self.nw = nw

    @property
    def states_as_output(self):
        return (self.ny == self.nx) and (self.C == np.eye(self.nx)).all()

    def measurement(self, x, noise=0):
        return self.C @ x + noise


class Controller:
    nx = 0
    ny = 0
    nu = 0
    is_dynamic = False
    h = None

    @property
    def is_discrete_time(self):
        return not(self.h is None)

    def output(self, y, xc=None):
        return np.zeros(self.nu)


class LinearController(Controller):
    A = np.array([])
    B = np.array([])
    C = np.array([])
    D = np.array([])

    def __init__(self, *args):
        if len(args) <= 2:  # Static controller
            self.is_dynamic = False

            K = args[0]
            try:
                h = args[1]
            except IndexError:
                h = None

            K = np.array(K)
            if len(K.shape) == 0:
                K = np.array([[K]])
            assert len(K.shape) == 2, \
                'K needs to be a scalar or a 2-dimensional array'
            nu, ny = K.shape

            self.nx = 0
            self.nu = nu
            self.ny = ny
            self.A = np.zeros((0, 0))
            self.B = np.zeros((0, ny))
            self.C = np.zeros((nu, 0))
            self.D = K
            self.h = h

        elif len(args) in (4, 5):  # Dynamic controller
            try:
                A, B, C, D, h = args
            except ValueError:
                A, B, C, D = args
                h = None

            dummy_plant = LinearPlant(A, B, C, D)
            self.A = dummy_plant.A
            self.B = dummy_plant.B
            self.C = dummy_plant.C
            self.D = dummy_plant.D
            self.nx = dummy_plant.nx
            self.nu = dummy_plant.nu
            self.ny = dummy_plant.ny
            self.h = h
            self.is_dynamic = True

    @property
    def K(self):
        if not self.is_dynamic:
            return self.D
        else:
            raise AttributeError('Controller is dynamic, no gain available')

    def output(self, y, xc=None):
        if self.is_dynamic:
            return self.C @ xc + self.D @ y
        else:
            return self.D @ y


'''
    ETC ERRORS
'''


class ETCWarning(Warning):
    pass


class ETCError(Exception):
    pass


'''
    BASE CLASS (SAMPLING STRATEGIES)
'''

class SampleAndHoldController:
    plant: Plant
    controller: Controller
    is_discrete_time = False

    def __init__(self, plant: Plant, controller: Controller):
        assert plant.ny == controller.ny and plant.nu == controller.nu, \
            'Controller and plant have inconsistent dimensions'
        self.plant = plant
        self.controller = controller
        self.is_state_feedback = plant.states_as_output and \
            not controller.is_dynamic


class LinearSampleAndHoldController(SampleAndHoldController):
    P: np.array  # Lyapunov matrix
    Qlyap:  np.array  # Continuous-time Lyapunov decay matrix

    def __init__(self, plant, controller, P=None, Q=None):
        super().__init__(plant, controller)

        nx = plant.nx
        Acl = plant.A + plant.B @ controller.K  # Closed loop matrix

        # Get/make Lyapunov info
        if P is None:
            if Q is None:
                Q = np.eye(nx)
            else:
                assert (Q == Q.T).all() and is_positive_definite(Q), \
                    'Q must be symmetric positive definite'
            P = ct.lyap(Acl.T, Q)
            if not is_positive_definite(P):
                msg = 'Closed loop system is not stable in continuous-time'
                warnings.warn(ETCWarning(msg))
        else:  # P is given
            assert (P == P.T).all() and is_positive_definite(P), \
                'Provided P matrix is not symmetric positive definite'
            if Q is None:
                Q = - Acl.T @ P - P @ Acl
                if not is_positive_definite(Q):
                    msg = 'Provided P matrix is not a continuous-time' \
                        ' Lyapunov matrix'
            else:
                assert la.norm(Acl.T@P + P@Acl + Q) <= _ASSERT_SMALL_NUMBER, \
                    'Provided P and Q do not satisfy the continuous-time' \
                    ' Lyapunov equation'

        self.P = P
        self.Qlyap = Q

    def fp(self, t, x, uhat, disturbance=None):
        if disturbance is not None:
            return self.plant.A @ x + self.plant.B @ uhat + \
                self.plant.E @ disturbance(t)
        else:
            return self.plant.A @ x + self.plant.B @ uhat

    def fc(self, t, x, yhat):
        return self.controller.A @ x + self.controller.B @ yhat

    def evaluate_run(self, sim_out, target_level=0.01):
        """Generate metrics of a simulation.

        Compute metrics of a simulation with respect to the current
        controller, such as time to reach the target Lyapunov level set
        and the number of samples to do so.

        Parameters
        ----------
        sim_out : dict
            As returned by a ETC/STC simulation

        Returns
        -------
        time_to_reach : float
            Time it took to reach level zero
        sample_count : int
            Number of samples before reaching level zero
        """
        n = len(sim_out['t'])
        for i in range(n):
            x = sim_out['xphat'][:, i]
            if x @ self.P @ x <= target_level:
                break
        else:
            raise ETCError('State never reached target Lyapunov value:',
                           f' Target is {target_level}, last value was'
                           f' {x @ self.P @ x}.')

        return sim_out['t'][i], sum(sim_out['sample'][:i+1])

    def evaluate(self, N=1000, T=2, seed=None, initial_level=10,
                 target_level=0.01, **kwargs):
        """Evaluate the controller's performance.

        Compute metrics for N randomly generated initial conditions.
        The initial conditions satisfy x(0)'Px(0) = V_max.

        Parameters
        ----------
        N : int, optional
            Number of simulations. The default is 1000.
        T : float, optional
            Max time of each simulation. The default is 2.
        seed : int, optional
            Seed for random number generation. The default is None,
            thus rng is not called.
        initial_level : float, optional.
            The desired initial value of the Lyapunov function. The
            default is 10.
        target_level : float, optional.
            The target value of the Lyapunov function. The default is
            0.01
        **kwargs : optional.
            Additional parameters for the simulation

        Returns
        -------
        time_to_reach_array : np.array(float)
            Time it took to reach level zero for each simulation.
        sample_count_array : np.array(int)
            Number of samples before reaching level zero for each
            simulation.
        """
        nx = self.plant.nx
        if seed is not None:
            random.seed(seed)

        x0_array = random.random_sample((N, nx))

        # Preallocate the output variables
        time_to_reach_array = np.zeros(N)
        sample_count_array = np.zeros(N)

        # Time array
        t = np.arange(0, T, self.h)

        # Main loop
        for i in range(N):
            x0 = x0_array[i, :]
            x0 = normalize_state_to_lyapunov(x0, self.P, initial_level)
            xc0 = np.zeros((self.controller.nx,))
            # print(x0 @ self.P @ x0)
            sim_out = simulate_sample_and_hold_control(self, t, x0, xc0,
                                                       **kwargs)
            try:
                time_to_reach, sample_count = \
                    self.evaluate_run(sim_out, target_level=target_level)
            except ETCError as e:
                print(x0)
                raise e
            time_to_reach_array[i] = time_to_reach
            sample_count_array[i] = sample_count

        return time_to_reach_array, sample_count_array



'''
    PERIODIC CONTROL CLASSES
'''


class PeriodicController(SampleAndHoldController):
    """Periodic Controller."""
    is_discrete_time = True

    def __init__(self, plant: Plant, controller: Controller, h=None):
        if not controller.is_discrete_time:
            if controller.is_dynamic:
                raise ETCError('Dynamic controller needs to be discrete-time')
            if h is None:
                raise ETCError('h must be provided if controller is not'
                               ' discrete-time.')
        super().__init__(plant, controller)

        if h is None:
            self.h = controller.h
        else:
            self.h = h
            self.controller.h = h

    def trigger(*args, **kwargs):
        return True


class PeriodicLinearController(PeriodicController,
                               LinearSampleAndHoldController):
    """Periodic Linear Controller of Linear Plant."""

    def __init__(self, plant: Plant, controller: Controller, h=None,
                 P=None, Q=None):
        PeriodicController.__init__(self, plant, controller, h)
        LinearSampleAndHoldController.__init__(self, plant, controller, P, Q)

        # Get discrete-time matrices
        nx = plant.nx
        nu = plant.nu
        s = la.expm(np.block([[plant.A, plant.B],
                              [np.zeros((nu, nx + nu))]])*h)
        self.Ad = s[0:nx, 0:nx]
        self.Bd = s[0:nx, nx:nx+nu]
        self.Phi = self.Ad + self.Bd @ controller.K


'''
    UTILITY PERIODIC FUNCTIONS
'''


def most_economic_periodic(plant: LinearPlant,
                           controller: LinearController,
                           h_step: float, h_max: float):
    h_best = 0
    decay = 1
    for h in np.arange(h_step, h_max, h_step):
        p = PeriodicLinearController(plant, controller, h)
        a = np.max(np.abs(la.eig(p.Phi)[0]))
        if a < decay:
            decay = a
            h_best = h
    if decay >= 1:
        raise ETCError('No stable periodic implementation for given'
                       ' parameters. Try reducing h_step and h_max.')
    return h_best, np.log(decay)/h_best


def tight_lyapunov(pc: PeriodicLinearController):
    """Compute a tight Lyapunov matrix for a periodic controller.

    Parameters
    ----------
    pc : PeriodicLinearController
        The periodic linear controller

    Raises
    ------
    ETCError
        If it fails to find a tight one.

    Returns
    -------
    Pl : np.array
        The output Lyapunov matrix
    lbd_max : TYPE
        The maximum decay associated with it.
    works_for_ct : TYPE
        True if it this Lyapunov matrix also satisfies the Lyapunov
        equation for the continuous-time implementation.

    """
    _MULTIPLIER = 1.0001

    nx = pc.plant.nx
    factor = 1

    Adcl = pc.Phi
    Acl = pc.plant.A + pc.plant.B @ pc.controller.K
    rho_max = np.abs(np.max(la.eig(Adcl)[0]))**2
    Pvar = cvx.Variable(shape=(nx, nx), PSD=True)
    objective = 0  # cvx.norm(Pvar)
    constraints = [Pvar >> np.eye(nx),
                   Adcl.T @ Pvar @ Adcl << factor * rho_max * Pvar,
                   Acl.T @ Pvar + Pvar @ Acl << 0]  # Works in c.t.
    solving = True
    works_for_ct = True
    attempts = 0
    while solving and attempts <= _MAX_TRIES_TIGHT_LYAPUNOV:
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve(eps=1e-6, max_iters=10000)
        if 'inaccurate' in prob.status:
            attempts += 1
        if prob.status == 'infeasible':
            if len(constraints) <= 2:
                attempts += 1
                factor *= _MULTIPLIER
                constraints[1] = Adcl.T @ Pvar @ Adcl \
                    << factor * rho_max * Pvar
            constraints = constraints[:2]
            works_for_ct = False
        else:
            solving = False
    if Pvar.value is None:
        raise ETCError('Tight Lyapunov failed,'
                       ' problem may be ill-conditioned.')

    Pl = Pvar.value
    Pl = (Pl + Pl.T)/2
    Ql = -(Acl.T @ Pl + Pl @ Acl)
    
    lbd_max = min(np.real(la.eig(Ql, Pl)[0]))
    lbd_max_dt = -np.log(rho_max)/pc.h
    lbd_max = max(lbd_max, lbd_max_dt)
    
    return Pl, lbd_max, works_for_ct


'''
    ETC CLASSES
'''


class ETC(SampleAndHoldController):
    threshold = 0
    is_dynamic = False
    must_trigger_at_first_crossing = True  # By default, most of them do

    def trigger(self, dt=None, x=None, xhat=None, y=None, yhat=None, u=None,
                uhat=None, t=None, *args):
        return False


class LinearETC(ETC, LinearSampleAndHoldController):
    pass


class LinearQuadraticETC(LinearETC):
    Qbar = np.array([])
    Qbar1 = np.array([])
    Qbar2 = np.array([])
    Qbar3 = np.array([])


class LinearPETC(LinearETC):
    kmin: int
    kmax: int
    h: float
    is_discrete_time = True
    triggering_function_uses_output = False
    triggering_is_time_varying = False

    def __init__(self, plant: LinearPlant,
                 controller: LinearController,
                 h=None, P=None, Q=None, kmin=1, kmax=50):
        super().__init__(plant, controller, P, Q)

        assert controller.is_discrete_time, \
            'Controller needs to be discrete-time'

        self.kmin = kmin
        if h is None:
            h = controller.h
        self.h = h
        if kmax is not None:
            self.kmax = kmax
        else:
            self.kmax = 50

        '''If provided P matrix is not a valid continuous-time Lyapunov
        matrix, for PETC it also suffices that it is a valid discrete-time
        Lyapunov matrix. The same goes even if the continuous-time controller
        is unstable. (TBP/TBC)
        '''
        nx = plant.nx
        p = PeriodicLinearController(plant, controller, h=h*kmin, P=P)
        controller.h = h  # Revert back because the statement above changed it
        Phi = p.Phi

        # Get/make Lyapunov info
        P = self.P
        if not is_positive_definite(P):
            if Q is None:
                Q = np.eye(nx)
            else:
                assert (Q == Q.T).all() and is_positive_definite(Q), \
                    'Q must be symmetric positive definite'
            P = ct.dlyap(Phi.T, Q)
            assert (P == P.T).all() and is_positive_definite(P), \
                'Closed-loop system is not stable'
        else:  # P is good in continuous time
            assert (P == P.T).all() and is_positive_definite(P), \
                'Provided P matrix is not symmetric positive definite'
            if Q is None:
                Q = -(Phi.T @ P @ Phi - P)
                assert is_positive_definite(Q), \
                    'Provided P matrix is not a Lyapunov matrix'
            else:
                assert (Phi.T @ P @ Phi - P == -Q).all(), \
                    'Provided P and Q do not satisfy the discrete-time' \
                    ' Lyapunov equation'
        self.P = P
        self.Qd = Q  # Qd for the discrete-time
        self.Phi = Phi
        self.Ad = p.Ad
        self.Bd = p.Bd


class LinearQuadraticPETC(LinearQuadraticETC, LinearPETC):
    def check_stability_impulsive(self, P = None, rho = None):
        """Check GES using the impulsive method of [1]
        
        Check global exponential stability of the closed-loop PETC
        by using Corollary III.3 in [1], which is based on an impulsive-
        systems modeling method. Using their remarks, it is the pair
        of LMIS, for some mu_i >= 0,
        
        exp(-2*rho*h) P + (-1)^i mu_i Q + A_i' Q A_i >= 0, i in {1,2}   (1)
        
        where
        
        A_1 = [A + BK,  0; I,  0],
        A_2 = [A,  BK; 0,  I].
        
        Note that, since lambda = exp(-2*rho*h) > 0, we can divide (1) by
            lambda, rename mu_i/lambda as nu_i and 1/lambda > 0 as alpha:
            
        P + (-1)^i nu_i Q + alpha A_i' Q A_i >= 0, i in {1,2}.   (2)
        
        This is linear on P, mu, and alpha.
        
        If P is given but rho is not, it minimizes rho to satisfy (1).
        If rho is given but P is not, it tries to find P that verifies (1).
        If both are given, it simply checks feasibility of (1), and, if it
            is not, the pair (None, None) is returned.
        If neither is given, it tries to find both rho and P simultaneously
            by solving (2).

        Parameters
        ----------
        P : np.array, optional
            The Lyapunov matrix for V(x) = x'Px. The default is None.
        rho : float, optional
            The desired GES decay rate. The default is None.

        Returns
        -------
        float
            The GES decay rate. None if none is found.
        np.array
            The Lyapunov matrix. None if none is found.
            
        [1] Heemels, W.P.M.H., Donkers, M.C.F., and Teel, A.R.
            (2013). Periodic event-triggered control for linear systems.
            IEEE Transactions on Automatic Control, 58(4),
            847--861.
        """
            
        if not self.is_state_feedback:
            raise ETCError('Output feedback not yet implemented.')
            
        raise ETCError('Future implementation.')
        
        n = self.plant.nx
        A1 = np.block([
            [self.Ad + self.Bd @ self.controller.K, self.zeros((n,n))],
            [np.eye(n),                             np.zeros((n,n))]
            ])
        A2 = np.block([
            [self.Ad,         self.Bd @ self.controller.K],
            [np.zeros((n,n)), np.eye(n)]
            ])        
        Q = self.Qbar
        
        if P is None and rho is None:
            P = cvx.Variable((n, n), PSD=True)
            nu1 = cvx.Variable(pos=True)
            nu2 = cvx.Variable(pos=True)
            alpha = cvx.Variable(pos=True)
            constraints = [P - nu1*Q >> 0]
            prob = cvx.Problem(cvx.Minimize(0), constraints)
            prob.solve(eps=1e-6, max_iters=10000)
        elif P is None:
            return 0.1
        elif rho is None:
            return np.eye(self.nx)
        else:
            return
        
    def check_stability_pwa(self, eps=1e-3):
        """Check GES using the piecewise affine method of [1]
        
        Check global exponential stability of the closed-loop PETC
        by using Theorem III.4 in [1], which is based on a piecewise-
        affine model. It is the group of LMIs, for some P > 0,
        alpha_ij >= 0, beta_ij >= 0, kappa_i >= 0, i,j in {1,2},
        
        $$ e^{(-2*\rho*h)}P_i - A_i^\top P_jA_i + (-1)^i alpha_ij Q  $$
            + (-1)^j beta_ij A_i'QA_i >= 0,     i,j in {1,2}    $$     (1)
        P_i + (-1)^ikappa_iQ > 0,    i in {1,2}                      (2)
         
        where
        
        A_1 = [A + BK,  0; I,  0],
        A_2 = [A,  BK; 0,  I].
        
        This method performs a bisection algorithm on lambda 
        := exp(-2*rho*h) to find the smallest value that satisfies
        (1) and (2).

        Parameters
        ----------
        eps : float, optional
            The precision for lambda in the bisection algorithm. The
            default is 1e-3.
            
        Returns
        -------
        float
            The GES decay rate, or 1 if it is not GES.
        (np.array, np.array)
            The Lyapunov matrices. None if system is not GES
            
        [1] Heemels, W.P.M.H., Donkers, M.C.F., and Teel, A.R.
            (2013). Periodic event-triggered control for linear systems.
            IEEE Transactions on Automatic Control, 58(4),
            847--861.
        """
            
        if not self.is_state_feedback:
            raise ETCError('Output feedback not yet implemented.')
        
        n = self.plant.nx
        A = {}
        A[1] = np.block([
            [self.Ad + self.Bd, np.zeros((n,n))],
            [np.eye(n),         np.zeros((n,n))]
            ])
        A[2] = np.block([
            [self.Ad,         self.Bd],
            [np.zeros((n,n)), np.eye(n)]
            ])
        Q = self.Qbar
        
        # CVX variables
        alpha = {(i,j): cvx.Variable(pos=True) for i in range(1,3) 
                 for j in range(1,3)}
        beta = {(i,j): cvx.Variable(pos=True) for i in range(1,3) 
                for j in range(1,3)}
        kappa = {i: cvx.Variable(pos=True) for i in range(1,3)}
        P = {i: cvx.Variable((2*n, 2*n), PSD=True) for i in range(1,3)}
        
        # CVX constraints : make a function of the externally defined lbd
        def make_constraints(lbd):
            con = []
            for i in range(1,3):
                for j in range(1,3):
                    con.append(lbd*P[i] - A[i].T @ P[j] @ A[i]
                               + ((-1)**i)*alpha[(i,j)]*Q
                               + ((-1)**j)*beta[(i,j)]*(A[i].T @ Q @ A[i])
                               >> 0)  # Eq. (1))
                con.append(P[i] + (-1)**i * kappa[i]* Q   # Eq. (2)
                           >> _LMIS_SMALL_IDENTITY_FACTOR*np.eye(2*n))
            return con
                
        # Start bisection algorithm: get extreme points
        a = 0
        b = 1
        
        # For b = 1, if GES then it must be feasible
        con = make_constraints(b)
        prob = cvx.Problem(cvx.Minimize(0), con)
        prob.solve()
        if 'infeasible' in prob.status:
            return 1, None
        Pout = (p.value for p in P)
        
        # For a = 0, if it is feasible then this is a deadbeat controller.
        # Can't be better then this
        con = make_constraints(a)
        prob = cvx.Problem(cvx.Minimize(0), con)
        prob.solve()
        if 'optimal' in prob.status:
            return 0, (p.value for p in P)
        
        # Now we should have b = 1 feasible and a = 0 infeasible. Start
        # bisection algorithm
        while b-a > eps:
            c = (a+b)/2
            con = make_constraints(c)
            prob = cvx.Problem(cvx.Minimize(0), con)
            prob.solve()
            if 'optimal' in prob.status:
                b = c
                Pout = (p.value for p in P)  # Store output P matrices
            elif 'infeasible' in prob.status:
                a = c
            else:
                warnings.warn(f'{prob.status}: TOL is {b-a}')
                break
        
        return -np.log(b)/2/self.h, Pout
    
        
    def check_stability_kmax(self, P):
        """Check stability based on simple S-procedure rule"""
        
        if not self.is_state_feedback:
            raise ETCError('Output feedback not yet implemented.')
            
        if self.kmax is None:
            raise ETCError('This method needs kmax')
        
        kmax = self.kmax
        Ad = self.Ad
        Bd = self.Bd
        K = self.controller.K
        n = self.plant.nx
        Q = self.Qbar
        
        mu = cvx.Variable((kmax,kmax),pos=True)  # S-procedure variable
        a = cvx.Variable(pos=True)  # Sample-wise contraction rate of V
        
        # Transition matrix at k=1
        M = Ad + Bd
        
        Nlist = []  # Store N matrices from 1 to k
        con = []  # constraint list
        # with the case oh k=1
            
        for k in range(kmax):
            MI = np.block([[M],[np.eye(n)]])
            N = MI.T @ Q @ MI
            con.append(-mu[k,k] * N.copy()
                       + sum(mu[i,k] * n.copy() for i,n in enumerate(Nlist))
                       - M.T.copy() @ P @ M.copy()
                       + a * P >> 0) 
            # Iterate for next M
            Nlist.append(N.copy())
            M = Ad @ M + Bd
            
        prob = cvx.Problem(cvx.Minimize(a), con)
        prob.solve()
        return a.value


class DirectLyapunovPETC(LinearQuadraticPETC):
    """ Check Lyapunov direcly.
    
    The triggering condition is based on the continuous-time Lyapunov
    condition 
             d(x'Px)/dt <= -rho*(x'Qx),     (1)
    where P and Q are the Lyapunov
    matrices of the continuous-time closed-loop system and 0 <= rho < 1
    is the margin parameter. The triggering can be either predictive or
    not. When not, it simply implements the condition (1). When
    predictive, it computes x(k+1) = Ad @ x(k) + Bd @ K @ xhat(k) as the
    estimate of the next checking time state value. Them it uses x(k+1)
    in (1) to determine the triggering condition."""

    def __init__(self, plant, controller, P=None, Q=None, rho=0.5,
                 threshold=0, h=None, kmin=1, kmax=None,
                 predictive=False):
        super().__init__(plant, controller, P=P, kmin=kmin, kmax=kmax)
        self.A = plant.A
        self.B = plant.B
        self.K = controller.K
        # self.P = P
        self.Q = Q
        if Q is None:
            self.Q = self.Qlyap
        self.rho = rho
        self.predictive = predictive
        if predictive:
            nx = plant.nx
            Abar = la.expm(np.block([[self.A, self.B @ self.K],
                              [np.zeros((nx, 2*nx))]]) * self.h)
            self.Abar = Abar
            self.Ad = Abar[:nx, :nx]
            self.Bd = Abar[:nx, nx:]
        self._generate_Qbar()

    def _generate_Qbar(self):
        nx = self.plant.nx
        Qbar1 = self.A.T @ self.P + self.P @ self.A + self.rho * self.Q
        Qbar2 = self.P @ self.B @ self.K
        Qbar3 = np.zeros((nx, nx))
        self.Qbar1 = Qbar1
        self.Qbar2 = Qbar2
        self.Qbar3 = Qbar3
        self.Qbar = np.block([[Qbar1, Qbar2], [Qbar2.T, Qbar3]])
        if self.predictive:
            self.Qbar = self.Abar.T @ self.Qbar @ self.Abar
            self.Qbar1 = self.Qbar[:nx,:nx]
            self.Qbar2 = self.Qbar[:nx,nx:]
            self.Qbar3 = self.Qbar[nx:,nx:]

    def trigger(self, dt, x, xhat, y=None, yhat=None,
                u=None, uhat=None, t=None, *args):
        if self.predictive:
            x = self.Ad @ x + self.Bd @ xhat
        z = self.A @ x + self.B @ self.K @ xhat
        return 2 * (z @ self.P @ x) + self.rho * (x @ self.Q @ x) > 0 \
            or dt >= (self.kmax - 1/2) * self.h

