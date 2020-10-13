#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:12:45 2018

original author: @ggleizer
modified by @asamant to include only relevant parts.

"""

import numpy as np
from numpy import random
import scipy.linalg as la
import linearetc as etc
from collections import defaultdict
import logging
import time
from collections import namedtuple
from etcutil import QuadraticForm, sdr_problem, QuadraticProblem
from tqdm import tqdm
import itertools
import warnings
from joblib import Parallel, delayed
import multiprocessing
import sympy


SMALL_NUMBER = 1e-7
__DEBUG__ = False
__TEST__ = False
ABSTRACTION_NO_COSTS = False
_RELATIVE_COST_TOLERANCE = 0.001
_QCQP_TOLERANCE = 1e-4
_SSC_MAX_ITERS = 30000  # Reaching maximum number of iterations is bad.
_SSC_MAX_ATTEMPTS = 3  # Number of times to try the SDP problem in inaccurate.
# Should avoid it at all cost. Increase this number if inaccurate results
# are obtained.
LEVEL_SMALL_NUMBER = 1e-6
IFAC_WC_2020 = True
CDC_2020 = False
NUM_CORES = max(1, multiprocessing.cpu_count()-1)

Partition = namedtuple('Partition', ['k', 'v'])


class ETCAbstractionError(etc.ETCError):
    pass


class TrafficModel:
    plant: etc.Plant
    controller: etc.Controller
    transition = {}  # Discrete-state transition matrices
    is_discrete_time: bool
    has_level_sets = False

    def region_of_state(self, x, **kwargs):
        """Determine which region state x belongs.

        Parameters
        ----------
        x: numpy.array
            Input state

        Returns
        -------
        int
            Region index (key of self.Q)
        """
        return 0

    def level_of_state(self, x: np.array):
        """ Determines the Lyapunov level where state x belongs.


        Parameters
        ----------
        x : np.array
            Input state

        Returns
        -------
        int
            The level index

        """
        return 0


class TrafficModelETC(TrafficModel):
    trigger: etc.ETC


class TrafficModelPETC(TrafficModelETC):
    is_discrete_time = True
    M = {}  # Transition matrices to state
    N = {}  # Transition matrices to input and output
    Q = {}  # Quadratic forms for the cones
    kmin = 1
    kmax = None  # Maximum triggering time
    kmaxextra = 1000  # Maximum time scheduler forces trigger
    n = 0  # dimension of the data-space (where the cones live in)
    predecessors = {}  # precedence (subset) relation for quadratic forms
    dP = []  # Matrices for Lyapunov function decay after k samples
    cost = {}  # cost of transition Lyapunov-wise
    transition = {}
    complete_cost = {}
    probability_region = {}
    probability_transition = {}

    def __init__(self, trigger: etc.LinearQuadraticPETC, kmax_extra=None,
                 no_costs=False, consider_noise=False, mu_threshold=0.0,
                 min_eig_threshold=0.0, reduced_actions=False,
                 early_trigger_only=False, max_delay_steps=0,
                 number_samples=10000, depth=1, etc_only=False, end_level=0.01,
                 bisim=False, solver='sdr', stop_around_origin=True):
        self.trigger = trigger
        self.plant = trigger.plant
        self.controller = trigger.controller
        self.consider_noise = consider_noise
        self.mu_threshold = mu_threshold
        self.min_eig_threshold = min_eig_threshold
        self.no_costs = no_costs
        self.reduced_actions = reduced_actions
        self.early_trigger_only = early_trigger_only
        self.max_delay_steps = max_delay_steps
        self.depth = depth
        self.etc_only = etc_only  # Only natural ETC triggers
        self.end_level = end_level
        self.bisim = bisim
        self.solver = solver
        self.stop_around_origin = stop_around_origin
        
        self.P = trigger.P

        if trigger.kmax is not None:
            self.kmax = trigger.kmax
        if trigger.kmin is not None:
            self.kmin = trigger.kmin

        self.kmaxextra = kmax_extra

        logging.info('Building cones for traffic model')
        self._build_cones()
        logging.info('kmin=%d, kmax=%d', self.kmin, self.kmax)
        if mu_threshold > 0.0 or self.reduced_actions \
                or self.early_trigger_only:
            logging.info('Reducing number of cones')
            self._reduce_cones()
            logging.info('Reduced from %d to %d regions',
                         self.kmax - self.kmin + 1, len(self.Q))
        logging.info('Building precedence relation for quadratic forms')
        self._build_precedence_relation()
        count_non_ideal = sum(1 for x, p in self.predecessors.items()
                              for y in self.Q if y < x
                              and y not in p)
        logging.info('Average number of preceding regions not subset'
                     ' of the next region: %d',
                     count_non_ideal/len(self.Q))
        
        self._prepare_transition()
        # Stores dP[k] such that x' dP[k] x is the actual decrease in the
        # Lyapunov (cost) function after [k] instants
        self.K = set(self.Q.keys())
        self.regions = set((k,) for k in self.K)
        self._minimal = set()

        new_regions = self.regions
        for d in range(self.depth):
            self.regions = new_regions
            logging.info('Computing terminal Lyapunov level sets')
            self._compute_costs()
            logging.info('Reducing region string lengths')
            self._reduce_regions()
            print('Reduced regions:\n' + str(self.regions))
            if self._minimal == self.regions:
                logging.info('Refinement is finished')
                break
            
            logging.info('Building transition model')
            self._build_transition()
            
            # if d < self.depth - 1:
            new_regions = self._split_regions()
            new_regions.update(x for x in self._minimal)

    
    def _prepare_transition(self):
        P = self.trigger.P
        M = self.M
        Q = self.Q
        
        dP = {k: m.T @ P @ m - P for k, m in M.items()}
        self.dP = dP

        # Quadratics MQM(i,j) define if state enters cone j after i+1 steps
        MQM = {(i, j): mi.T @ qj @ mi
               for i, mi in M.items() for j, qj in Q.items()}
        # When threshold is zero, we can also normalize these matrices
        MQM = {x: mqm/la.norm(mqm) for x, mqm in MQM.items()}
        self.MQM = MQM
        
    def _split_regions(self):
        """Split regions one step with reachability.
        
        If a region represents a string of sampling times i1i2...in, after
        spliting a region represents a string of sampling times i1i2...i(n+1).
        
        Returns
        -------
        set
            Its elements are tuples containing the possible sequences of
            sampling times generated by the PETC of length n+1, where n is
            the length of the string of the current abstraction.
        """
        
        # Form concatenated substrings
        out_set = set()
        for (i,k),j_list in self.transition.items():
            if k == i[-1]:
                for j in j_list:
                    # n = len(i)
                    ij = i+j
                    if self._substring_exists(ij):
                        out_set.add(ij)
                    # for m in range(n):
                    #     if ij[m:n+1+m] not in out_list:
                    #         # print(ij[m:n+1+m])
                    #         out_list.append(tuple(ij[m:n+1+m]))
        return out_set
    
    def _substring_exists(self, ij):
        """Check if all substrings of ij exist in self.regions.
        
        Check all substrings of ij of length compatible with lengths in 
        self.regions. If one such substring does NOT exist, return False.
        """
        L = len(ij)
        for l in range(min(L, self.maxL), self.minL - 1, -1):
            for m in range(0, L-l):
                s = ij[m:m+l]  # This is the substring to be checked.
                for r in self.regions:
                    lr = len(r)
                    if lr < l and any(r == s[i:lr+i] for i in range(l-lr+1)):
                        break
                    if any(s == r[i:l+i] for i in range(lr - l + 1)):
                        break
                else:
                    return False
        return True
            
    def _compute_costs(self):
        self._compute_dP()
        old_cost = self.cost.copy()
        to_be_deleted = set()
        
        for i in tqdm(self.regions):
            try:
                if i not in old_cost:
                    self.cost[i] = 1 \
                                   + self._transition_cost(i, i, self.dP[i])[1]
            except ETCAbstractionError:
                to_be_deleted.add(i)
        self.regions.difference_update(to_be_deleted)
        self._initial = set(i for i,c in self.cost.items()
                             if c <= self.end_level and i in self.regions)
    
    def _reduce_regions(self):
        """Reduce each the string length of each region if it is terminal.
        

        Returns
        -------
        None.
        """
        
        # The cost dictionary will accumulate costs of every visited string
        # for computational reasons
        for r in tqdm(self._initial):
            self.regions.remove(r)
            while len(r) > 1:
                r_new = r[:-1]
                try:
                    cost = self.cost[r_new]
                except KeyError:
                    try:
                        dPr = self._dP_of_region(r_new)
                        cost = 1 + self._transition_cost(r_new, r_new, dPr)[1]
                        self.cost[r_new] = cost
                    except ETCAbstractionError as e:
                        if 'infeasible' == str(e)[-10:]:
                            # This region was infeasible all along
                            try:
                                self.cost.pop(r)
                            except KeyError:
                                pass
                            # self.cost.pop(r_new)
                            continue
                if cost > self.end_level:  # previous region was minimal
                    self.regions.add(r)
                    self._minimal.add(r)
                    break
                r = r_new
                    
        self._initial = set(i for i,c in self.cost.items()
                             if c <= self.end_level)
        self.regions.update()

    # time-varying Q for Relaxed PETC, checks just the condition if the
    # Lyapunov function exceeds the bound at the next time instant
    def _Q_time_var(self, k, h):
        nx = self.plant.nx
        ny = self.plant.ny
        nu = self.plant.nu
        nz = ny + nu

        M = np.block([[self.trigger.Ad, self.trigger.Bd @ self.controller.K]])
        Z = np.zeros((nx, nx))
        Pe = self.trigger.P*np.exp(-self.trigger.lbd*k*h)
        Qbar = M.T @ self.trigger.P @ M - np.block([[Z, Z], [Z, Pe]])
        Qbar1 = Qbar[:nx, :nx]
        Qbar2 = Qbar[:nx, nx:]
        Qbar3 = Qbar[nx:, nx:]
        # Qbar1 = self.trigger.P
        # Qbar2 = np.zeros((nx, nx))
        # Qbar3 = -self.trigger.P*np.exp(-self.trigger.lbd*k*h)
        self.Qbar1 = Qbar1
        self.Qbar2 = Qbar2
        self.Qbar3 = Qbar3
        # self.Q = np.block([[Qbar1, Qbar2], [Qbar2.T, Qbar3]])
        Qbar_yuyu = np.zeros((nz*2, nz*2))
        Qbar_yuyu[0:ny, 0:ny] = Qbar1
        Qbar_yuyu[0:ny, nz:nz+ny] = Qbar2
        Qbar_yuyu[nz:nz+ny, 0:ny] = Qbar2.T
        Qbar_yuyu[nz:nz+ny, nz:nz+ny] = Qbar3

        return Qbar_yuyu

    def _build_cones(self):
        ''' Transition matrices '''
        # Compute transition matrix M(\dk) such that
        # zeta(k+\dk) = M(\dk)[xp;xc;y]

        p = self.plant
        c = self.controller
        t = self.trigger

        # TODO: think about how to treat fundamental PETC and
        # abstraction times differently
        h_abs = t.h

        nxp = p.nx
        nxc = c.nx
        ny = p.ny
        nu = p.nu
        nz = ny + nu

        # First the more obvious CE: [y;u] = CE[xp;xc;y]
        CE = np.block([np.zeros((nz, nxp)),
                       np.block([[np.zeros((ny, nxc)), np.eye(ny)],
                                 [c.C, c.D]])])

        # Fundamental transition matrices
        Abar = np.block([[p.A, p.B], [np.zeros((nu, nxp+nu))]])
        Phibar = la.expm(Abar*h_abs)  # Think about this h_abs
        Phip = Phibar[0:nxp, 0:nxp]
        Gammap = Phibar[0:nxp, nxp:]

        # Loop to compute Mks
        Phipk = Phip
        Gammapk = Gammap
        Ack = c.A
        Bck = c.B

        Mlist = []
        Nlist = []
        Qlist = []
        maxeig = []
        mineig = []

        if t.must_trigger_at_first_crossing:
            kmax = None
        else:
            kmax = self.kmax

        if not t.triggering_is_time_varying:
            if t.triggering_function_uses_output:
                Qbar_yuyu = t.Qbar
            else:
                Qbar_yuyu = np.zeros((nz*2, nz*2))
                Qbar_yuyu[0:ny, 0:ny] = t.Qbar1
                Qbar_yuyu[0:ny, nz:nz+ny] = t.Qbar2
                Qbar_yuyu[nz:nz+ny, 0:ny] = t.Qbar2.T
                Qbar_yuyu[nz:nz+ny, nz:nz+ny] = t.Qbar3

        for i in tqdm(range(0, t.kmax)):  # TODO: Think about this kmax_abs
            # Transition matrices from [xp, xc, y] after k=i+1 steps
            # [xip(t+kh), xic(t+kh)].T = M(k)[xp, xc, y].T
            # [psi(t+kh); ups(t+kh)].T = N(k)[xp, xc, y].T
            M1 = np.block([Phipk, Gammapk @ c.C, Gammapk @ c.D])
            M2 = np.block([np.zeros((nxc, nxp)), Ack, Bck])
            N1 = p.C @ np.block([Phipk, Gammapk @ c.C, Gammapk @ c.D])
            N2 = np.block([np.zeros((nu, nxp)), c.C @ Ack, c.C @ Bck + c.D])
            M = np.block([[M1], [M2]])
            N = np.block([[N1], [N2]])

            # Update transition matrices
            Phipk = Phip @ Phipk
            Ack = c.A @ Ack
            Gammapk = Gammap + Phip @ Gammapk
            Bck = c.B + c.A @ Bck

            # Remember: k := i+1
            # Q(k): defines the cone: [xp;xc;y]'Q(k)[xp;xc;y] > 0:
            # trigger at t+kh (or before)
            if t.triggering_is_time_varying:
                Q = np.block([N.T, CE.T]) @ self._Q_time_var(i+1, h_abs) \
                    @ np.block([[N], [CE]])
            else:
                Q = np.block([N.T, CE.T]) @ Qbar_yuyu @ np.block([[N], [CE]])
            # No noise --> no need to separate xp from y, there is redundancy
            if not self.consider_noise:
                # y = Cxp - colapse to only xp-dependency
                IIC = np.zeros((nxp+nxc+ny, nxp+nxc))
                IIC[:-ny, ] = np.eye(nxp+nxc)
                IIC[nxp+nxc:, :nxp] = p.C
                # [xp; xc] = IIC [xp; xc; y]
                Q = IIC.T @ Q @ IIC
                M = M @ IIC
                N = N @ IIC
                # Normalize Q
                Q = Q/la.norm(Q)

            # print(Q)
            Qlist.append(Q)
            Mlist.append(M)
            Nlist.append(N)
            lbd, _ = la.eig(Q)
            maxeig.append(max(np.real(lbd)))
            mineig.append(min(np.real(lbd)))
            # At this point, all states have triggered
            if mineig[-1] > 0 and kmax is None:
                kmax = i+1
                if self.early_trigger_only:
                    self.kmaxextra = kmax
                    break
            if kmax is not None and mineig[-1] <= 0:
                kmax = None
        # end for
        # print([x > -self.min_eig_threshold for x in mineig])
        # print(mineig)
        if kmax is None:  # maximum triggering time prevented finding Q(k) > 0
            kmax = t.kmax
        # Erase Qs up to kmaxextra

        Qlist = Qlist[:kmax]
        if mineig[-1] > -self.min_eig_threshold:
            # Retroactive search of the last k: mineig[k] <= thresh
            for i in range(kmax-1, -1, -1):
                if mineig[i] <= -self.min_eig_threshold:
                    break
            kmax = i+2
            Qlist = Qlist[:kmax]

        try:
            kbeg = next(i for i, l in enumerate(maxeig) if l > 0) + 1
        except StopIteration:
            print(mineig)
            raise ETCAbstractionError(
                f'No triggering would occur up to {self.kmax}-th iteration')

        ''' NEED TO FIGURE THIS OUT ONCE AND FOR ALL '''
        # try:
        #     kend = next(i for i, l in enumerate(mineig) if l > 0) + 1
        # except StopIteration:
        #     kend = kmax
        kend = kmax

        # Add data to class
        self.kmin = max(self.kmin, kbeg)
        self.kmax = kend
        self.n = Qlist[0].shape[0]
        if self.kmaxextra is None:
            self.kmaxextra = kend + self.max_delay_steps

        self.M = {i+1: m for i, m in enumerate(Mlist) if i+1 <= self.kmaxextra}
        self.N = {i+1: n for i, n in enumerate(Nlist) if i+1 <= self.kmaxextra}
        self.Q = {i+1: q for i, q in enumerate(Qlist)
                  if i+1 >= self.kmin and maxeig[i] > 0}
        self.Q[kend] = np.eye(Qlist[0].shape[0])  # For the last, all states
        # should trigger
        # Normalizing Q requires adjusting non-zero threshold
        if t.threshold is not None and t.threshold != 0:
            raise ETCAbstractionError('Method is not prepared for non-zero'
                                      'threshold in triggering function')

    def _reduce_cones(self):
        new_Q = {self.kmin: self.Q[self.kmin]}
        Q1 = QuadraticForm(self.Q[self.kmin])
        for k in tqdm(sorted(self.Q)):
            if k > self.kmin:
                Q2 = QuadraticForm(self.Q[k])
                if Q1.difference_magnitude(Q2) >= self.mu_threshold:
                    # Add to reduced dictionary
                    new_Q[k] = self.Q[k]
                    # Update current Q
                    Q1 = Q2
        self.Q = new_Q
        if self.reduced_actions:
            self.M = {k: m for k, m in self.M.items() if k in self.Q}
            self.N = {k: n for k, n in self.N.items() if k in self.Q}
        if self.early_trigger_only:
            self.M = {k: m for k, m in self.M.items() if k <= max(self.Q)}
            self.N = {k: n for k, n in self.N.items() if k <= max(self.Q)}

    def _build_precedence_relation(self):
        # Computes in advance which cones for 1:k-1 are contained in cone for k
        # This will reduce the size of the QCQP problem

        Q = self.Q
        # Check quadratic forms ordering
        q = {k: QuadraticForm(x) for k, x in Q.items()}
        predecessors = {}
        for i, Qi in q.items():
            if i == 1:
                predecessors[i] = {}
            if i == 2:
                predecessors[i] = {j for j, Qj in q.items()
                                   if i > j and Qi > Qj}
            else:
                predecessors[i] = set()
                candidates = sorted([j for j in q if j < i])
                while len(candidates) > 0:
                    j = candidates.pop()
                    if Qi > q[j]:
                        predecessors[i].add(j)
                        # include predecessors[j] in predecessors of i
                        predecessors[i].update(predecessors[j])
                        for k in predecessors[j]:
                            if k in candidates:
                                candidates.remove(k)
        # nonpredecessors = {i: {j for j in range(0,i)
        #                        if j not in predecessors[i]}
        #                    for i in predecessors}
        self.predecessors = predecessors
        # self.nonpredecessors = nonpredecessors

    def _dP_of_region(self, r):
        l = len(r)
        m = self._M_prod(r, l)
        return m.T @ self.trigger.P @ m - self.trigger.P
        
    def _compute_dP(self):
        # Compute dP matrices of regions
        self.dP = {r: self._dP_of_region(r) for r in self.regions}
    
    def _build_transition(self):
        # Builds a list of transitions where l[(i,j)] is the list of cones
        # reachable from region i (related to instant i+1) after j+1 steps
        M = self.M
        dP = self.dP
        R = self.regions
        
        # Store min and max lengths for later use
        self.minL = min(len(x) for x in self.regions)
        self.maxL = max(len(x) for x in self.regions)

        # Reachability problem: is there x in region i that reaches region j
        # after k+1 sample?
        # Cost comes almost for free here. Use cost computation instead of pure
        # feasibility
        transition = {}
        # (complete_cost[((i,k),j)])
        complete_cost = {}  # for costs depending on reached set j.
        
        nIK = len(R)*len(M)
        for i, k in tqdm(itertools.product(R, M), total=nIK):
            k_tuple = (k,)
            if type(i) is not tuple:
                i = (i,)
            if self.early_trigger_only:
                if k > i[0]:
                    continue
            if self.max_delay_steps > 0:
                if k > i[0] + self.max_delay_steps:
                    continue
            if self.etc_only:
                if k != i[0]:
                    continue
                k_tuple = i
            # New: if there is a terminal cost, don't perform reachability if
            # i is associated with a terminal string.
            if i in self._initial and k == i[0]:
                continue
                
            if not self.no_costs:
                dPk = dP[k]
            transition[(i, k)] = set()
            for j in R:
                if type(j) is not tuple:
                    j = (j,)
                if self.no_costs:
                    if self._reaches(i, j, k_tuple):
                        transition[(i, k)].add(j)
                else:  # TODO: Bi-simulation case
                    try:
                        logging.debug('Checing transition for %d --%d-> %d',
                                      i, k, j)
                        (cost_low, cost_up) \
                            = self._transition_cost(i, k, dPk, j)
                        if cost_low < -1 or cost_up < -1:
                            logging.debug(
                                'Cost of %d --%d-> %d broken: low=%g, up=%g',
                                i, k, j, cost_low, cost_up)
                            continue
                        if cost_low > cost_up:
                            cost_error = abs(1. - abs(cost_low/cost_up))
                            if cost_error > _RELATIVE_COST_TOLERANCE:
                                logging.debug(
                                    'Cost of %d --%d-> %d broken:'
                                    ' low=%g, up=%g',
                                    i, k, j, cost_low, cost_up)
                                continue
                            else:
                                logging.debug('Cost of %d --%d-> %d slightly'
                                              ' broken: low=%g, up=%g',
                                              i, k, j, cost_low, cost_up)
                            cost_low = (cost_low + cost_up)/2.
                            cost_up = cost_low
                        complete_cost[((i, k), j)] = (cost_low, cost_up)
                    except Exception as e:
                        if 'Relaxation problem status: infeasible' in str(e):
                            continue
                        else:
                            raise e
                    transition[(i, k)].add(j)

        self.transition = transition
        self.complete_cost = complete_cost

    def _M_prod(self, i_tuple, l):
        M = self.M
        if self.bisim:
            prod = sympy.Identity(self.n)
        else:
            prod = np.eye(self.n)
        for i in range(l):
            prod = M[i_tuple[i]] @ prod
        return prod
        
    # For building the QCQP problems
    def _add_constraints_for_region_i(self, i_tuple, con):
        # con += the set of constraints  for x \in R_i
        # Constraints related to the current region:
        #   x'Q(i)x > 0
        #   x'Q(s)x <= 0 for all s < i
        #   if p < s and p is in the predecessors of s,
        #      we do not need to include x'Q(p)x <= 0 in the list.
        # NEW: Using reachability for multiple step regions
        # Constraints related to region i1i2i3...iL
        #   x'Q(i1)x > 0
        #   x'Q(s)x <= 0 for all s < i1
        #   x'M(i1)'Q(i2)M(i1)x > 0
        #   x'M(i1)'Q(s)M(i1)x <= 0 for all s < i2
        #   ...
        #   x'(prod_n(M))'Q(in)prod_n(M)x > 0
        #   x'(prod_n(M))'Q(s)prod_n(M)x <= 0 for all s < in
        #      where prod(M) is M(i(L-1))M(i(L-2))...M(i1)

        i_list = sorted(self.Q)
        for l,i in enumerate(i_tuple):
            i_index = i_list.index(i)
            Mprod = self._M_prod(i_tuple, l)
            if i < i_list[-1]:
                MQM = Mprod.T @ self.Q[i] @ Mprod
                if not self.bisim:
                    MQM = MQM / max(1e-6, min(abs(la.eigvalsh(MQM))))
                con.add(QuadraticForm(- MQM.copy(), strict=True))
            if i >= i_list[1]:
                i_prev = i_list[i_index-1]
                MQM = Mprod.T @ self.Q[i_prev] @ Mprod
                if not self.bisim:
                    MQM = MQM / max(1e-6, min(abs(la.eigvalsh(MQM))))
                con.add(QuadraticForm(MQM.copy()))
                for p in i_list[:i_index-1]:
                    if p not in self.predecessors[i_prev]:
                        MQM = Mprod.T @ self.Q[p] @ Mprod
                        if not self.bisim:
                            MQM = MQM / max(1e-6, min(abs(la.eigvalsh(MQM))))
                        con.add(QuadraticForm(MQM.copy()))
        return con
    
    def _add_constraints_for_reaching_j_after_k(self, j_tuple, k_tuple, con):
        # con += the set of constraints  for M(k)x \in R_j

        j_list = sorted(self.Q)
        MK = self._M_prod(k_tuple, len(k_tuple))
        for l,j in enumerate(j_tuple):
            j_index = j_list.index(j)
            Mprod = self._M_prod(j_tuple, l)
            if j < j_list[-1]:
                MQM = MK.T @ Mprod.T @ self.Q[j] @ Mprod @ MK
                MQM = MQM / min(abs(la.eigvalsh(MQM)))
                con.add(QuadraticForm(-MQM.copy()))
            if j >= j_list[1]:
                j_prev = j_list[j_index-1]
                MQM = MK.T @ Mprod.T @ self.Q[j_prev] @ Mprod @ MK
                MQM = MQM / min(abs(la.eigvalsh(MQM)))
                con.add(QuadraticForm(MQM.copy()))
                # Trivially, if p subset s, it also holds for the MQM related 
                # cones
                for p in j_list[:j_index-1]:
                    if p not in self.predecessors[j_prev]:
                        MQM = MK.T @ Mprod.T @ self.Q[p] @ Mprod @ MK
                        MQM = MQM / min(abs(la.eigvalsh(MQM)))
                        con.add(QuadraticForm(MQM.copy()))
        return con

    # Potentially deprecated (can be used if costs are not wanted,
    # since it is probably faster)
    def _reaches(self, i, j, k):
        '''Reachability problem:
            is there x in region i that reaches region j after k+1 sample?
        '''
        n = self.n
        # kmin = self.kmin
        # kmax = self.kmax

        # assert i >= kmin and j >= kmin, 'Initial and final sets must be' \
        #                                 'greater or equal than kmin'
        # assert i <= kmax and j <= kmax, 'Time greater than kmax given'
        # assert k <= self.kmaxextra, 'k greater than kmaxextra given'

        logging.debug('Checking transition for ' + str((i, j, k)))
        
        # First: check if all substrings of the current length exist in region
        # set
        if i == k:
            ij = i + j
            if not self._substring_exists(ij):
                return False
        
        con = self._add_constraints_for_region_i(i, set())
        con = self._add_constraints_for_reaching_j_after_k(j, k, con)
        # print(len(con))

        # Build and solve QCQP problem (SDR)
        # prob = sdr_problem(QuadraticForm(np.eye(n)), con, unit_ball=True)
        prob = sdr_problem(QuadraticForm(np.zeros((n, n))),
                           con, unit_ball=True)

        n_tries = 0
        while n_tries < _SSC_MAX_ATTEMPTS:
            prob.solve(eps=_QCQP_TOLERANCE, max_iters=_SSC_MAX_ITERS,
                          verbose=__TEST__)
            if 'inaccurate' not in prob.status:
                break
            n_tries += 1

        # print(prob.status)
        if 'optimal' in prob.status:
            if 'inaccurate' in prob.status:
                logging.info(f'{i}--{k}-->{j} gave {prob.status}')
            return True
        elif 'infeasible' in prob.status:
            return False
        elif 'unbounded' in prob.status:
            warnings.warn(f'{i}--{k}-->{j} gave {prob.status}')
            return True
        raise ETCAbstractionError(
                'Some unknown exception happened! The Semi-Definite Relaxation'
                ' problem should be either feasible or infeasible.  If you are'
                ' here that means it was neither.  Possibily, the problem is'
                ' numerically ill conditioned and failed')

    def add_level_sets(self, minV, maxV, nV):
        '''
        Performs additional partitioning on the state space, on top of the
        original cones.  These partitions are of the type
                        {x: V_i <= x'Px <= V_{i+1}},
        where
            V_1 = minV: this can be regarded as the terminal value;
            V_{nV} = maxV: this can be regarded as a safety value; and
            V_i = alpha*V_{i+1}, where alpha is computed from the other
                                 given parameters.
        Note: the terminal set is {x: x'Px <= V_1} and the last set, which is
        unsafe, is {x: x'Px >= V_nV}.
        '''

        self.has_level_sets = True

        alpha = (maxV/minV)**(1./(nV-2))
        V_list = [minV*(alpha**z) for z in range(0, nV-1)]
        V_list[-1] = maxV

        # Storage
        self.alpha = alpha
        self.minV = minV
        self.maxV = maxV
        self.V_list = V_list

        ''' Redo reachability '''
        # Step 1: for each i, j, we know the minimum and maximum decay of the
        # Lyapunov function.  This can translate to how many discrete level set
        # jumps can be attained.

        transition_levels = {}
        log_alpha = np.log(alpha)

        for (((i, k), j), (cost_low, cost_high)) in self.complete_cost.items():
            steps_down = int(np.floor(np.log(1+cost_low)/log_alpha))
            # steps_up = int(np.ceil(np.log(alpha+cost_high)/log_alpha)) - 1
            # It looks like the method above is unnecessarily conservative
            steps_up = int(np.ceil(np.log(1+cost_high)/log_alpha))
            transition_levels[((i, k), j)] = (steps_down, steps_up)

        # Step 2: build the complete reachability map
        complete_transition = {}
        partitions = {}  # For reference, build a list of partitions
        for ((i, k), cone_list) in self.transition.items():
            sampling_time = i  # Using the actual discrete sampling time
            for v in range(0, nV):
                # Set the partition (index 1: discrete time,
                #                    index 2: Lyapunov level interval)
                if v == 0:
                    level_interval = (0, V_list[0])
                elif v == nV-1:
                    level_interval = (V_list[-1], np.inf)
                else:
                    level_interval = (V_list[v-1], V_list[v])

                partitions[(i, v)] = Partition(sampling_time, level_interval)

                # Fill in transition dictionary
                complete_transition[((i, v), k)] = []
                for j in cone_list:
                    steps_down, steps_up = transition_levels[((i, k), j)]
                    low_v = max(0, v + steps_down)
                    high_v = min(nV - 1, v + steps_up)
                    complete_transition[((i, v), k)] \
                        += [(j, n) for n in range(low_v, high_v + 1)]
                # Delete empty transitions - these are unsafe edges
                if not complete_transition[((i, v), k)]:
                    del complete_transition[((i, v), k)]

        self.transition_levels = transition_levels
        self.partitions = partitions
        self.complete_transition = complete_transition

    def _transition_cost(self, i, k, dPk, j=None):
        """
        Cost of transition from i after k sampling instants if set j is reached
        Problem: min/max (cost(x(k)) - cost(x(0)))
                 s.t. x'Px = 1 (normalized cost)
                      x(0) in R(i) where R is the region where x(0) belongs
                                  when it is supposed to trigger at instant i.
                      x(k) = M(k)x(0) in R(j),  (omitted if j=None)
        where cost(x) := x'Px

        Parameters
        ----------
        i : int, in self.Q.keys()
            index of the origin region
        k : int, in self.M.keys()
            sampling instant
        dPk : np.array, square matrix
            difference of Lyapunov matrices P(k) - P(0)
        j : int, in self.Q.keys(), optional
            index of the target region

        Returns
        -------
        The interval (mincost, maxcost)

        Raises
        ------
        ETCAbstractionError
            If either the problem is infeasible, indicating Reachability is
            not satisfied, or if an unexpected error occured.
        """

        n = dPk.shape[0]
        # Mk = self.M[k]
        P = self.trigger.P

        # Start building QCQP
        # dPk_norm = la.norm(dPk)
        dPk_norm = 1.  # Override normalization (to see if it improves)
        # (apparently it changes nothing, since the solver already normalizes
        # things...)
        dPk = dPk/dPk_norm

        # Objective function
        obj = QuadraticForm(dPk)

        # First constraint: x.T @ P @ x == 1  (normalization)
        con = {QuadraticForm(P, np.zeros(n), -1),  # f(x) <= 1: f(x) - 1 <= 0
               QuadraticForm(-P, np.zeros(n), 1)}  # f(x) >= 1: -f(x) + 1 <= 0

        # Adding a valid cut:
        # x'*dP(k)*x = x'*P(k)*x - x*P*x = x'*P(k)*x - 1 >= -1
        # con += [x.T @ dPk @ x >= -1]  # Not effective
        con = self._add_constraints_for_region_i(i, con)
        if j is not None:
            con = self._add_constraints_for_reaching_j_after_k(j, k, con)

        # Build and solve QCQP problem (SDR)
        probMax = sdr_problem(obj, con, minimize=False)

        # Eigenvalues are global bounds
        Pinv_dPk_eigs = la.eig(la.solve(P, dPk))[0]  # Should be real anyway
        max_global_decay = max(np.real(Pinv_dPk_eigs))
        min_global_decay = min(np.real(Pinv_dPk_eigs))

        n_tries = 0
        while n_tries < _SSC_MAX_ATTEMPTS:
            probMax.solve(eps=_QCQP_TOLERANCE, max_iters=_SSC_MAX_ITERS,
                          verbose=__TEST__)
            max_value = probMax.value
            if 'inaccurate' not in probMax.status:
                break
            n_tries += 1

        if 'inaccurate' in probMax.status:
            logging.info(f'MAX {i}--{k}-->{j} is {probMax.status}')
        if 'infeasible' == probMax.status:
            raise ETCAbstractionError(f'MAX {i}--{k}-->{j} is infeasible')
        elif probMax.status in ('unbounded_inaccurate',
                                'infeasible_inaccurate'):
            logging.debug('%s. Max eig(P,dPk): %g',
                          probMax.status, max_global_decay)
            max_value = max_global_decay
        elif 'optimal' not in probMax.status:
            raise ETCAbstractionError(
                    f'MAX {i}--{k}-->{j}: ' + 
                    'Unknown error. Status of the CVX problem is %s'
                     % probMax.status)
        # x'dPk x <= probMax.value  (valid cut, not effective)
        # con.add(QuadraticForm(dPk, np.zeros(n), -probMax.value))
        probMin = sdr_problem(obj, con)

        n_tries = 0
        while n_tries < _SSC_MAX_ATTEMPTS:
            probMin.solve(eps=_QCQP_TOLERANCE, max_iters=_SSC_MAX_ITERS,
                          verbose=__TEST__)
            min_value = probMin.value
            if 'inaccurate' not in probMin.status:
                break
            n_tries += 1

        if 'inaccurate' in probMin.status:
            logging.info(f'MIN {i}--{k}-->{j} is {probMin.status}')
        if 'infeasible' == probMin.status:
            raise ETCAbstractionError(f'MIN {i}--{k}-->{j} is infeasible')
        elif probMin.status in ('unbounded_inaccurate',
                                'infeasible_inaccurate'):
            logging.debug('%s. Min eig(P,dPk): %g',
                          probMin.status, min_global_decay)
            min_value = min_global_decay
        elif 'optimal' not in probMin.status:
            raise ETCAbstractionError(
                    f'MIN {i}--{k}-->{j}: ' + 
                    'Unknown error. Status of the CVX problem is %s'
                     % probMin.status)

        logging.debug(
            'MAX %d --%d--> %d: SDR bound: %g; maximum eigenvalue: %g',
            i, k, j, probMax.value, max_global_decay)
        logging.debug(
            'MIN %d --%d--> %d: SDR bound: %g; minimum eigenvalue: %g',
            i, k, j, probMin.value,  min_global_decay)

        maxdecay = min(max_value, max_global_decay)
        mindecay = max(min_value, min_global_decay)

        return (mindecay*dPk_norm, maxdecay*dPk_norm)

    def region_of_state(self, x):
        """ Determines which region state x belongs

        Parameters
        ----------
        x: numpy.array
            Input state

        Returns
        -------
        int
            Region index (key of self.Q)

        """

        for k in sorted(self.regions):
            if all(x in Q
                   for Q in self._add_constraints_for_region_i(k, set())):
                return k
        raise ETCAbstractionError('State %s belongs to no region', str(x))

    def level_of_state(self, x: np.array):
        """ Determines the Lyapunov level where state x belongs.


        Parameters
        ----------
        x : np.array
            Input state

        Returns
        -------
        int
            The level index

        """
        Vmin = self.V_list[0]
        V = x @ (self.trigger.P @ x)
        logging.debug('V = %g', V)
        real_z = np.log(V/Vmin)/np.log(self.alpha)
        z = int(real_z) + 1

        if z >= len(self.V_list):
            # If very close to the upper edge, choose the lower level.
            if real_z % 1 <= LEVEL_SMALL_NUMBER:
                z -= 1
            else:
                raise ETCAbstractionError('State is out of bounds.'
                                          ' Level would be %d',
                                          round(z))

        return max(0, min(len(self.V_list), z))

    def reached_region_of_state(self, x, i):
        """ Determines the region state will be from x after i sampling
        instants

        Parameters
        ----------
        x: numpy.array
            Input state

        Returns
        -------
        int
            Region index (key of self.Q)

        """

        y = self.M[i] @ x
        for j in sorted(self.Q.keys()):
            if all(y in Q
                   for Q in self._add_constraints_for_region_i(j, set())):
                return j
        raise ETCAbstractionError('State %s reached no region', str(x))

    def estimate_transition(self, N):
        """ Estimate transition using gridded sampling.

        Builds self.transition_estimated, a dictionary

        Parameters
        ----------
        N: int
            number of points per dimension

        Returns
        -------
        None
        """

        # Initialize estimated transition
        transition_estimated = defaultdict(set)

        # Create uniform grid of angles
        angles = []
        for i in range(self.n-2):
            angles.append(np.arange(0, np.pi, np.pi/N))
        angles.append(np.arange(0, 2*np.pi, np.pi/N))

        # Loop for creating points on the unit ball
        # TODO: use tangents instead, for speed
        for phi in itertools.product(*angles):
            x = np.ones(self.n)
            for i in range(self.n):
                for j in range(i):
                    x[i] *= np.sin(phi[j])
                if i != self.n - 1:
                    x[i] *= np.cos(phi[i])
            # The point x is made, check the region it is in...
            i = self.region_of_state(x)
            for k in self.M:  # ... and, for each sampling instant,...
                # ... check the region it reaches
                j = self.reached_region_of_state(x, k)
                transition_estimated[(i, k)].add(j)

        self.transition_estimated = transition_estimated

    def _estimate_probabilities(self, N):
        """ Estimate and initial condition and transition probabilities.

        Generate the probability of transitions and initial condition.
        For initial condition, the estimate comes from checking the
        region of N normally i.i.d. distributed vectors. For
        transitions, out of each region and given each possible action,
        estimate the probability of reaching each of the possible
        target regions.

        This function creates self.probability_transition and
        self.probability_region

        Parameters
        ----------
        N: int
            number of points per dimension

        Returns
        -------
        None
        """
        self.probability_region = {k:0 for k in self.Q}
        # ((region, sample)): {region: prob}
        self.probability_transition \
            = {key: {j: 0 for j in v} for key, v in self.transition.items()}

        # Generate random uniformally distributed numbers
        xs = random.normal(size=(N,self.n))

        # Loop to count
        for p in range(N):
            x = xs[p, :]
            i = self.region_of_state(x)
            self.probability_region[i] += 1
            for k in self.M:
                if self.early_trigger_only:
                    if k > i:
                        continue
                j = self.reached_region_of_state(x, k)
                if j not in self.probability_transition[(i, k)]:
                    warnings.warn(
                        f'Region {j} is not expected to be reachable from '
                        f'region {i} if sample time is {k}: state is {x}.')
                    self.probability_transition[(i, k)][j] = 0
                    self.transition[(i, k)].add(j)
                    dPk = self.dP[k]
                    mincost, maxcost = self._transition_cost(i, k, dPk)
                    self.complete_cost[((i, k), j)] = (mincost, maxcost)
                self.probability_transition[(i, k)][j] += 1

        # Turn counts into probabilities
        return

        # Normalization is not really needed. But then it is not a probability,
        # but rather a "probability weight" as used in UPPAAL.
        self.probability_region = {i:c/N
                                   for i, c in self.probability_region.items()}
        for key, v in self.probability_transition.items():
            total = sum(v.values())
            self.probability_transition[key] = {j:c/total
                                                for j, c in v.items()}

    
    
    
    
    
    
    
    
    
