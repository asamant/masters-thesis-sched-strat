import shortuuid

from tga_models.ta import TGA, pyuppaal


class ControlLoop(TGA):
    """
    A Control Loop class to generate a TGA for the control loop 
    model developed for my thesis.

    Original author: P. Schalkwijk
    """

    # constants
    to_region_decl = 'to_region'
    from_region_decl = 'from_region'
    count_decl = 'count'

    def __init__(self, nta, name='ControlLoop', initial_location=None,
                 sync='up', nack='nack', ack='ack', timeout='timeout',
                 down='down', d=2):
        super().__init__()
        self.name = name

        # communicating actions
        self.sync = sync
        self.nack = nack
        self.ack = ack
        self.down = down
        self.timeout = timeout

        self.index = shortuuid.uuid()[:6]
        self.max_delay_steps = 0

        # check first if the "max delay steps" property has been added
        if hasattr(nta.abstraction, 'max_delay_steps'):
            self.max_delay_steps = nta.abstraction.max_delay_steps

        # Add locations for regions first
        self.locations.update({f'R{location}' for location in nta.locations})

        # create early trigger locations
        self.urgent = {f"Ear{loc}" for loc in nta.locations}

        self.locations.update(self.urgent)

        # Add other locations
        common_locations = {'Trans_loc', 'Clk_wait', 'Bad'}
        self.locations.update(common_locations)

        # Add Trans_loc to urgent locs
        self.urgent.update({'Trans_loc'})

        # Maybe not necessary, or for future use? (nta.actions is empty)
        self.actions_u.update(nta.actions)
        self.clocks.update(nta.clocks)

        # create all edges
        self.edges = self.create_and_update_edges(nta)

        self.invariants.update(self.create_invariants(nta.invariants))

        # create inital location

        if initial_location is None:

            self.locations.update({f'R0'})
            self.edges.update([(f'R0', str(True).lower(),
                                frozenset(
                                    {f'{self.from_region_decl} = {location}'}),
                                frozenset(), frozenset(),
                                f'R{location}') for location in nta.locations])
        else:
            self.locations.update({f'R0'})
            self.edges.update([(f'R0', str(True).lower(),
                                frozenset(
                                    {f'{self.from_region_decl} = {location}'}),
                                frozenset(), frozenset(),
                                f'R{location}') for location in initial_location])

        self.urgent.update({"R0"})
        self.l0 = 'R0'

    def create_and_update_edges(self, nta):
        """
        Create all edges required for the control loop TGA model
        :param nta:
        :return edges:
        """
        edges = set()

        # TODO: in case we wish to add multiple triggering coeffs,
        # this needs to include the index of sigma
        # edges.update([(f'R{location}',True,False,
        #           False,frozenset(),f'R{location}_s')]
        #                      for location in nta.locations)

        # first add edges for transitions to the trans_loc
        # (including early to trans_loc edges too)
        edges.update(self.create_edges_to_trans_loc(nta.edges))

        # add edges for transition to Early locations
        edges.update(self.create_early_edges(nta))

        # add nack_edges
        edges.update(self.create_nack_edges(nta.locations))

        # add ack edge
        edges.update(self.create_ack_edge())

        # add delay edges for communication network
        edges.update(self.create_comm_delay_edges(nta.locations))

        # Edge to Bad location
        bad_edge = ('Trans_loc',
                    f'{self.max_delay_steps} <= {self.count_decl}',
                    frozenset({f'{self.timeout}!'}),
                    frozenset(), frozenset(), 'Bad')

        edges.update({bad_edge})

        return edges

    def create_edges_to_trans_loc(self, nta_edges):
        """
        Convert an edge from (l,g,a,c,l') -> (l,g,a_c,a_u,c,l') 
        where the final location is the 'Trans_loc'

        :param edges:
        :return edges:
        """
        edges = set()

        # in this model we don't want to reset clocks on these edges
        reset = frozenset()

        for edge in nta_edges:

            (start, guard, assignment, clocks, end) = edge  # given edge

            # first R to trans_state
            ia = set(assignment)
            ia.update({'EarNum = 0'})

            # NOTE: the end should not have "R" before the region number
            ia.update({f"{self.to_region_decl} = {end}"})
            natural_assignment = frozenset(ia)

            # now ear_R to trans_state
            ea = set(assignment)

            # NOTE: Ideally the "end" value should come from a separate
            # reachability analysis for early edges
            ea.update({f"{self.to_region_decl} = {end}"})
            ea.update({'EarNum = EarNum + 1'})
            early_assignment = frozenset(ea)

            # add edges
            edge_nat = (f'R{start}', guard, natural_assignment,
                        frozenset({f'{self.sync}!'}), reset, 'Trans_loc')

            # Early edges should not have any guard
            edge_earl = (f'Ear{start}', str(True).lower(),
                         early_assignment, frozenset({f'{self.sync}!'}),
                         reset, 'Trans_loc')

            edges.update({edge_nat})
            edges.update({edge_earl})

        # endfor
        return edges

    def create_early_edges(self, nta, d=2):
        """
        Create edges between Ri and Eari
        :param nta:
        :param d (max early trigger steps):
        :return edge:
        """

        edges = set()
        c = next(iter(nta.clocks))  # Assuming one clock is present per CL

        # limits are not necessarily a range
        if hasattr(nta.abstraction, 'limits'):

            # trigger earlier when the clock is between tau_l - d and tau_l,
            # which is the lower bound on IET for a region
            edges.update([(f'R{location}',
                           f'{c} < {nta.abstraction.limits[location][0]} &&'
                           f'{nta.abstraction.limits[location][0]} - {d} <= {c} &&'
                           'EarNum < EarMax', False, False, frozenset(),
                           f'Ear{location}') for location in nta.locations])

        elif (nta.abstraction.is_discrete_time):
            for location in nta.locations:
                # assumption: location = number of steps
                edges.update([(f'R{location}', f'{c} < {location} &&'
                               f'{location} - {d} <= {c} && EarNum < EarMax',
                               False, False, frozenset(), f'Ear{location}')])
        else:
            pass  # TODO: case in which ETC is implemented instead of PETC through ETCTime

        return edges

    def create_nack_edges(self, nta_locations):
        """
        Create nack? edges between Trans_loc and Ri
        :param nta_locations:
        :return edges:
        """
        nack_edges = set()

        for location in nta_locations:
            loc_s = f'R{location}'

            guard = f'{self.from_region_decl} == {location} &&' \
                f'{self.count_decl} < {self.max_delay_steps}'

            # update the number of retries
            ca = set()
            ca.update({f"{self.count_decl} = {self.count_decl} + 1"})

            ca.update({f"{self.nack}?"})

            controllable_action = frozenset(ca)
            nack_edges.update({('Trans_loc', guard,
                                controllable_action, frozenset(), frozenset(), loc_s)})

        return nack_edges

    def create_ack_edge(self):
        """
        Add an ack? edge between the Trans_loc and Clk_wait loc
        :param None:
        :return edge:
        """

        action_c = set({f'{self.ack}?'})
        action_c.update({f'{self.from_region_decl} = {self.to_region_decl}'})
        action_c.update({f'{self.count_decl} = 0'})

        resets = frozenset(self.clocks)

        edge = ('Trans_loc', str(True).lower(), frozenset(action_c),
                frozenset(), resets, 'Clk_wait')

        return {edge}

    def create_comm_delay_edges(self, nta_locations, delta=1):
        """
        To model a control loop occupying a network channel for delta time steps
        :param nta_locations:
        :param delta:
        :return edges:
        """

        edges = set()

        # Assuming a single clock is present per CL
        c = next(iter(self.clocks))

        for location in nta_locations:

            guard = f'{self.to_region_decl} == {location} && {delta} <= {c}'

            action_c = frozenset({f'{self.down}!'})
            resets = frozenset(self.clocks)
            edge = ('Clk_wait', guard, action_c,
                    frozenset(), resets, f'R{location}')
            edges.update({edge})

        return edges

    def create_invariants(self, nta_invariants, delta=1):

        # Assuming a single clock is present per CL
        c = next(iter(self.clocks))

        invariants = dict()
        invariants.update({f'R{location}': inv
                           for location, inv
                           in nta_invariants.items()})

        invariants.update(
            {'Trans_loc': f'{self.count_decl} <= {self.max_delay_steps}'})
        invariants.update({'Clk_wait': f'{c} <= {delta}'})

        return invariants

    def generate_transitions(self):
        """ 
        convert edges to transitions 
        """

        transitions = []
        for (source, guard, actions_c, actions_u, resets, target) in self.edges:
            props = {}
            if guard:
                props.update({'guard': str(guard).lower()
                              if type(guard) is bool else guard})
            if actions_u:
                props.update({'synchronisation': '\n'.join(actions_u)})
                props.update({'controllable': False})
            if resets:
                props.update({'assignment': ', '.join(
                    [f'{clock}=0' for clock in resets])})
            if actions_c:
                clock_assignments = props.get('assignment', False)
                action_assignments = []
                for action in actions_c:
                    # sync actions
                    if (("?" in action) or ("!" in action)):

                        # expected to have only one sync action per edge
                        props.update({'synchronisation': action})
                        props.update({'controllable': True})
                        continue
                    # end
                    action_assignments.append(action)

                # endfor

                action_assignments = ', '.join(action_assignments)

                if clock_assignments:

                    action_assignments = ', '.join(filter(None,
                                                          [clock_assignments, action_assignments]))

                props.update({'assignment': action_assignments})

            if target not in self.locations:
                print(f'{target} is not found in set of locations')
            else:
                transitions.append(
                    pyuppaal.Transition(source, target, **props))

        return transitions

    def generate_clocks(self):
        return f"clock {', '.join(self.clocks)};"

    def generate_declarations(self):

        clock_decl = f'{self.generate_clocks()}'

        # variables needed for the new automata model
        ints_decl = f"int {', '.join([self.to_region_decl, self.from_region_decl, self.count_decl])};"

        # include the max num of retries
        n_max = self.max_delay_steps

        return clock_decl + '\n' + ints_decl + '\n' + f"const int n_max = {n_max};" + '\n'

    def to_xml(self, layout=False):
        template = self.template
        if layout:
            template.layout(auto_nails=True)
        return template.to_xml()

    ############################### Original code for methods (P. Schalkwijk) #########################

    def early(self, edge):
        """
        Convert an edge from (l,g,a,c,l') -> (Ear(l),g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, assignment, clocks, end) = edge
        ia = set(assignment)
        ia.update({'EarNum = EarNum + 1'})
        assignment = frozenset(ia)
        return f'Ear{start}', guard, assignment, frozenset({f'{self.sync}!'}), clocks, f'R{end}'

    def uncontrollable(self, edge):
        """
        Convert an edge from (l,g,a,c,l') -> (l,g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, internal_action, clocks, end) = edge
        ia = set(internal_action)
        ia.update({'EarNum = 0'})
        internal_action = frozenset(ia)
        return f'R{start}', guard, internal_action, frozenset({f'{self.sync}!'}), clocks, f'R{end}'

    def controllable(self, edge):
        """
        Convert an edge from (l,g,a,c,l') -> (l,g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, internal_action, clocks, end) = edge
        return f'R{start}', guard, internal_action, False, clocks, f'R{end}'

    def generate_locations(self):
        locations = [pyuppaal.Location(invariant=self.invariants.get(loc), name=loc, urgent=(loc in self.urgent))
                     for loc in self.locations]
        return locations


# TODO: Add concept of changing trig. coeffcients to the new control loop TGA model
class sigmaControlLoop(TGA):
    def __init__(self, *ntas, name='ControlLoop', initial_location=None, sync='up', d=5):
        super().__init__()
        self.name = name
        self.sync = sync
        self.index = shortuuid.uuid()[:6]
        self.ntas = ntas
        for index, nta in enumerate(ntas):
            self.locations.update(
                {f'R{location}_s{index}' for location in nta.locations})
            self.actions_u.update(nta.actions)
            self.clocks.update(nta.clocks)
            self.edges = {self.uncontrollable(
                edge, index) for edge in nta.edges}
            self.edges.update({self.early(edge, index) for edge in nta.edges})
            self.invariants.update({f'{key}_s{index}': value}
                                   for (key, value) in nta.invariants.items())

            # create early trigger locations and make choosing sigma urgent
            self.urgent = {f"Ear{loc}_s{index}" for loc in nta.locations}
            self.urgent.update({f'R{location}' for location in nta.locations})
            self.locations.update(self.urgent)

            # Add edge from Ri to Eari
            self.edges.update([(f'R{location}_s{index}', f'c < {nta.abstraction.limits[location][0]} &&'
                                f'{nta.abstraction.limits[location][0]} - {d} <= c && EarNum < EarMax', False, False, frozenset(
                                ),
                                f'Ear{location}_s{index}')
                               for location in nta.locations])
            self.edges.update([(f'R{location}', True, False, False, frozenset(), f'R{location}_s{index}')]
                              for location in nta.locations)
            if initial_location is None:
                self.locations.update({f'R0'})
                self.edges.update([(f'R0', False, False, False, frozenset(), f'R{location}')
                                   for location in nta.locations])
            else:
                self.locations.update({f'R0'})
                self.edges.update([(f'R0', False, False, False, frozenset(), f'R{location}')
                                   for location in initial_location])
        self.urgent.update({"R0"})
        self.l0 = 'R0'

    def early(self, edge, index):
        """
        Convert an edge from (l,g,a,c,l') -> (Ear(l),g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, assignment, clocks, end) = edge
        ia = set(assignment)
        ia.update({'EarNum = EarNum + 1'})
        assignment = frozenset(ia)
        return f'Ear{start}_s{index}', guard, assignment, frozenset({f'{self.sync}!'}), clocks, f'R{end}'

    def uncontrollable(self, edge, index):
        """
        Convert an edge from (l,g,a,c,l') -> (l,g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, internal_action, clocks, end) = edge
        ia = set(internal_action)
        ia.update({'EarNum = 0'})
        internal_action = frozenset(ia)
        return f'R{start}', guard, internal_action, frozenset({f'{self.sync}!'}), clocks, f'R{end}'

    def controllable(self, edge):
        """
        Convert an edge from (l,g,a,c,l') -> (l,g,a_c,a_u,c,l')
        :param edge:
        :return:
        """
        (start, guard, internal_action, clocks, end) = edge
        return f'R{start}', guard, internal_action, False, clocks, f'R{end}'

    def generate_locations(self):
        locations = [pyuppaal.Location(invariant=self.invariants.get(loc), name=loc, urgent=(loc in self.urgent))
                     for loc in self.locations]
        return locations

    def generate_transitions(self):
        """ convert edges to transitions """
        transitions = []
        for (source, guard, actions_c, actions_u, resets, target) in self.edges:
            props = {}
            if guard:
                props.update({'guard': str(guard).lower()
                              if type(guard) is bool else guard})
            if actions_u:
                props.update({'synchronisation': '\n'.join(actions_u)})
                props.update({'controllable': False})
            if resets:
                props.update({'assignment': ', '.join(
                    [f'{clock}=0' for clock in resets])})
            if actions_c:
                clock_assignments = props.get('assignment', False)
                action_assignments = ''
                for action in actions_c:
                    # sync actions
                    if (("?" in action) or ("!" in action)):

                        # expected to have only one sync action per edge
                        props.update({'synchronisation': action})
                        continue
                    # end
                    action_assignments = ', '.join(
                        [action_assignments, action])

                # endfor

                if clock_assignments:
                    action_assignments = ', '.join(
                        [clock_assignments, action_assignments])
                props.update({'assignment': action_assignments})
            if target not in self.locations:
                print(f'{target} is not found in set of locations')
            else:
                transitions.append(
                    pyuppaal.Transition(source, target, **props))
        return transitions

    def generate_clocks(self):
        return f"clock {', '.join(self.clocks)};"

    def generate_declarations(self):
        return f'{self.generate_clocks()}\n'

    def to_xml(self, layout=False):
        template = self.template
        if layout:
            template.layout(auto_nails=True)
        return template.to_xml()
