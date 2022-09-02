import operator
import sys

import numpy as np
from matplotlib import pyplot as plt
import math
from grispy import GriSPy
from skspatial import objects as sk_objects
import random


class WLC:
    """
    A class used to build DNA using the WLC model

    ...

    Attributes
    ----------
    n : int
        the number of DNA segments to build
    s_link : float
        the length of each segment in nm
    l_p : float
        DNA persistence length in nm
    alpha : float
        angle from reference vector (usually [0,0,1]) to segment in radians
    alpha_lim : flaot
        when arccos is applied this float will give the largest alpha angle allowed


    Methods
    -------
    generate_boundaries()
        generates boundaries for DNA
    rotation_matrix_from_vectors(vec1, vec2)
        returns the rotation matrix to get from vec1 to vec2
    phi()
        returns phi angle
            requires alpha_lim
    theta()
        returns theta angle
    wlc_walk()
        produces self.cum_sum
    theo_r_squared_wlc()
    real_r_squared()
    angle_test()
        returns angle_between_vecs list
    visualizer()
    """
    bp_len = 0.34  # base pair length in nm

    e_coli_vol_l = 6.7 * (10 ** (-16))

    def __init__(self, bp, boundaries=None, conc=None, targets=None, s_link=0.2, l_p=50, boundary_mode='target_concentration', generated_vecs=None):
        """
        :param bp: target bp amount
        :param boundaries: boundaries in nm
        :param conc: concentration in M (bp or dna general?)
        :param s_link: size of each segment
        :param l_p: persistence lenght nm
        n: number of segments to produce, given bp
        """

        self.bp = bp
        self.n = round((self.bp * WLC.bp_len) / s_link)
        self.s_link = s_link
        self.l_p = l_p
        self.targets = targets
        self.generated_vecs = generated_vecs

        self.boundaries = boundaries
        self.conc = conc

        self.alpha = np.arccos(np.exp(-(
                    s_link / l_p)))  # alpha = bond angle (eq from http://www.rsc.org.ez.sun.ac.za/suppdata/nr/c4/c4nr07153k/c4nr07153k1.pdf) also (http://www.rsc.org/suppdata/c6/py/c6py01656a/c6py01656a1.pdf eq s2) and originally from (https://arxiv.org/pdf/1404.4332.pdf eq 6)
        self.alpha_lim = np.cos(self.alpha)

        self.wide_phi_lim = np.cos(np.radians(random.randint(0, 360))) # 135
        self.wide_phi = np.random.uniform(self.wide_phi_lim, 1)

        # walk vars
        self.angle_between_vecs = []
        self.cum_sum = None
        self.bp_dict = None

        if boundary_mode is not None:
            self.generate_boundaries(boundary_mode)
        # starting point
        if self.generated_vecs is None:
            self.generated_vecs = [np.array([0, 0, 0])]
        elif self.generated_vecs == 'random':
            self.generated_vecs = [np.array([np.random.uniform(self.boundaries[0][0], self.boundaries[0][1]) for _ in range(3)])]


    def generate_boundaries(self, mode):
        """
        :param mode: if 'target concentration' run method before wlc is built
                     if 'genome_first' run after genome is built
        :return: Set boundaries if needed, else no boundaries
        """

        if mode == 'target_concentration':
            self.bound_target_conc(self.conc)
        elif mode == 'genome_first':
            self.bound_build_genome_first()
        elif mode is None:
            print('No boundaries specified')

    # setter for boundaries, if concentration is first
    def bound_target_conc(self, conc):
        moles = self.bp / (6 * (10 ** (23)))
        volume = (moles / conc)  # L

        cubic_nm = volume * (10 ** 24)
        self.edge_len = cubic_nm ** (1 / 3)

        cubic_m = (self.edge_len * (10 ** (-9))) ** 3
        self.box_vol_l = cubic_m * (10 ** 3)

        half_edge_len = self.edge_len / 2
        boundaries = [0 - half_edge_len, 0 + half_edge_len]

        self.boundaries = [boundaries, boundaries, boundaries]

    # setter for boundaries, if genome is built first
    def bound_build_genome_first(self):
        # get min, max of x, y, z; box_dim is bp_len away from these max values
        x_bound = [min(self.cum_sum[:, 0]) - WLC.bp_len, max(self.cum_sum[:, 0]) + WLC.bp_len]
        y_bound = [min(self.cum_sum[:, 1]) - WLC.bp_len, max(self.cum_sum[:, 1]) + WLC.bp_len]
        z_bound = [min(self.cum_sum[:, 2]) - WLC.bp_len, max(self.cum_sum[:, 2]) + WLC.bp_len]

        self.boundaries = [x_bound, y_bound, z_bound]

    def set_equal_bounds(self, value):
        bound = [-value, value]
        self.boundaries = [bound, bound, bound]

    def rotation_matrix_from_vectors(self, vec1,
                                     vec2):  # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transformation matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Enable numpy error raising
        np.seterr(divide='raise')

        try:
            1 / (s ** 2)
        except FloatingPointError:
            print('Division by zero. Likely no rotation.')
            return None
        else:
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    def phi(self, angle):
        return np.arccos(angle)

    @staticmethod
    def theta():
        return np.random.uniform(-np.pi, np.pi)

    def generate_new_vec(self, i, theta_list, phi_list, rotation_matrix):
        spherical_coord = np.array(
            [(self.s_link * math.cos(theta_list[i]) * math.sin(phi_list[i])),
             (self.s_link * math.sin(theta_list[i]) * math.sin(phi_list[i])),
             (self.s_link * math.cos(phi_list[i]))])  # generate new vector
        new_vec = np.dot(rotation_matrix, spherical_coord)  # rotate new spherical_coord by rotation_matrix
        return self.s_link * (new_vec / np.linalg.norm(new_vec))  # normalize new_vec

    def generate_new_vec_wide_phi(self, rotation_matrix, angle):
        spherical_coord = np.array(
            [(self.s_link * math.cos(WLC.theta()) * math.sin(self.phi(angle))),
             (self.s_link * math.sin(WLC.theta()) * math.sin(self.phi(angle))),
             (self.s_link * math.cos(self.phi(angle)))])  # generate new vector
        new_vec = np.dot(rotation_matrix, spherical_coord)  # rotate new spherical_coord by rotation_matrix
        return self.s_link * (new_vec / np.linalg.norm(new_vec))  # normalize new_vec

    def generate_cone(self, point_count, rotation_matrix, phi):
        curent_angle = -np.pi
        theta_list = [curent_angle]

        angle_increment = (2 * np.pi) / point_count

        for _ in range(point_count):
            curent_angle += angle_increment
            theta_list.append(curent_angle)

        points_list = []

        for i in range(point_count):
            theta = theta_list[i]

            spherical_coord = np.array(
                [(self.s_link * math.cos(theta) * math.sin(phi)),
                 (self.s_link * math.sin(theta) * math.sin(phi)),
                 (self.s_link * math.cos(phi))])  # generate new vector

            generated_vec = self.s_link * (spherical_coord / np.linalg.norm(spherical_coord))

            rotated_vec = np.dot(rotation_matrix, generated_vec)
            points_list.append(rotated_vec)

        # all cone points from origin points
        points_list = np.array(points_list)

        # filter points according to boundary
        points_list = list(points_list)

        # filter base on bounds
        count = 0
        while True:
            if not (self.boundaries[0][0] < points_list[count][0] < self.boundaries[0][1]) and (
                    self.boundaries[1][0] < points_list[count][1] < self.boundaries[1][1]) and (
                    self.boundaries[2][0] < points_list[count][2] < self.boundaries[2][1]):
                points_list.pop(count)
            else:
                count += 1

            if count == len(points_list):
                break

        points_list = np.array(points_list)

        if points_list.size == 0:
            return None
        else:
            # select one
            random_point = random.choice(points_list)

            # return one point
            return random_point

    def wlc_walk(self, start_angle='random'):
        """
        :param start_angle: random or (phi,theta)
        :return:
        """

        phi = []  # vertical spherical coordinate angle
        theta = []  # polar angle
        if start_angle != 'random':
            phi_first = np.radians(start_angle[0])
            theta_first = np.radians(start_angle[1])
        else:
            phi_first = np.random.uniform(np.radians(0), np.radians(360))
            theta_first = np.random.uniform(np.radians(0), np.radians(360))
        # Generate 1st vector to avoid division by 0
        new_vec = np.array(
            [(self.s_link * math.cos(theta_first) * math.sin(phi_first)),
             (self.s_link * math.sin(theta_first) * math.sin(phi_first)),
             (self.s_link * math.cos(phi_first))])

        self.generated_vecs.append(new_vec)

        for i in range(self.n - 1):
            phi.append(self.phi(self.alpha_lim))  # remember to remove theta if not valid
            theta.append(WLC.theta())  # remember to remove theta if not valid

            # Previous vec
            reference_vec = np.array(
                [0, 0, self.s_link])  # eg.[0,0,1], should be equal to spherical coord reference vector
            norm_prev_vec = self.generated_vecs[-1] / np.linalg.norm(self.generated_vecs[-1])  # normalize last vector
            rotation_matrix = self.rotation_matrix_from_vectors(reference_vec,
                                                                norm_prev_vec)  # find matrix describing the rotation from reference_vec to norm_prev_vec

            current_position = sum(self.generated_vecs)

            # New vec
            norm_new_vec = self.generate_new_vec(i, theta, phi, rotation_matrix)  # self references instance

            pot_current_position = current_position + norm_new_vec
            if self.boundaries is not None:
                # if inside boundaries
                if (self.boundaries[0][0] < pot_current_position[0] < self.boundaries[0][1]) and (
                        self.boundaries[1][0] < pot_current_position[1] < self.boundaries[1][1]) and (
                        self.boundaries[2][0] < pot_current_position[2] < self.boundaries[2][1]):

                    current_position = pot_current_position
                    self.generated_vecs.append(norm_new_vec)
                else:

                    curr_phi = self.phi(self.alpha_lim)
                    phi_incr = self.phi(self.alpha_lim)
                    while not ((self.boundaries[0][0] < pot_current_position[0] < self.boundaries[0][1]) and (
                            self.boundaries[1][0] < pot_current_position[1] < self.boundaries[1][1]) and (
                                       self.boundaries[2][0] < pot_current_position[2] < self.boundaries[2][1])):
                        curr_phi += phi_incr
                        # add better vector handling
                        norm_new_vec = self.generate_cone(100, rotation_matrix, curr_phi)

                        if norm_new_vec is not None:
                            pot_current_position = current_position + norm_new_vec
                            if (self.boundaries[0][0] < pot_current_position[0] < self.boundaries[0][1]) and (
                                    self.boundaries[1][0] < pot_current_position[1] < self.boundaries[1][1]) and (
                                    self.boundaries[2][0] < pot_current_position[2] < self.boundaries[2][1]):
                                self.generated_vecs.append(norm_new_vec)
                                break

            else:
                self.generated_vecs.append(norm_new_vec)

            # for angle_test()

            self.angle_between_vecs.append(np.arccos(np.dot(self.generated_vecs[-1], self.generated_vecs[-2]) / (
                    np.linalg.norm(self.generated_vecs[-1]) * np.linalg.norm(self.generated_vecs[-2]))))

        norm_new = np.reshape(self.generated_vecs, (-1, 3))

        self.cum_sum = np.cumsum(norm_new, axis=0)

        return self.cum_sum

    def theo_r_squared_wlc(self):  # (eq from http://polymerdatabase.com/polymer%20physics/Wormlike%20Chain.html)
        l = self.n * self.s_link

        return (2 * l * self.l_p) - (2 * (self.l_p ** 2) * (1 - np.exp(-(l / self.l_p))))

    def real_r_squared(self):

        return np.linalg.norm(self.cum_sum[0]-self.cum_sum[-1]) ** 2

    def r_g(self):
        """
        :return: radius of gyration (nm)
        """
        # PBC p.321

        # centre of mass
        r_cm = np.array(np.sum(self.cum_sum)) / len(self.cum_sum)

        # r_g^2
        r_g_sq_ave = np.sum([(i - r_cm) ** 2 for i in self.cum_sum]) / len(self.cum_sum)

        # r_g (nm)
        r_g = np.sqrt(r_g_sq_ave)

        return r_g

    def angle_test(self):
        if not self.angle_between_vecs:  # is angle_between_vecs empty
            print('No angles in angle_between_vecs')
        else:
            return self.angle_between_vecs

    def generate_bp_positions(self):
        # dict of {segment number:([point1],[point2])}
        self.bp_dict = {i: ((self.cum_sum[i], self.cum_sum[i + 1]), (i + 1) * self.s_link) for i in
                        range(len(self.cum_sum) - 1)}

    def create_targets(self, targ_type, amt=1, size_of_target=10, prop_len=None,
                       spec_bp=None):  ### still need to add checks for multiple targets. especially for random

        # create targets equal to amt
        for _ in range(amt):
            Target(self, targ_type=targ_type, size_of_target=size_of_target, prop_len=prop_len, spec_bp=spec_bp)

        self.targets = [i for i in Target.get_target_list()]

    def visualizer(self):
        ax = plt.axes(projection='3d')
        self.adj_cum_sum = self.cum_sum[:, 0], self.cum_sum[:, 1], self.cum_sum[:, 2]  # xs, ys, zs
        x, y, z = self.adj_cum_sum
        ax.plot3D(x, y, z, 'gray')
        ax.set_title('3D random walk')
        if self.boundaries is not None:
            ax.set_xlim3d(self.boundaries[0][0], self.boundaries[0][1])
            ax.set_ylim3d(self.boundaries[1][0], self.boundaries[1][1])
            ax.set_zlim3d(self.boundaries[2][0], self.boundaries[2][1])
        return plt.show()


class Target(WLC):
    target_list = []

    def __init__(self, chain, targ_type=None, size_of_target=10, prop_len=None, spec_bp=None):
        """
        :param chain:
        :param targ_type: 'random' or 'prop_len' or 'spec_bp'
        :param size_of_target:
        :param prop_len: only fill if type is 'prop_len'
        :param spec_bp: only fill if type is 'spec_bp'
        """
        """
        different ways to generate target:
            'random'
                choose from (0 + (size_of_target/2)) to (max(self.bp_dict) * self.s_link - (size_of_target/2))
                if more than one target make sure they dont overlap
            'prop_len'
                proportion of total length (eg. 0.1 or 0.3)
                if specification is on edge or overlapping, shift
            'spec_bp'
                specific bp positions (give bp positions as input and feed to model in relevant nm positions)
                specific bp positions

        output: (1st bp, 2nd bp), (1st nm, 2nd nm)
        """
        ### num_targ=None should be in DNA_class

        self.size_of_target = size_of_target
        self.final_bp_position = chain.bp
        self.min_targ_left_edge = 0
        self.max_targ_left_edge = self.final_bp_position - self.size_of_target

        self.left_edge = None
        self.right_edge = None
        self.targ_tuple = None

        if targ_type == 'random':
            self.targ_tuple = self.gen_random()
        elif targ_type == 'prop_len':
            self.targ_tuple = self.gen_prop_len(prop_len)
        elif targ_type == 'spec_bp':
            self.targ_tuple = self.gen_spec_bp(spec_bp)

        # check overlapping targets
        left_edge_list = [[i[0], i[1]] for i in Target.target_list]

        for i in left_edge_list:
            if i[0][0] < self.left_edge < i[0][1]:
                self.shift_right(i)

            elif i[0][0] < self.left_edge < i[0][1]:
                self.shift_left(i)

        print('targ_tuple', self.targ_tuple)

        Target.target_list.append(self.targ_tuple)

    def shift_right(self, i):
        self.left_edge = i[1]
        self.right_edge = self.left_edge + self.size_of_target

        nm_left_edge = self.left_edge * WLC.bp_len
        nm_right_edge = self.right_edge * WLC.bp_len

        left_edge_list = [[j[0], j[1]] for j in Target.target_list]

        for j in left_edge_list:

            if j[0] < self.left_edge < j[1] or j[0] < self.right_edge < j[1] or self.right_edge > self.final_bp_position:
                raise ValueError('No space to shift right')

        return (self.left_edge, self.right_edge), (nm_left_edge, nm_right_edge)

    def shift_left(self, i):
        self.right_edge = i[0]
        self.left_edge = self.right_edge - self.size_of_target

        nm_left_edge = self.left_edge * WLC.bp_len
        nm_right_edge = self.right_edge * WLC.bp_len

        left_edge_list = [[j[0], j[1]] for j in Target.target_list]

        for j in left_edge_list:

            if j[0] < self.left_edge < j[1] or j[0] < self.right_edge < j[1] or self.left_edge < 0:
                raise ValueError('No space to shift left')

        return (self.left_edge, self.right_edge), (nm_left_edge, nm_right_edge)

    def gen_random(self):
        """choose from (0 + (size_of_target/2)) to (max(self.bp_dict) * self.s_link - (size_of_target/2))
                if more than one target make sure they dont overlap"""
        self.left_edge = np.random.uniform(self.min_targ_left_edge, self.max_targ_left_edge)
        self.right_edge = self.left_edge + self.size_of_target

        nm_left_edge = self.left_edge * WLC.bp_len
        nm_right_edge = self.right_edge * WLC.bp_len
        # returns single target (bp positions), (nm chain positions)
        return (self.left_edge, self.right_edge), (nm_left_edge, nm_right_edge)

    def gen_prop_len(self, prop_len):
        """proportion of total length (eg. 0.1 or 0.3)
                if specification is on edge or overlapping, shift"""

        pot_bp = (self.final_bp_position * prop_len) - self.size_of_target / 2

        # adjust for edges
        if pot_bp < self.min_targ_left_edge:
            pot_bp = self.min_targ_left_edge
        elif pot_bp > self.max_targ_left_edge:
            pot_bp = self.max_targ_left_edge

        self.left_edge = pot_bp
        self.right_edge = pot_bp + self.size_of_target

        nm_left_edge = self.left_edge * WLC.bp_len
        nm_right_edge = self.right_edge * WLC.bp_len
        # returns single target (bp positions), (nm chain positions)
        return (self.left_edge, self.right_edge), (nm_left_edge, nm_right_edge)

    def gen_spec_bp(self, spec_bp):

        assert 0 < spec_bp < self.final_bp_position
        pot_bp = spec_bp - (self.size_of_target / 2)

        # adjust for edges
        if pot_bp < self.min_targ_left_edge:
            pot_bp = self.min_targ_left_edge
        elif pot_bp > self.max_targ_left_edge:
            pot_bp = self.max_targ_left_edge

        self.left_edge = pot_bp
        self.right_edge = pot_bp + self.size_of_target

        nm_left_edge = self.left_edge * WLC.bp_len
        nm_right_edge = self.right_edge * WLC.bp_len
        # returns single target (bp positions), (nm chain positions)
        return (self.left_edge, self.right_edge), (nm_left_edge, nm_right_edge)

    @classmethod
    def get_target_list(cls):
        return cls.target_list