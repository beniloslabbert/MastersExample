from Dna import WLC
from Diffusers import TF

import operator
import sys

import numpy as np
from matplotlib import pyplot as plt
import math
from grispy import GriSPy
from skspatial import objects as sk_objects
import random
import os.path
import pickle
import tempfile
import warnings


class System:

    property_names_list = ['k_off', 'k_on', 'k_a', 'k_d', 'f_dna', 'mean_unbind_bind', 'mean_sliding', 'total_searched_fraction', 'mean_unbind_bind_displacement', 'simple_redundant_bp_fraction', 'total_time']

    def __init__(self, dna_object, tf_objects=None, theoretical_time=None):
        self.tf_objects = tf_objects
        self.theoretical_time = theoretical_time

        self.sys_df = None

        # tf averages lists
        self.sys_k_off = []
        self.sys_k_on = []
        self.sys_k_a = []
        self.sys_k_d = []
        self.sys_total_time = []
        self.sys_f_dna = []
        self.sys_mean_unbind_bind = []
        self.sys_mean_sliding = []
        self.sys_total_searched_fraction = []
        self.sys_mean_unbind_bind_displacement = []
        self.sys_simple_redundant_bp_fraction = []

        self.sys_property_lists = []

    def fill_sys_property_list(self):
        self.sys_property_lists = [self.sys_k_off, self.sys_k_on, self.sys_k_a, self.sys_k_d, self.sys_f_dna, self.sys_mean_unbind_bind, self.sys_mean_sliding, self.sys_total_searched_fraction, self.sys_mean_unbind_bind_displacement, self.sys_simple_redundant_bp_fraction, self.sys_total_time]

    def theoretical_total_time(self, dna_object):
        """
        To check expected seconds adjusted for volume. This is allowed because the volume/time relationship is linear, as described by Gomez 2016 (p. 11186). T = V/k.
        I could test this myself by changing volume and seeing whether time scales monotonously with it.
        :param dna_object: dna chain
        :return: seconds for search expected, adjusted for decreased simulation volume.
        """
        return self.theoretical_time * (dna_object.box_vol_l / WLC.e_coli_vol_l)

    def system_metrics(self):

        # add all TF metrics to system metric list


        # produce TF data
        # pass below
        for i in TF.active_tfs:
            for j in range(len(System.property_names_list)):
                i.tfdata.append_metrics()
                self.sys_property_lists[j].append(i.tfdata.tf_df[System.property_names_list[j]])
                #self.sys_property_lists[j].append(i.tfdata.append_metrics().tf_df[System.property_names_list[j]])

        self.sys_df = {('sys_' + System.property_names_list[i]): self.sys_property_lists[i] for i in range(len(self.sys_property_lists))}


    def vizualize_system(self, dna_object, tf_objects):
        # ax to plot values
        ax = plt.axes(projection='3d')

        # plot
        self.tfs_cum_sum = np.reshape(tf_objects.positions, (-1, 3))
        self.adj_tfs_cum_sum = self.tfs_cum_sum[:, 0], self.tfs_cum_sum[:, 1], self.tfs_cum_sum[:, 2]  # xs, ys, zs
        tf_x, tf_y, tf_z = self.adj_tfs_cum_sum
        # plot dna
        ax.plot3D(tf_x, tf_y, tf_z, 'blue')

        self.adj_dna_cum_sum = dna_object.cum_sum[:, 0], dna_object.cum_sum[:, 1], dna_object.cum_sum[:,
                                                                                   2]  # xs, ys, zs
        x, y, z = self.adj_dna_cum_sum
        # plot dna
        ax.plot3D(x, y, z, 'gray', alpha=0.5)

        ax.scatter3D(tf_x[0], tf_y[0], tf_z[0], 'green')
        ax.scatter3D(tf_x[-1], tf_y[-1], tf_z[-1], 'red')

        ax.set_title('3D random walk')
        ax.set_xlim3d(dna_object.boundaries[0][0], dna_object.boundaries[0][1])
        ax.set_ylim3d(dna_object.boundaries[1][0], dna_object.boundaries[1][1])
        ax.set_zlim3d(dna_object.boundaries[2][0], dna_object.boundaries[2][1])
        return plt.show()
