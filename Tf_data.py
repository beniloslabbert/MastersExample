from Diffusers import TF
from System import System
from Dna import WLC

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


class TfSimProp:

    def __init__(self, tf_instance):
        """
        :param tf_instance: single TF
        """
        self.tf_instance = tf_instance

        self.bind_times = [[0, None]]  # [[unbound t, bound t], ...]
        self.unbind_bind_bp = {'bind': [], 'unbind': []}  # when calculating

        self.t_three_d = None
        self.t_one_d = tf_instance.targ_time_1d * (10 ** -6)

        self.k_on = None
        self.k_d = None
        self.k_a = None
        self.total_time = None

        # fraction of total time spent on DNA
        self.f_dna = None

        # var for bp distance between binding and unbinding
        self.mean_unbind_bind = None

        # radius of gyration for tf
        self.r_g = None

        # convert microsec to s
        self.k_off = 1 / (tf_instance.targ_time_1d / (10 ** (6)))

        # bp sliding distance
        self.sliding_list = []
        self.sliding_min_max = [None, None]
        self.mean_sliding = None
        self.sliding_min_max_list = []
        self.overlap_list = []
        self.redundant_bp_list = []
        self.redundancy_bp_count = {}
        self.non_redundant_slide = []
        self.total_searched_fraction = None
        self.simple_redundant_bp_fraction = None

        self.bind_pos = []
        self.unbind_pos = []
        self.unbind_bind_displacement_list = []
        self.mean_unbind_bind_displacement = None

        self.p_bind_list = None
        self.bind_times_ave = None

        #tf distributions
        self.diff_bind_times = []  # time between unbinding and subsequent rebinding
        self.bp_diff = []  # bp change between unbinding and subsequent rebinding
        # self.sliding_list
        # self.simple_redundant_bp_fraction
        # self.redundancy_bp_count
        # self.unbind_bind_displacement_list

        self.tf_property_list = []

    def fill_tf_property_list(self):
        self.tf_property_list = [self.k_off, self.k_on, self.k_a, self.k_d, self.f_dna, self.mean_unbind_bind, self.mean_sliding, self.total_searched_fraction, self.mean_unbind_bind_displacement, self.simple_redundant_bp_fraction, self.total_time]

    def add_binding(self, bind_time):
        """
        :param bind_time: current time of simulation
        :return: update bind_times list once bound
        """
        self.bind_times[-1][1] = bind_time
        self.bind_pos.append(self.tf_instance.curr_pos[0])

        # unbind travel distance
        self.unbind_bind_bp['bind'].append(self.tf_instance.curr_pos[1][1])

        # slide distance (add tf size)
        half_tf_contact_size = (self.tf_instance.contact_size/2 * WLC.bp_len)  # add bp contact left and right edges
        self.sliding_min_max[0], self.sliding_min_max[1] = self.tf_instance.curr_pos[1][1]-half_tf_contact_size, self.tf_instance.curr_pos[1][1]+half_tf_contact_size

        self.bind_times.append([None, None])

    def add_unbinding(self, unbind_time, unbinding_bp):

        self.bind_times[-1][0] = unbind_time
        self.unbind_bind_bp['unbind'].append(unbinding_bp)
        self.unbind_pos.append(self.tf_instance.curr_pos[0])

        # sliding distance
        self.sliding_list.append((self.sliding_min_max[1] - self.sliding_min_max[0])/0.34)

        # add min max sliding to list
        try:
            min_bp, max_bp = int(round(self.sliding_min_max[0]/0.34)), int(round(self.sliding_min_max[1]/0.34))
        except TypeError:  # if sliding min or max is None
            pass
        else:
            self.sliding_min_max_list.append([min_bp, max_bp])
        # empty sliding min max
        self.sliding_min_max = [None, None]

    def update_sliding_min_max(self):

        half_tf_contact_size = (self.tf_instance.contact_size / 2 * WLC.bp_len)

        if self.tf_instance.curr_pos[1][1] - half_tf_contact_size < self.sliding_min_max[0] and self.sliding_min_max[0] is not None:
            self.sliding_min_max[0] = self.tf_instance.curr_pos[1][1] - half_tf_contact_size
        elif self.tf_instance.curr_pos[1][1] + half_tf_contact_size > self.sliding_min_max[1] and self.sliding_min_max[1] is not None:
            self.sliding_min_max[1] = self.tf_instance.curr_pos[1][1] + half_tf_contact_size


    def kon_and_kd(self):
        """
        Note: run at end of sim
        :return: return TF specific kon
        """

        try:
            self.bind_times[1]
        except IndexError:
            print('No values for binding time')
            self.diff_bind_times = None

        else:

            if len(self.bind_times) == 1:
                self.diff_bind_times = (self.bind_times[0][1] - self.bind_times[0][0]) * TF.time_scale

        # kon converted to seconds
        try:
            self.diff_bind_times = [float(self.bind_times[i][1] - self.bind_times[i][0]) * TF.time_scale for i in
                                   range(len(self.bind_times) - 1)]
        except TypeError:
            self.diff_bind_times = None

        # Catch runtime warning of np.mean operating on empty array, and act appropriately
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                self.bind_times_ave = np.mean(self.diff_bind_times, dtype=float)
            except RuntimeWarning:
                bind_times_ave = None

                self.k_on = 1 / TF.iter_count

                self.t_three_d = self.k_on ** (-1)

                # k_d
                self.k_d = (self.k_off / self.k_on)
                # k_a
                self.k_a = self.k_d ** (-1)

                self.f_dna = self.k_on / (self.k_on + self.k_off)

            else:
                self.k_on = 1 / self.bind_times_ave

                self.t_three_d = self.k_on ** (-1)

                # k_d
                self.k_d = (self.k_off / self.k_on)
                # k_a
                self.k_a = self.k_d ** (-1)

                self.f_dna = self.k_on / (self.k_on + self.k_off)

            self.total_time = TF.iter_count

    def calc_mean_unbind_bind(self):

        bind_list = self.unbind_bind_bp['bind'][1:]

        unbind_list = self.unbind_bind_bp['unbind']

        for i in range(len(bind_list)):
            try:
                self.bp_diff.append(abs(unbind_list[i]-bind_list[i]))
            except IndexError:
                pass

        try:
            self.mean_unbind_bind = sum(self.bp_diff)/len(self.bp_diff)
        except ZeroDivisionError:
            self.mean_unbind_bind = None

    def calc_r_g(self):
        """
        :return: tf radius of gyration (nm)
        """
        # PBC p.321

        # centre of mass
        r_cm = np.array(np.sum(self.tf_instance.positions)) / len(self.tf_instance.positions)

        # r_g^2
        r_g_sq_ave = np.sum([(i - r_cm) ** 2 for i in self.tf_instance.positions]) / len(self.tf_instance.positions)

        # r_g (nm)
        self.r_g = np.sqrt(r_g_sq_ave)

    def calc_mean_sliding(self):

        # calc mean sliding
        try:
            self.mean_sliding = sum(self.sliding_list)/len(self.sliding_list)
        except ZeroDivisionError:
            self.mean_sliding = None

    #def calc_MSD(self):


    def calc_total_and_redundant_sliding(self, dna):
        # produce non-redundant bp sliding list
        range_list = []

        for i in self.sliding_min_max_list:
            range_list += list(range(i[0], i[1] + 1))

        filtered = list(set(sorted(range_list)))

        seg_list = []

        curr_idx_start = 0
        curr_idx_check = 1

        # split into consecutive [start, stop]
        while curr_idx_check < len(filtered):
            if filtered[curr_idx_check] - filtered[curr_idx_check - 1] != 1:
                seg_list.append([filtered[curr_idx_start], filtered[curr_idx_check - 1]])
                curr_idx_start = np.copy(curr_idx_check)

            if curr_idx_check == len(filtered) - 1:
                seg_list.append([filtered[curr_idx_start], filtered[curr_idx_check]])

            curr_idx_check += 1

        self.non_redundant_slide = seg_list

        curr_bps_visited = 0

        for i in seg_list:
            curr_bps_visited += i[1] - i[0]

        self.total_searched_fraction = curr_bps_visited/dna.bp

        # produce bp redundancy list
        if len(self.sliding_min_max_list) > 1:
            # compare all ranges
            for count, i in enumerate(self.sliding_min_max_list):
                # stop if last index
                if count != len(self.sliding_min_max_list)-1:
                    # compare current range to all ranges following it
                    for j in self.sliding_min_max_list[count+1:]:
                        # append all bp values
                        for k in range(max(i[0], j[0]), min(i[-1], j[-1])+1):
                            self.overlap_list.append(k)

        # redundancy between 1D rounds
        # simple redundancy
        self.redundant_bp_list = set(self.overlap_list)

        self.simple_redundant_bp_fraction = len(self.redundant_bp_list)/dna.bp

        # bp wise redundancy
        self.redundancy_bp_count = {i: self.overlap_list.count(i) + 1 for i in self.redundant_bp_list}

    def calc_unbinding_binding_distance(self):

        bind_dict = {count: i for count, i in enumerate(self.bind_pos)}
        unbind_dict = {count: i for count, i in enumerate(self.unbind_pos)}

        for i in range(len(unbind_dict)):
            try:
                self.unbind_bind_displacement_list.append(np.linalg.norm(bind_dict[i + 1] - unbind_dict[i]))
            except KeyError:
                pass
        try:
            self.mean_unbind_bind_displacement = sum(self.unbind_bind_displacement_list)/len(self.unbind_bind_displacement_list)
        except ZeroDivisionError:
            self.mean_unbind_bind_displacement = None

    def update_p_bind_list(self):

        self.p_bind_list = self.tf_instance.p_bind_list

    def write_property_list(self):

        property_name_list = ['diff_bind_times', 'bp_diff', 'sliding_list',
                              'simple_redundant_bp_fraction', 'redundancy_bp_count',
                              'unbind_bind_displacement_list', 'Time']

        property_value_list = [self.diff_bind_times, self.bp_diff, self.sliding_list, self.simple_redundant_bp_fraction, self.redundancy_bp_count, self.unbind_bind_displacement_list, TF.iter_count]

        # add tmp file support
        f = tempfile.NamedTemporaryFile(mode='w+t', delete=False, dir='./output/tf_properties/', prefix='tmp_tfprop_',
                                        suffix='.txt')

        for j in range(len(property_name_list)):
            f.write(property_name_list[j] + '=' + str(property_value_list[j]) + '\n')
        f.close()

        f = tempfile.NamedTemporaryFile(mode='w+b', delete=False, dir='./output/tf_properties/', prefix='tmp_tfobj_', suffix='.obj')
        pickle.dump(self.tf_instance, f)
        f.close()


class TfData:
    def __init__(self):
        self.tf_df = None

    def append_metrics(self):
        for i in TF.active_tfs:
            self.tf_df = {System.property_names_list[j]: TfSimProp(i).tf_property_list[j] for j in range(len(TfSimProp(i).tf_property_list))}