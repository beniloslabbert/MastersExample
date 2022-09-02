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


class TF:
    # condition which keeps simulation running
    run_sim = True

    active_tfs = []
    deleted_tfs = []

    iter_count = 0

    boundaries = None

    time_scale = None

    stop_condition = None

    # passing instance variables to other class: https://stackoverflow.com/questions/19993795/how-would-i-access-variables-from-one-class-to-another

    def __init__(self, dna_object, boundaries=None, curr_pos=None, bind_dist=1, p_bind=1, targ_time_1d=5.0,
                 one_d_diff_constant=0.1, three_d_diff_constant=0.5, time_scale='nanosec', contact_size=1):
        """
        :param dna_object:
        :param boundaries:
        :param curr_pos: None will pick random position in box. 'box_edge' will initiate TF in a random position on the edge of the box.
        :param bind_dist: Marklund 2013, puts LacI reaction radius at 2.8 nm. Riggs 1970 0.5 nm (LacI). Berg 1976 6 nm (LacI). Diaz de la Roza 2010 10.6 nm (LacI). Mahmutovic 2015 5.5 nm (LacI and nsDNA) (Remember some might distinguish macro-/micro-scopic dissociation)
        :param p_bind: nm
        :param targ_time_1d: ms
        :param one_d_diff_constant: um^2/s
        :param three_d_diff_constant: um^2/s
        :param time_scale:
        """

        if curr_pos is None:  # Important, mutable default argument should be written like this.
            curr_pos = [np.array(
                ([np.random.uniform(dna_object.boundaries[count][0], dna_object.boundaries[count][1]) for count, _ in enumerate(range(3))])),
                        ['unbound',
                         None]]  # for completely random start: np.array(([np.random.uniform(dna_object.boundaries[0][0], dna_object.boundaries[0][1]) for _ in range(3)]))
        elif curr_pos == 'box_edge':
            curr_pos = [np.array(
                ([np.random.uniform(dna_object.boundaries[count][0], dna_object.boundaries[count][1]) for count, _ in enumerate(range(3))])),
                ['unbound',
                 None]]
            # choose x,y,z and -/+
            rand_coord = np.random.randint(0, 3)
            rand_pos_or_neg = np.random.randint(0, 2)
            if rand_pos_or_neg == 0:
                curr_pos[0][rand_coord] = dna_object.boundaries[rand_coord][0]
            else:
                curr_pos[0][rand_coord] = dna_object.boundaries[rand_coord][1]
            # other 2 should be unifromally chosen from one boundary to other. Done first.

        self.boundaries = boundaries
        # make DNA boundaries TF class attribute if passed in to instance
        if self.boundaries is not None and TF.boundaries is None:
            TF.boundaries = self.boundaries

        self.curr_pos = curr_pos  # specify at start
        self.positions = [self.curr_pos[0]]
        self.idx_bound_seg = None

        # handle time scale (these can be class atributes)
        if time_scale == 'nanosec':
            TF.time_scale = 1 / (10 ** 9)
            self.targ_time_1d = targ_time_1d * 10 ** 6
        elif time_scale == 'milisec':
            TF.time_scale = 1 / (10 ** 3)
            self.targ_time_1d = targ_time_1d
        elif time_scale == 'microsec':
            TF.time_scale = 1 / (10 ** 6)
            self.targ_time_1d = targ_time_1d * 10 ** 3

        # convert all len scales to nm
        self.one_d_diff_constant = one_d_diff_constant * ((10 ** 3) ** 2) * TF.time_scale
        self.three_d_diff_constant = three_d_diff_constant * ((10 ** 3) ** 2) * TF.time_scale

        self.step_size_1d = np.sqrt(
            2 * self.one_d_diff_constant)  # one_d_diff_constant and time_scale must be same units
        self.step_size_3d = np.sqrt(
            6 * self.three_d_diff_constant)  # three_d_diff_constant and time_scale must be same units

        self.bind_dist = bind_dist  # Must not be more than max 3D move dist!!!
        self.p_bind = p_bind
        self.p_bind_list = []
        self.curr_time_1d = 0
        self.contact_size = contact_size

        # TfSimProp instance becomes attribute of TF
        self.tfsimprop = TfSimProp(self)

        self.tfdata = TfData(self)

        """# dataframe variable, to be populated after run is done.
        self.TfData = None"""

        """
        #assert self.step_size_3d >= self.bind_dist  # for 1D unbinding to adhere to time
        """
        # create new TF
        TF.active_tfs.append(self)

    def sim(self, mode, dna_object, one_d_mode, stop_condition, target_iter=None):  # remember to pass dna object
        """
        :param mode:
        :param dna_object:
        :param one_d_mode: 'edge_fall' or 'edge_remain'
        :param stop_condition: 'iter_target' or 'bp_target'
            Note: iter_target stop built into sim method, bp_target built into 1D and 3D diffusion
        :param target_iter: None if stop_condition set to bp_target otherwise indicate rounds of sim
        :return:
        """

        # set TF stop condition
        TF.stop_condition = stop_condition

        while TF.run_sim:
            for i in TF.active_tfs:

                # i.curr_pos[1][0] has to be 'bound' or unbound
                assert i.curr_pos[1][0] == 'unbound' or i.curr_pos[1][0] == 'bound'

                if i.curr_pos[1][0] == 'unbound':
                    print('Started 3D')
                    self.three_d_diffusion(mode, dna_object, self.step_size_1d, one_d_mode)
                    print('Finished 3d')
                else:
                    print('Started 1D')
                    self.one_d_diffusion(dna_object, one_d_mode, mode)
                    print('Finished 1d')
                self.positions.append(self.curr_pos[0])
                TF.iter_count += 1
            print('iter count', TF.iter_count)

            # condition to stop sim, change to different running modes
            if TF.iter_count == target_iter and TF.stop_condition == 'iter_target':
                TF.run_sim = False
                return

        # after sim calculate kon, kd
        for i in TF.active_tfs:
            i.tfsimprop.kon_and_kd()

    def three_d_diffusion(self, mode, dna_object, step_size_1d, one_d_mode, ratio_displacement_realized=1,
                          curr_func=None):

        def targ_check_3D_1D():

            if dna_object.targets is None and TF.stop_condition == 'bp_target':
                return
            else:
                for i in dna_object.targets:  # may lose time. Think of doing remaining for targets
                    if i[1][0] <= self.curr_pos[1][1] <= i[1][1]:
                        TF.run_sim = False
                        return

        # used in both binding_trajectory_check and three_d_diffusion
        def test_binding(p1, p2, p3, r):
            """
            :param p1: First point of segment
            :param p2: Second point of segment
            :param p3: TF position
            :param r: Distance of TF to segment to check
            :return: Closest point from TF to segment if in range (r)
                     If TF not in range (r) return 'None'
            """

            def point_seg_dist(p1, p2, p3):  # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                """
                Run if part of segment in sphere (line_intersect_sphere returns '1')
                :param p1: First point of segment
                :param p2: Second point of segment
                :param p3: TF position
                :return: float(dist) closest part of segment to TF (p3)
                            Takes into account if situated between p1,p2 or not. If not
                """

                def perpendicular_point(p1, p2, p3):
                    """
                    :param p1: First point of segment
                    :param p2: Second point of segment
                    :param p3: TF position
                    :return: point (p) on line (p1,p2) perpendicular to TF (p3)
                    """

                    if type(p1) != type(np.array(0)):
                        p1 = np.array(p1)
                        p2 = np.array(p2)
                        p3 = np.array(p3)

                    t = -(np.dot(p1 - p3, p2 - p1)) / abs(np.dot(p2 - p1, p2 - p1))

                    # p x,y,z: https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
                    p = p1 + t * (p2 - p1)

                    return p

                dist = None
                closest_point_on_seg = None

                p1 = np.array(p1)
                p2 = np.array(p2)
                p3 = np.array(p3)

                # Is the point between the lines?
                vec_v = p2 - p1
                vec_w0 = p3 - p1
                vec_w1 = p3 - p2

                # For angles: https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
                # control for tf position being equal to one of segment sides
                if (p3 == p1).all():
                    closest_point_on_seg, dist = p1, 0.0
                elif (p3 == p2).all():
                    closest_point_on_seg, dist = p2, 0.0
                else:
                    theta0 = np.arccos(np.dot((vec_v / np.linalg.norm(vec_v)), (
                            vec_w0 / np.linalg.norm(
                        vec_w0))))  # use unit vec (absolute vec) to find angle using dot prod

                    theta1 = np.arccos(np.dot((vec_v / np.linalg.norm(vec_v)),
                                              (vec_w1 / np.linalg.norm(
                                                  vec_w1))))  # np.linalg.norm() gives len of vector

                    if theta0 < math.radians(90) < theta1:  # http://geomalgorithms.com/a02-_lines.html
                        closest_point_on_seg = perpendicular_point(p1, p2,
                                                                   p3)  # p3 falls between p1,p2 and perpendicular distance is closest. Perpendicular distance from continuous line, which intersects 2 given points (p1,p2), to point (p3)
                        dist = np.linalg.norm(p3 - closest_point_on_seg)
                    elif theta0 >= math.radians(90):
                        dist = np.linalg.norm(vec_w0)  # p1 is closest to p3
                        closest_point_on_seg = p1
                    elif theta1 <= math.radians(90):
                        dist = np.linalg.norm(vec_w1)  # p2 is closest to p3
                        closest_point_on_seg = p2
                    else:
                        print('All dist checks failed')
                return closest_point_on_seg, dist

            def line_intersect_sphere(p1, p2, p3, r):  # http://paulbourke.net/geometry/circlesphere/
                """
                :param p1: First point of segment
                :param p2: Second point of segment
                :param p3: TF position
                :param r: Distance of TF to segment to check
                :return: 'None' if no part of line in sphere
                         '1' if any part of line in sphere
                """

                result = None

                a = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2

                b = 2 * ((p2[0] - p1[0]) * (p1[0] - p3[0]) + (p2[1] - p1[1]) * (p1[1] - p3[1]) + (p2[2] - p1[2]) * (
                        p1[2] - p3[2]))

                c = p3[0] ** 2 + p3[1] ** 2 + p3[2] ** 2 + p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2 - 2 * (
                        p3[0] * p1[0] + p3[1] * p1[1] + p3[2] * p1[2]) - r ** 2

                quad_sol = b * b - 4 * a * c

                # If this is the case, keep diffusing
                if quad_sol < 0:
                    u_pos = 'undef'
                    u_neg = 'undef'
                    return result, u_pos, u_neg  # result should be none
                # If this is less than 0 then the line does not intersect the sphere.
                # If it equals 0 then the line is a tangent to the sphere intersecting it at one point, namely at u = -b/2a.
                # If it is greater then 0 the line intersects the sphere at two points.

                u_pos = (-b + math.sqrt(quad_sol)) / (2 * a)

                u_neg = (-b - math.sqrt(quad_sol)) / (2 * a)

                # if any of these are true bind to point (p) nearest the origin of the circle (p3)
                if (u_pos < 0 and u_neg > 1) or (u_neg < 0 and u_pos > 1):
                    # Line segment doesn't intersect and is inside sphere
                    result = 1
                    return result

                elif (0 < u_pos < 1 and (u_neg < 0 or u_neg > 1)) or (0 < u_neg < 1 and (u_pos < 0 or u_pos > 1)):
                    # Line segment intersects at one point
                    result = 1
                    return result

                elif (u_pos != u_neg and (0 < u_pos < 1 and 0 < u_neg < 1)):
                    # Line segment intersects at two points
                    result = 1
                    return result

                elif (u_pos == u_neg and (0 < u_pos < 1 and 0 < u_neg < 1)):
                    # Line segment is tangential to the sphere
                    result = 1
                    return result

            if line_intersect_sphere(p1, p2, p3, r) == 1:
                return point_seg_dist(p1, p2, p3)[0]
            else:
                return None

        def binding_trajectory_check(pot_position, dna_object):

            def genome_radius_check(curr_point, pot_point, dna_object):
                # box_low, box_high should be read from DNA instance
                box_low_x = dna_object.boundaries[0][0]
                box_high_x = dna_object.boundaries[0][1]
                box_low_y = dna_object.boundaries[1][0]
                box_high_y = dna_object.boundaries[1][1]
                box_low_z = dna_object.boundaries[2][0]
                box_high_z = dna_object.boundaries[2][1]
                # centre should be half way between TF_start and TF_end
                centre = np.array(curr_point + (pot_point - curr_point) / 2)
                centres = centre.reshape(1, (len(centre)))
                # upper_radii should be half distance from TF_start to TF_end + tf bind_dist
                upper_radii = np.linalg.norm(pot_point - centre) + self.bind_dist

                # Build the grid with data
                gsp = GriSPy(dna_object.cum_sum)
                periodic = {0: (box_low_x, box_high_x), 1: (box_low_y, box_high_y), 2: (box_low_z, box_high_z)}
                gsp.set_periodicity(periodic)

                # Query for neighbors with upper_radii
                bubble_dist, bubble_ind = gsp.bubble_neighbors(
                    centres, distance_upper_bound=upper_radii
                )
                # sort index by dist
                sorted_bubble_ind = [x for y, x in sorted(zip(bubble_dist[0], bubble_ind[0]))]
                # bubble_ind = index in 'data' of point in bubble
                # bubble_dist = distance of point, from corresponding index, to TF
                return [dna_object.cum_sum[i] for i in sorted_bubble_ind], sorted_bubble_ind  # radii_points, index

            def test_binding_multi_sphere(p1, p2, tf_start, tf_end,
                                          r):  ### think of using scikit-spacial to make this more efficient
                """
                :param p1: one point of dna segment being checked
                :param p2: other point of dna segment being checked
                :param tf_start: start of tf trajectory
                :param tf_end: potential end of tf trajectory
                :param r: self.bind_dist
                :return: Closest point from TF to segment if in range (r)
                         If TF not in range (r) return 'None'
                """
                list_of_points = []

                # tf coordinates start and end
                vec = np.array(tf_end) - np.array(tf_start)

                # divide tf_unit_vector into segments of size r

                unit_vec = vec / np.linalg.norm(vec)

                if np.all([np.isnan(i) for i in unit_vec]):
                    unit_vec = [0, 0, 0]
                else:
                    unit_vec = vec / np.linalg.norm(vec)
                amt_whole_unit_vec_segs = int(np.linalg.norm(vec) // r)

                # if there are no points to check
                if amt_whole_unit_vec_segs == 0:
                    pass  # should this be pass?
                else:
                    for i in range(amt_whole_unit_vec_segs + 1):
                        list_of_points.append(tf_start + (i * unit_vec))

                list_of_points.append(tf_end)

                # do test_binding on each of those points
                for i in list_of_points:
                    binding_test = test_binding(p1, p2, i, r)
                    # p = probability to bind. test at each test_binding
                    if binding_test is not None:# and np.random.uniform() <= self.p_bind: # filter for points close together
                        return binding_test

                # return point if found, else return None
                return

            def filter_binding():
                """
                filter segments availilble for binding and to probability check
                :return: segments_to_check
                """

                # create cylinder
                cylinder_vec = pot_position - self.curr_pos[0]

                # cylander vec may be 0
                try:
                    trajectory_cylinder = sk_objects.Cylinder(self.curr_pos[0], cylinder_vec, self.bind_dist)
                except ValueError:
                    cylinder_vec = self.curr_pos[0] - np.array([0.000001, 0.000001, 0.000001])

                # filter if not in cylinder
                grispy_points_dict = {tf_trajectory_sphere[1][j]: tf_trajectory_sphere[0][j] for j in
                                      range(len(tf_trajectory_sphere[0]))}

                # remove max dna point point
                try:
                    # grispy_points_dict[max(dna_object.bp_dict, key=dna_object.bp_dict.get)]
                    grispy_points_dict[max(dna_object.bp_dict) + 1], max(dna_object.bp_dict)
                except KeyError:
                    pass
                else:
                    grispy_points_dict.pop(max(dna_object.bp_dict) + 1)

                dict_copy = grispy_points_dict.copy()

                for j in dict_copy.keys():  ### j needs to be coordinates
                    if not trajectory_cylinder.is_point_within(grispy_points_dict[j]):
                        grispy_points_dict.pop(j)

                # get line-point distance
                trajectory_line = sk_objects.Line(self.curr_pos[0], cylinder_vec)
                point_line_distance_list = [trajectory_line.distance_point(grispy_points_dict[j]) for j in
                                            grispy_points_dict.keys()]

                # point:dist dict
                point_line_dict = {j: point_line_distance_list[count] for count, j in
                                   enumerate(grispy_points_dict.keys())}
                sorted_point_line_dict = dict(sorted(point_line_dict.items(), key=operator.itemgetter(1)))

                # get bp dict values
                try:
                    filtered_bp_dict = {j: dna_object.bp_dict[j] for j in sorted_point_line_dict.keys()}
                except KeyError:
                    print('sorted_point_line_dict', sorted_point_line_dict.keys())
                    sys.exit()
                # remove all which are not 25nm apart
                filtered_bp_dict_list = list(filtered_bp_dict.items())

                selected_index = 0
                filter_dict = True

                while filter_dict:

                    if filtered_bp_dict_list == []:
                        break

                    sliced_list = filtered_bp_dict_list[(selected_index + 1):]

                    ### what should distance cutoff be?
                    for j in sliced_list:
                        if filtered_bp_dict_list[selected_index][1][-1] - 25 < j[1][-1] < \
                                filtered_bp_dict_list[selected_index][1][-1] + 25:
                            filtered_bp_dict_list.pop(filtered_bp_dict_list.index(j))

                    selected_index += 1

                    if selected_index == len(filtered_bp_dict_list):
                        filter_dict = False

                if filtered_bp_dict_list:
                    print(filtered_bp_dict_list[0][1][0])

                # from points in sphere produce p1, p2 pairs
                segments_to_check = []

                for j in range(len(filtered_bp_dict_list)):
                    segments_to_check.append([filtered_bp_dict_list[j][1][0][0], filtered_bp_dict_list[j][1][0][1]])

                return segments_to_check

            point = None
            # possible genome binding points: Grispy
            tf_trajectory_sphere = genome_radius_check(self.curr_pos[0], pot_position, dna_object)

            ### start filtering here
            segments_to_check = filter_binding()
            print('directly following filter_binding') # if this shows error happens after this

            # p_bind for entire binding event if there is a possible segment in trajectory cylinder
            if segments_to_check is not None and np.random.random() < self.p_bind:
                self.p_bind_list.append(True)

                # check binding on trajectory
                for count, i in enumerate(segments_to_check):
                    point = test_binding_multi_sphere(p1=i[0], p2=i[1], tf_start=self.curr_pos[0], tf_end=pot_position,
                                                      r=self.bind_dist)

                    if point is not None:
                        # set 'bound' if bound to dna eg. if 'point' returns something
                        self.curr_pos[1][0] = 'bound'  # this shouldn't be an issue
                        print('point', point)

                        # search for 'point' idx position in bp_dict
                        for list_count, k in enumerate(list(dna_object.bp_dict.values())):
                            if any(point in j for j in dna_object.bp_dict[list_count][0]):
                                self.idx_bound_seg = list_count
                                print('idx_check', self.idx_bound_seg)
                        return point
            else:
                self.p_bind_list.append(False)

            print('directly following segments_to_check')  # if this shows error happens after this
            # point = ([x,y,z], dist), point_seg_dist() return value
            return pot_position

        def resid_dynam_3D_1D(ratio_3d_realized, dna_object, one_d_mode, mode):
            # do 1D according to what remains
            print('3D to 1D resid')
            self.one_d_diffusion(dna_object, one_d_mode, mode, ratio_3d_realized)

        def bp_position_updating(dna_object):
            """
            :param dna_object:
            :return:
            Purpose:
                check whether bound point on seg
                find and update TF bp position
            """

            def point_is_between(a, b, c):

                a = np.array(a)
                b = np.array(b)
                c = np.array(c)

                ab = b - a
                ac = c - a

                # crossproduct parallel: https://mathworld.wolfram.com/ParallelVectors.html
                crossproduct = np.cross(ab, ac)
                comparison = abs(crossproduct) == np.zeros(np.shape(crossproduct))
                equal_arrays = comparison.all()

                # crossprod should be 0 vec and all c coords should be between a and b
                return bool(equal_arrays and all(a[i] <= c[i] <= b[i] for i in range(len(c))))

            # coords of bound seg
            point1, point2 = dna_object.bp_dict[self.idx_bound_seg][0]
            # is point on line seg
            if not point_is_between(point1, point2, self.curr_pos[0]):
                print('bound point is not on line')
            # find bp position and set to self.curr_pos[1][1]
            final_point_to_bound_len = np.linalg.norm(
                point2 - self.curr_pos[0])  # dist between final point and bound point
            self.curr_pos[1][1] = dna_object.bp_dict[self.idx_bound_seg][
                                      1] - final_point_to_bound_len  # bp len at last point on seg min above

            # control for numerical imprecision: if bp position over max or min, set to max or min respectively
            if self.curr_pos[1][1] < 0:
                self.curr_pos[1][1] = 0.0
            elif self.curr_pos[1][1] > dna_object.bp_dict[max(dna_object.bp_dict)][1]:
                self.curr_pos[1][1] = dna_object.bp_dict[max(dna_object.bp_dict)][1]

        def on_edge_three_d_diffusion(current_position, ratio_displacement_realized):

            # produce appropriate step given which boundary TF is on

            pot_position = generic_three_d_diffusion(current_position, ratio_displacement_realized)

            # is greater than bound? which axis/coord. Set those to boundary coord
            for count, i in enumerate(pot_position):
                if (TF.boundaries[count][0] > i):
                    pot_position[count] = TF.boundaries[0][0]
                elif (TF.boundaries[count][1] < i):
                    pot_position[count] = TF.boundaries[0][1]

            return pot_position

        def generic_three_d_diffusion(current_position, ratio_displacement_realized):

            def spherical_coords():
                theta = np.random.uniform(-np.pi, np.pi)
                phi = np.random.uniform(-np.pi, np.pi)
                rad = abs(np.random.normal(0, 1))

                spherical_coord = np.array(
                    [(rad * math.cos(theta) * math.sin(phi)),
                     (rad * math.sin(theta) * math.sin(phi)),
                     (rad * math.cos(phi))])

                return spherical_coord

            pot_position = current_position

            new_step = spherical_coords() * self.step_size_3d * ratio_displacement_realized
            pot_position += new_step

            return pot_position

        def replace_tf(self, step_size_1d):  # currently not used anywhere
            self.remove_tf()
            self.add_tf(step_size_1d)

        def add_tf(step_size_1d):

            def random_point_on_box():
                x_y_z_constant = np.random.randint(0, 3)
                # x constant
                if x_y_z_constant == 0:
                    # max x, min x varying y,z
                    if np.random.randint(0, 2) == 0:
                        x, y, z = TF.boundaries[0][0], np.random.uniform(TF.boundaries[1][0],
                                                                         TF.boundaries[1][1]), np.random.uniform(
                            TF.boundaries[2][0], TF.boundaries[2][1])
                    else:
                        x, y, z = TF.boundaries[0][1], np.random.uniform(TF.boundaries[1][0],
                                                                         TF.boundaries[1][1]), np.random.uniform(
                            TF.boundaries[2][0], TF.boundaries[2][1])
                    curr_pos = [x, y, z]
                # y constant
                elif x_y_z_constant == 1:
                    # max y min y varying x,z
                    if np.random.randint(0, 2) == 0:
                        x, y, z = np.random.uniform(TF.boundaries[0][0], TF.boundaries[0][1]), TF.boundaries[1][
                            0], np.random.uniform(TF.boundaries[2][0], TF.boundaries[2][1])
                    else:
                        x, y, z = np.random.uniform(TF.boundaries[0][0], TF.boundaries[0][1]), TF.boundaries[1][
                            1], np.random.uniform(TF.boundaries[2][0], TF.boundaries[2][1])
                    curr_pos = [x, y, z]
                # z constant
                elif x_y_z_constant == 2:
                    # max z min z varying x,y
                    if np.random.randint(0, 2) == 0:
                        x, y, z = np.random.uniform(TF.boundaries[0][0], TF.boundaries[0][1]), np.random.uniform(
                            TF.boundaries[1][0], TF.boundaries[1][1]), TF.boundaries[2][0]
                    else:
                        x, y, z = np.random.uniform(TF.boundaries[0][0], TF.boundaries[0][1]), np.random.uniform(
                            TF.boundaries[1][0], TF.boundaries[1][1]), TF.boundaries[2][1]
                    curr_pos = [x, y, z]
                else:
                    print('No index chosen')
                    return
                return curr_pos

            # where new TF should be created
            print('create new tf')
            TF.__init__(self, step_size_1d, curr_pos=[random_point_on_box(), ['unbound', None]], bind_dist=1,
                        p_bind=1)
            # first move should be away from box: done in .on_edge_three_d_diffusion()

        def remove_tf():
            TF.deleted_tfs.append(self)
            TF.active_tfs.remove(self)

        # Copy not set equal!!! Otherwise the variable stays attached to original array: https://stackoverflow.com/questions/30683166/numpy-array-values-changed-without-being-aksed
        pot_position = np.copy(self.curr_pos[0])
        prev_position = np.copy(self.curr_pos[0])

        # generate potential new position
        if any(item in list(self.curr_pos[0]) for item in [j for i in TF.boundaries for j in i]):  # compare tf position coords with boundaries
            pot_position = on_edge_three_d_diffusion(pot_position,
                                                     ratio_displacement_realized)  # if on edge move away from edge
        else:
            pot_position = generic_three_d_diffusion(pot_position,
                                                     ratio_displacement_realized)  # if not on edge. Changed to 1

        if curr_func == '1D_3D':
            self.curr_pos[0] = pot_position
            return

        # do 3d diffusion if in box
        if TF.boundaries[0][0] < pot_position[0] < TF.boundaries[0][1] and TF.boundaries[1][0] < pot_position[1] < TF.boundaries[1][1] and TF.boundaries[2][0] < pot_position[2] < TF.boundaries[2][1]:

            # check binding on trajectory

            self.curr_pos[0] = binding_trajectory_check(pot_position, dna_object)

            # 'bound' is set in above function if bound
            if self.curr_pos[1][0] == 'bound':
                # find bp position and set to self.curr_pos[1][1]
                bp_position_updating(dna_object)

                # add binding time for TF
                self.tfsimprop.add_binding(TF.iter_count)

                ### put target check
                if TF.stop_condition == 'bp_target':
                    targ_check_3D_1D()
                    if not TF.run_sim:  # TF.run_sim set to false if on target
                        print('end sim')
                        return

                curr_pos_displaced = abs(np.linalg.norm(self.curr_pos[0] - prev_position))
                pot_pos_displaced = abs(np.linalg.norm(pot_position - prev_position))

                if curr_pos_displaced - self.bind_dist < pot_pos_displaced < curr_pos_displaced + self.bind_dist:  # if final position is where it bound from set ratio_3d = 1
                    ratio_3d_realized = 1
                else:
                    ratio_3d_realized = 1 - (abs(np.linalg.norm(self.curr_pos[0] - prev_position)) / abs(
                        np.linalg.norm(pot_position - prev_position)))
                    resid_dynam_3D_1D(ratio_3d_realized, dna_object, one_d_mode, mode)
                assert ratio_3d_realized >= 0

        # if not in box check 'boundary mode' and react appropriately
        else:
            # TF 3D diffusion should stay within box
            if mode == 'boundary_interaction':
                # check binding on trajectory. If no binding curr_pos[0] = pot_position
                self.curr_pos[0] = binding_trajectory_check(pot_position, dna_object)

                # 'bound' is set in above function if bound
                if self.curr_pos[1][0] == 'bound':
                    # find bp position and set to self.curr_pos[1][1]
                    bp_position_updating(dna_object)

                    # add binding time for TF
                    self.tfsimprop.add_binding(TF.iter_count)

                    ### put target check
                    if TF.stop_condition == 'bp_target':
                        targ_check_3D_1D()
                        if not TF.run_sim:  # TF.run_sim set to false if on target
                            return

                    ratio_3d_realized = 1 - (abs(np.linalg.norm(self.curr_pos[0] - prev_position)) / abs(
                        np.linalg.norm(pot_position - prev_position)))

                    #assert ratio_3d_realized >= self.bind_dist
                    resid_dynam_3D_1D(ratio_3d_realized, dna_object, one_d_mode, mode)
                else:
                    # box interaction
                    if not TF.boundaries[0][0] < self.curr_pos[0][0] < TF.boundaries[0][1]:  # x bound
                        if TF.boundaries[0][0] > self.curr_pos[0][0]:  # min
                            self.curr_pos[0][0] = TF.boundaries[0][0]
                        elif self.curr_pos[0][0] > TF.boundaries[0][1]:  # max
                            self.curr_pos[0][0] = TF.boundaries[0][1]
                    if not TF.boundaries[1][0] < self.curr_pos[0][1] < TF.boundaries[1][1]:  # y bound
                        if TF.boundaries[1][0] > self.curr_pos[0][1]:  # min
                            self.curr_pos[0][1] = TF.boundaries[1][0]
                        elif self.curr_pos[0][1] > TF.boundaries[1][1]:  # max
                            self.curr_pos[0][1] = TF.boundaries[1][1]
                    if not TF.boundaries[2][0] < self.curr_pos[0][2] < TF.boundaries[2][1]:  # z bound
                        if TF.boundaries[2][0] > self.curr_pos[0][2]:  # min
                            self.curr_pos[0][2] = TF.boundaries[2][0]
                        elif self.curr_pos[0][2] > TF.boundaries[2][1]:  # max
                            self.curr_pos[0][2] = TF.boundaries[2][1]

            # TF 3D diffusion should leave box and new TF created
            elif mode == 'no_boundary_interaction':
                self.curr_pos[0] = binding_trajectory_check(pot_position, dna_object)
                # 'bound' is set in above function if bound
                if self.curr_pos[1][0] == 'bound':
                    # find bp position and set to self.curr_pos[1][1]
                    bp_position_updating(dna_object)

                    # add binding time for TF
                    self.tfsimprop.add_binding(TF.iter_count)

                    ### put target check
                    if TF.stop_condition == 'bp_target':
                        targ_check_3D_1D()
                        if not TF.run_sim:  # TF.run_sim set to false if on target
                            return

                    ratio_3d_realized = 1 - (abs(np.linalg.norm(self.curr_pos[0] - prev_position)) / abs(
                        np.linalg.norm(pot_position - prev_position)))
                    assert ratio_3d_realized >= 0
                    resid_dynam_3D_1D(ratio_3d_realized, dna_object, one_d_mode, mode)
                else:
                    # if beyond boundaries, stop running that TF: specified in sim method
                    remove_tf()
                    # create new TF to enter box. First step guarantee movement into box, but after that make random?
                    add_tf(step_size_1d)
                return

        ### check interaction: dna
        if self.curr_pos[1][0] == 'unbound':  # if bound by resid skip this 3D point search
            possible_binding = []
            # use Grispy, fixed radius nearest neighbors, to get points close to cur_pos
            for i in range(len(dna_object.cum_sum) - 1):
                curr_binding = test_binding(dna_object.cum_sum[i], dna_object.cum_sum[i + 1], self.curr_pos[0],
                                            self.bind_dist)  # curr_binding = (closest_point_on_seg, dist)
                if curr_binding is not None:
                    possible_binding.append({i: curr_binding})

    def sliding_distance_update(self):

        if self.curr_pos[1][1] < self.tfsimprop.sliding_min_max[0] or self.tfsimprop.sliding_min_max[0] is None:
            self.tfsimprop.sliding_min_max[0] = self.curr_pos[1][1]

        elif self.curr_pos[1][1] > self.tfsimprop.sliding_min_max[1] or self.tfsimprop.sliding_min_max[0] is None:
            self.tfsimprop.sliding_min_max[1] = self.curr_pos[1][1]

    def one_d_diffusion(self, dna_object, one_d_mode, mode, ratio_displacement_realized=1,
                        unbinding_mode='spherical_unbinding', curr_func=None):
        """
        :param dna_object:
        :param one_d_mode:
        :param mode:
        :param ratio_displacement_realized:
        :param unbinding_mode: 'spherical_unbinding' or 'perpendicular_unbinding'
        :return:
        """


        def dna_edge(dna_object):
            """
            :param dna_object: generated dna object
            :param displacement: diplacement for 1D step
            :param end_len: bp position for last point of seg
            :return: update self.curr_pos[1][1] for displacement and find current seg
            """
            # control for over min/max bp value. Careful with this, not just copy to dna_edge_fall
            if self.curr_pos[1][1] < 0:
                self.curr_pos[1][1] = 0.0
            elif self.curr_pos[1][1] > dna_object.bp_dict[max(dna_object.bp_dict)][1]:
                self.curr_pos[1][1] = dna_object.bp_dict[max(dna_object.bp_dict)][1]

            # update idx_bound
            # if not in segment len, find seg
            curr_idx = self.idx_bound_seg
            # find segment for given final bp position
            filterd_dict = dict(filter(lambda elem: abs(elem[1][1] - self.curr_pos[1][1]) <= dna_object.s_link,
                                       dna_object.bp_dict.items()))
            target_idx = max(filterd_dict.keys())

            # stay bound right at edge
            self.idx_bound_seg = target_idx

            # if tf is at edge of dna
            is_tf_max = self.curr_pos[1][1] == dna_object.bp_dict[max(dna_object.bp_dict)][1]
            is_tf_min = self.curr_pos[1][1] == dna_object.bp_dict[min(dna_object.bp_dict)][1] - dna_object.s_link

            # segment_to_end_up_on
            filterd_dict = dict(filter(lambda elem: abs(elem[1][1] - self.curr_pos[1][1]) <= dna_object.s_link,
                                       dna_object.bp_dict.items()))
            segment_to_end_up_on = dna_object.bp_dict[max(filterd_dict)]

            return is_tf_max, is_tf_min, segment_to_end_up_on

        def resid_dynam_1D_3D(ratio_displacement_realized, mode, one_d_mode, dna_object, step_size_1d,
                              curr_func='1D_3D'):
            # do 3D according to what remains (control for immediate rebinding?)
            print('1D to 3D resid')
            self.three_d_diffusion(mode, dna_object, step_size_1d, one_d_mode, ratio_displacement_realized, curr_func)

        def resid_dynam_1D_1D(ratio_displacement_realized, direction, dna_object, one_d_mode, mode,
                              curr_func='1D_1D'):
            # do 1D according to what remains (in opposite direction from edge)
            print('1D to 1D resid')
            self.one_d_diffusion(dna_object, one_d_mode, mode, ratio_displacement_realized, curr_func)

        def truncated_standard_norm(min_guess, max_guess):
            guess = 10 ** 6
            while guess < min_guess or guess > max_guess:
                guess = np.random.normal(0, abs(max_guess))

            return guess

        def three_d_diffusion_1D_3D(current_position, ratio_displacement_realized):
            print('1D to 3D')
            old_current_position = current_position

            def random_range(bind_dist):
                if np.random.randint(0, 2) == 0:
                    # random_range neg
                    return truncated_standard_norm(-bind_dist * 5, -bind_dist)
                else:
                    # random_range pos
                    return truncated_standard_norm(bind_dist, bind_dist * 5)

            # run after unbinding
            # ensure unbinding coord > binding distance (remember that this is the case. Keep track of remaining 3D to complete on start of next round.)
            new_step = random_range(self.bind_dist)  # * ratio_displacement_realized
            # self.bind_dist#* self.step_size_3d  # should I include ratio_displacement? May make less than min unbind dist
            current_position += new_step
            return current_position

        def update_tf_pos():
            """
            Update the TFs 3D coordinate position based following 1D movement.

            Note: The function is robust, because it returns the coordinates based on the 1D position,
            thus if its wrong once that error will not be amplified
            :return: The updated 3d coordinate of tf.
            """
            # update coords
            dist_from_end_point = dna_object.bp_dict[self.idx_bound_seg][1] - self.curr_pos[1][1]

            # uv of current seg
            vec = (dna_object.bp_dict[self.idx_bound_seg][0][1] - dna_object.bp_dict[self.idx_bound_seg][0][0])

            unit_vec = vec / np.linalg.norm(vec)

            # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
            self.curr_pos[0] = dna_object.bp_dict[self.idx_bound_seg][0][1] + dist_from_end_point * unit_vec

        def generate_new_vec_perp(curr_pos_coord):  # perpendicular point from seg

            theta = np.pi / 2  # perpendicular solution np.pi/2
            phi = np.random.uniform(-np.pi, np.pi)
            rad = truncated_standard_norm(cylinder_r, sphere_r)

            spherical_coord = np.array(
                [(rad * math.cos(theta) * math.sin(phi)),
                 (rad * math.sin(theta) * math.sin(phi)),
                 (rad * math.cos(phi))])  # generate new vector

            return spherical_coord + self.curr_pos[0]

        def generate_new_vec_satisfy(curr_pos_coord):  # spherical point between sphere_r and cylinder_r from seg
            spherical_coord = np.copy(curr_pos_coord)  # make cur xyz
            while True:  # run atleast once to set spherical coord different to curr_pos_coord
                theta = np.random.uniform(-np.pi, np.pi)
                phi = np.random.uniform(-np.pi, np.pi)
                rad = truncated_standard_norm(cylinder_r, (rad_range + cylinder_r))

                spherical_coord = np.array(
                    [(rad * math.cos(theta) * math.sin(phi)),
                     (rad * math.sin(theta) * math.sin(phi)),
                     (rad * math.cos(phi))])  # generate new vector
                if not cylinder.is_point_within(spherical_coord):  # point should be outside cylinder
                    break
            return spherical_coord + self.curr_pos[0]


        # add check if current method is 1D_1D resid, it should not run unbind check
        if self.targ_time_1d - 1 <= self.curr_time_1d:  # check unbinding before 1D starts

            # set time to 0 for next round
            self.curr_time_1d = 0

            prev_coord_position = np.copy(self.curr_pos[0])
            potential_position = three_d_diffusion_1D_3D(np.copy(prev_coord_position),
                                                         1)  # this is only to get potential binding, for eventual undibing below
            potential_displacement = np.linalg.norm(
                potential_position - prev_coord_position)  # generate potential 3D displacement (for resid_dynam_1D_3D used in resid_dynam_1D_3D)            # unbind:
            unbinding_bp = self.curr_pos[1][1]
            # change TF self.curr_pos: curr_pos = [np.array([x0,y0,z0]), ['bound', bp_pos]] -> [np.array([x1,y1,z1]), ['unbound', None]]
            self.curr_pos[1][0], self.curr_pos[1][1] = 'unbound', None
            self.tfsimprop.add_unbinding(TF.iter_count, unbinding_bp)

            # self.curr_pos[0] = find new coords
            sphere_r = 1.01 * self.bind_dist  # unbind dist
            cylinder_r = self.bind_dist  # bind dist
            start_seg_point, stop_seg_point = dna_object.bp_dict[self.idx_bound_seg][0][0], \
                                              dna_object.bp_dict[self.idx_bound_seg][0][1]
            dir_vec = start_seg_point - stop_seg_point
            cylinder = sk_objects.Cylinder(stop_seg_point, dir_vec,
                                        cylinder_r)  # (dna_object.bp_dict[self.idx_bound_seg][0][1], )
            sphere = sk_objects.Sphere(self.curr_pos[0], sphere_r)

            rad_range = sphere_r - cylinder_r

            if unbinding_mode == 'spherical_unbinding':
                self.curr_pos[0] = generate_new_vec_satisfy(self.curr_pos[0])
            elif unbinding_mode == 'perpendicular_unbinding':
                self.curr_pos[0] = generate_new_vec_perp(self.curr_pos[0])

            actual_displacement = np.linalg.norm(self.curr_pos[0] - prev_coord_position)
            resid_dynam_1D_3D_val = 1 - (abs(actual_displacement) / abs(potential_displacement))
            assert resid_dynam_1D_3D_val >= 0

            # do 3D resid, guaranteed unbinding from current seg
            # if resid_dynam_1D_3D_val < 1:
            resid_dynam_1D_3D(resid_dynam_1D_3D_val, mode, one_d_mode, dna_object, self.step_size_1d)
            return  # if unbound, should not try to continue with 1D
            # then test whether correct amount of rounds occur and tf position continuous (remember to adject 'edge_fall' match 'edge_remain')

        assert self.curr_pos[1][1] is not None  # if bp position is None, TF has unbound, or something has broken

        # adjust for if tf on edge. Might be an issue becuase I may not want this to happen in other mode
        if self.curr_pos[1][1] > dna_object.bp_dict[max(dna_object.bp_dict)][1]:  # max
            direction = truncated_standard_norm(-5, 0)
        elif self.curr_pos[1][1] <= 0.0:  # min
            direction = truncated_standard_norm(0, 5)
        else:
            direction = np.random.normal(0, 1)
        displacement = direction * self.step_size_1d * ratio_displacement_realized
        end_len = dna_object.bp_dict[self.idx_bound_seg][1]  # bp position for last point of seg
        # update bp position
        prev_bp_position = np.copy(self.curr_pos[1][1])  # only used if residual times needed
        self.curr_pos[1][1] = self.curr_pos[1][1] + displacement

        # unbind increment time
        self.curr_time_1d += ratio_displacement_realized  # Remember timestep is 1 by default. Ratio captures this.

        ### put target check
        if dna_object.targets is not None and TF.stop_condition == 'bp_target':

            for i in dna_object.targets:  # may lose time. Think of doing remaining for targets
                if prev_bp_position <= i[1][0] <= self.curr_pos[1][1] or self.curr_pos[1][1] <= i[1][
                    0] <= prev_bp_position or \
                        prev_bp_position <= i[1][1] <= self.curr_pos[1][1] or self.curr_pos[1][1] <= i[1][
                    1] <= prev_bp_position:
                    TF.run_sim = False
                    print('TF.run_sim = False')
                    return


        is_tf_max, is_tf_min, segment_to_end_up_on = dna_edge(dna_object)

        # how to deal with edges
        if one_d_mode == 'edge_fall':

            # control for edges, ensure TF moves away from edge for remaining dynamics
            if is_tf_max or is_tf_min:
                print('edge_fall')
                actual_displacement = self.curr_pos[1][1] - prev_bp_position
                # once hit edge, what time remains?
                ratio_displacement_realized = 1 - (abs(actual_displacement) / abs(displacement))  # think of renaming
                assert ratio_displacement_realized >= 0
                update_tf_pos()
                self.tfsimprop.update_sliding_min_max()
                self.sliding_distance_update()

                unbinding_bp = self.curr_pos[1][1]
                self.tfsimprop.add_unbinding(TF.iter_count, unbinding_bp)

                # do remaining dynamics
                # this depends on direction set above
                self.curr_pos[1][0], self.curr_pos[1][1] = 'unbound', None
                resid_dynam_1D_3D(ratio_displacement_realized, mode, one_d_mode, dna_object, self.step_size_1d)
                return
        elif one_d_mode == 'edge_remain':

            # control for edges, ensure TF moves away from edge for remaining dynamics
            if is_tf_max or is_tf_min:
                actual_displacement = self.curr_pos[1][1] - prev_bp_position
                # once hit edge, what time remains?
                ratio_displacement_realized = 1 - (abs(actual_displacement) / abs(displacement))  # think of renaming
                assert ratio_displacement_realized >= 0
                update_tf_pos()
                self.tfsimprop.update_sliding_min_max()
                self.sliding_distance_update()

                # do remaining dynamics
                # this depends on direction set above
                resid_dynam_1D_1D(ratio_displacement_realized, direction, dna_object, one_d_mode, mode)

        update_tf_pos()
        self.tfsimprop.update_sliding_min_max()
        self.sliding_distance_update()

    @staticmethod
    def perpendicular_distance(p1, p2, p3):
        """
        :param p1: First point of segment
        :param p2: Second point of segment
        :param p3: TF position
        :return: distance from point (p) tor TF (p3)
        """

        if type(p1) != type(np.array(0)):
            p1 = np.array(p1)
            p2 = np.array(p2)
            p3 = np.array(p3)

        t = -(np.dot(p1 - p3, p2 - p1)) / abs(np.dot(p2 - p1, p2 - p1))

        sq_dist = ((p1[0] - p3[0]) + (p2[0] - p1[0]) * t) ** 2 + ((p1[1] - p3[1]) + (p2[1] - p1[1]) * t) ** 2 + (
                (p1[2] - p3[2]) + (p2[2] - p1[2]) * t) ** 2

        return np.sqrt(sq_dist)


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
        half_tf_contact_size = (self.tf_instance.contact_size/2 * 0.34)  # add bp contact left and right edges
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

        half_tf_contact_size = (self.tf_instance.contact_size / 2 * 0.34)

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
    property_names_list = ['k_off', 'k_on', 'k_a', 'k_d', 'f_dna', 'mean_unbind_bind', 'mean_sliding', 'total_searched_fraction', 'mean_unbind_bind_displacement', 'simple_redundant_bp_fraction', 'total_time']

    def __init__(self, tf_instance):
        self.tf_df = {}

        self.tf_instance = tf_instance

    def append_metrics(self):
        for i in TF.active_tfs:
            self.tf_df = {TfData.property_names_list[j]: i.tfsimprop.tf_property_list[j] for j in range(len(i.tfsimprop.tf_property_list))}
