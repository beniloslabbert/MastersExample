from System import System
from Dna import WLC, Target
from Diffusers import TF

import pandas as pd
import csv
import datetime
import os.path
import traceback
from parameters import parameters_10k_confined as par
import generate_dna as pic
import tempfile
import shutil
from sys import platform
import re


def simulate():

    wlc = pic.pickle_load(par.DNA_OBJECT)

    ###

    # tfs
    tfs = TF(wlc, boundaries=wlc.boundaries, targ_time_1d=par.TARG_TIME_1D, one_d_diff_constant=par.ONE_D_DIFF_CONSTANT, three_d_diff_constant=par.THREE_D_DIFF_CONSTANT,
             bind_dist=par.BIND_DIST, time_scale=par.TIME_SCALE, p_bind=par.P_BIND, contact_size=par.CONTACT_SIZE_BP)  # um/s

    system = System(wlc, tfs, theoretical_time=None)

    tfs.sim(mode=par.MODE, dna_object=wlc, one_d_mode=par.ONE_D_MODE, stop_condition=par.STOP_CONDITION,
            target_iter=par.TARGET_ITER)

    # after sim calculate kon, kd
    for i in TF.active_tfs:

        i.tfsimprop.kon_and_kd()
        i.tfsimprop.calc_r_g()
        i.tfsimprop.calc_mean_unbind_bind()
        i.tfsimprop.calc_mean_sliding()
        i.tfsimprop.calc_total_and_redundant_sliding(wlc)
        i.tfsimprop.calc_unbinding_binding_distance()
        i.tfsimprop.update_p_bind_list()


        i.tfsimprop.fill_tf_property_list()  # add properties to list (Don't do in init!!!)

        i.tfsimprop.write_property_list() #(dir_to_write=par.OUTPUT_PATH + 'tf_properties/')

    # do system metrics
    system.fill_sys_property_list()  # add properties to list (Don't do in init!!!)
    system.system_metrics()
    #system.vizualize_system(wlc, tfs)

    # reset class attributes (otherwise the next run inherits these)
    TF.run_sim = True
    TF.active_tfs = []
    TF.deleted_tfs = []
    TF.iter_count = 0
    TF.boundaries = None
    TF.time_scale = None
    TF.stop_condition = None

    return system, TF.active_tfs, wlc


def main():
    begin_time = datetime.datetime.now()

    # lists for all runs
    k_off_list, k_on_list, k_a_list, k_d_list, f_dna_list, mean_unbind_bind_list = [[] for _ in range(6)]

    try:
        system, active_tfs, wlc = simulate()
    except Exception as e:

        with open(par.OUTPUT_PATH+'error.csv', 'a', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=',')
            my_writer.writerow([str(e)])
            my_writer.writerow(str(datetime.datetime.now() - begin_time))

        print('Restarting!', e)
        print(traceback.print_tb(e.__traceback__))
        # reset class attr
        TF.run_sim = True
        TF.active_tfs = []
        TF.deleted_tfs = []
        TF.iter_count = 0
        TF.boundaries = None
        TF.time_scale = None

        Target.target_list = []

    else:

        # write results to csv
        df = pd.DataFrame(system.sys_df)

        if not os.path.isfile(par.OUTPUT_PATH+'results.csv'):
            with open(par.OUTPUT_PATH+"results.csv", "w") as f:
                file_writer = csv.writer(f)

                file_writer.writerows([df.columns])

        f = tempfile.NamedTemporaryFile(mode='w+t', delete=False, dir=par.OUTPUT_PATH + 'tf_properties/', prefix='tmp_tfdf_')

        my_str = ''
        for i in list(df.iloc[0]):
            my_str = my_str + f'{str(i)},'

        my_str = my_str[0:-1]

        f.write(my_str)
        f.seek(0)

        f.close()


if __name__ == '__main__':
    main()
