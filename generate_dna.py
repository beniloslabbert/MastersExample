import pickle
from Dna import WLC


# create chain
def create_chain():
    wlc = WLC(10000, conc=0.5, generated_vecs='random')  # conc in M
    wlc.wlc_walk()
    wlc.generate_bp_positions()
    print(wlc.theo_r_squared_wlc(), wlc.real_r_squared())

    wlc.create_targets(amt=1, targ_type='prop_len', size_of_target=10, prop_len=0.5, spec_bp=None)

    return wlc


def pickle_save(dna_chain, dna_obj_name='confined_10k.obj'):
    file_handler_save = open(dna_obj_name, 'wb')
    pickle.dump(dna_chain, file_handler_save)


def pickle_load(dna_obj_name='1k_confined.obj'):
    file_handler_open = open(dna_obj_name, 'rb')
    wlc_open = pickle.load(file_handler_open)

    return wlc_open


def main():
    wlc = create_chain()

    wlc.visualizer()

    pickle_save(wlc)


if __name__ == '__main__':
    main()
