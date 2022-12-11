### WLC
BP = 1000
BOUNDARIES=None
BOUNDARY_MODE='target_concentration'
CONC=0.0167
S_LINK=0.2
L_P=50
GENERATED_VECS = 'random'  # or None

# targets
TARG_AMT = 1
TARG_TYPE='prop_len'
SIZE_OF_TARGET=10
PROP_LEN=0.5
SPEC_BP=None

### TFs
DNA_OBJECT = 'dna_object.obj'
CURR_POS= 'box_edge'  # or None
BIND_DIST=3
P_BIND=0.5
TARG_TIME_1D=0.1  # milisec # for 1st round = 1
ONE_D_DIFF_CONSTANT=0.05
THREE_D_DIFF_CONSTANT=3
TIME_SCALE='microsec'
CONTACT_SIZE_BP=10

# targ_time for LacI has been shown to be 5 microsecs
# bind_dist (see bind_dist in Classes). Marklund 2013 estimates LacI radius at 2.8 nm.
# it is not uncommon for 3D/1D diffusion constants to be 100 (3D 2 order larger). Tabaka 2014, Marcovitz 2012.
# p_bind is related to binding site proportion of TF solvent accessible surface area 0.2-0.5. Mirny 2009.

# sim
MODE = 'boundary_interaction'
ONE_D_MODE = 'edge_fall'
STOP_CONDITION = 'iter_target'  # 'bp_target' or 'iter_target'
TARGET_ITER = 100000 # default 100000

# outpath
OUTPUT_PATH = './output/'

