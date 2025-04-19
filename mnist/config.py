import sys, os, dnnlib, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Adjust Paths
INIT_PKL = '/Users/gigimerabishvili/Desktop/Trunc/Checkpoints/Checkpoints/stylegan2_mnist_32x32-con.pkl'

MODEL = '/Users/gigimerabishvili/Desktop/Wina_frontier-generation-latent-space-interpolation/mnist/models/cnnClassifier_lowLR.h5'
num_classes = 10

FRONTIER_PAIRS = 'mnist/eval'          


STYLEGAN_INIT = {
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds"    : [[0, 1]],
        "w_load"      : None,
        "class_idx"   : None,
        "mixclass_idx": None,
        "stylemix_idx": [],
        "patch_idxs"  : None,
        "stylemix_seed": None,
        "trunc_psi"   : None,
        "trunc_cutoff": None,
        "random_seed" : 0,
        "noise_mode"  : 'random',
        "force_fp32"  : False,
        "layer_name"  : None,
        "sel_channels": 1,           
        "base_channel": 0,
        "img_scale_db": 0,
        "img_normalize": False,      
        "to_pil"      : True,
        "input_transform": None,
        "untransform" : False,
    },
    "device": DEVICE,
    "renderer": None,
    "pretrained_weight": INIT_PKL,
}


SEARCH_LIMIT         = 1000
STYLEMIX_SEED_LIMIT  = 100
SSIM_THRESHOLD       = 0.95
L2_RANGE             = 0.20
POPSIZE              = 100
STOP_CONDITION       = "iter"
NGEN                 = 10
RUNTIME              = 3600
STEPSIZE             = 10
MUTLOWERBOUND        = 0.01
MUTUPPERBOUND        = 0.6
RESEEDUPPERBOUND     = 10
K_SD                 = 0.1
K                    = 1
ARCHIVE_THRESHOLD    = 4.0
MUTOPPROB            = 0.5
MUTOFPROB            = 0.5
IMG_SIZE             = 28
INITIALPOP           = 'random'
GENERATE_ONE_ONLY    = True
RESULTS_PATH         = 'results'
REPORT_NAME          = 'stats.csv'
DATASET              = 'mnist/original_dataset/janus_dataset_comparison.h5'
EXPLABEL             = 5
INTERPRETER          = '/home/vin/yes/envs/tf_gpu/bin/python'
