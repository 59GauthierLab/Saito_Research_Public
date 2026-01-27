import datetime

# ============================================================
# 設定ブロック
# ============================================================
MODEL_NAME = "unet_w_stft"
USE_HILBERT_FEATURES = True
MODEL_IN_CH = 2 + (2 if USE_HILBERT_FEATURES else 0)
MODEL_BASE = 48
BOTTLENECK_RNN_ENABLED = True
BOTTLENECK_RNN_CHANNELS = 16
BOTTLENECK_RNN_HIDDEN = 256
BOTTLENECK_RNN_LAYERS = 2
BOTTLENECK_RNN_DROPOUT = 0.0

DATASET_INPUT_DIR = "../dataset/input_cropped"
DATASET_TEACHER_DIR = "../dataset/teacher_cropped"
VALIDATION_RATIO = 0.2
SPLIT_SEED = 42

SR_RATE = 44100  # サンプリングレート

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SEG_FRAMES = 256
OVERLAP_RATIO = 0.5

BATCH_SIZE = 3
NUM_EPOCHS = 256
LEARNING_RATE = 1e-4
LOG_WEIGHT = 0.005
MAG_WEIGHT = 0.09
SNR_WEIGHT = 0.03
SIL_WEIGHT = 0.498
SIL_ALPHA = 0.32

OUTPUT_DIR = (
    "output_"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + MODEL_NAME
    + "_"
    + str(LOG_WEIGHT)
    + "-"
    + str(MAG_WEIGHT)
    + "-"
    + str(SNR_WEIGHT)
    + "-"
    + str(SIL_WEIGHT)
    + "-"
    + str(SIL_ALPHA)
)
