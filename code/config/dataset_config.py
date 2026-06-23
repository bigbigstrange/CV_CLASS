"""Dataset-specific constants and recommended training settings."""

# All sizes are (width, height).
# KSDD and KSDD2 share a similar portrait aspect ratio (~0.36 w/h).

# KSDD (JIM 2019): resize used in SegDec-Net and follow-up papers
KSDD_INPUT_SIZE = (512, 1408)

# KSDD2 (COMIND 2021): SuperSimpleNet uses 232 x 640 (w x h)
KSDD2_INPUT_SIZE = (232, 640)
SSN_INPUT_SIZE = KSDD2_INPUT_SIZE

# Cross-dataset training (scheme A): one size for both KSDD and KSDD2
UNIFIED_INPUT_SIZE = (512, 1408)

# Grayscale normalization on [0, 1] pixel values
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MASK_THRESHOLD = 0

# KSDD official SegDec-Net uses dilated masks (dilate=5)
KSDD_MASK_DILATE = 5
# SuperSimpleNet supervised KSDD2 default
KSDD2_MASK_DILATE = 7

# Phase-1 system build: small KSDD with official 3-fold CV
KSDD_DEFAULT_FOLD = 0

# Phase-2 model optimization: full supervised masks on KSDD2 train split
KSDD2_DEFAULT_WEAK_SPLIT = "split_weakly_246"