#experiment_config.py
from pathlib import Path
from datetime import datetime

class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("./")
       
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("data/luna25/luna25_nodule_blocks")
        self.MASKDIR = Path("data/luna25/luna25_mask_blocks")
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("Luna25/data/luna25")
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "logs"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)

        self.EXPERIMENT_NAME = "I3D_UNet"
        self.MODE = "3D" # 2D or 3D

        self.AUX_LOSS_WEIGHT = 0.5
        self.CKPT_PATH = None
        self.WANDB_ID = None

        if self.CKPT_PATH is None:
            self.RUN_NAME = f"{self.EXPERIMENT_NAME}-{self.MODE}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.RUN_NAME = self.CKPT_PATH.split("/")[-2]

        # Training parameters
        self.DEVICE = "cuda:0"
        self.SEED = 2025
        self.NUM_WORKERS = 6
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 16

        self.BALANCED = True
        self.ROTATION = ((-45, 45), (-45, 45), (-45, 45))
        self.TRANSLATION = True
        # self.FLIP_PROBS = [0.3, 0.3, 0.3]
        self.FLIP_PROBS = [0, 0, 0]
        self.INTERPOLATE_SCALE = 2
        self.USE_MONAI_TRANSFORMS = True

        self.THRESHOLD = 0.5
        self.EPOCHS = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5

        self.HARD_MINING = False   # full-batch training again
        self.HARD_MINING_RATIO = 0.6

config = Configuration()
