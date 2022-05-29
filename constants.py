import os
import logging
# INFO, DEBUG, WARNING, CRITICAL
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
            # logging.FileHandler(f"lm_finetune_{datetime.now().isoformat()}.log"),
            logging.StreamHandler()
        ]
)
logger = logging.getLogger(__name__)

TQDM_DISABLE = True if 'TQDM_DISABLE' in os.environ and str(os.environ['TQDM_DISABLE']) == '1' else False
WANDB_DISABLE = True if 'WANDB_DISABLE' in os.environ and str(os.environ['WANDB_DISABLE']) == '1' else False
