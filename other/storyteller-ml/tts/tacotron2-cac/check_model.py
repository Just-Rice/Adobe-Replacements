import os
import json
# import re
import argparse
# from string import punctuation

import torch
import numpy as np
from torch.utils.data import DataLoader
# from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device #, read_lexicon
from dataset import TextDataset
from text import text_to_sequence, sequence_to_text
from text.cleaners import to_arpa
import sys
device = "cpu"
from model import Tacotron2

def get_tac2(configs, device):
    (preprocess_config, model_config, train_config) = configs

    model = Tacotron2(preprocess_config, model_config, train_config).to(device)

    model.eval()
    model.requires_grad_ = False
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
      print(f"Model path {args.checkpoint} incorrect, file not found.")
      sys.exit(2)



    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    stats = None
    mel_stats = None

    # Get model
    model = get_tac2(configs, device)
    try:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model"])
    except:
        print("Error loading model")
        sys.exit(1) # Error
    
    print("Model is valid.")
    sys.exit(0) # Success
    
 