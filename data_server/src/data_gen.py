import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage import io, transform
from collections import Counter
from konlpy.corpus import kolaw
import re
import numpy as np
import string
import random
import os
import copy
import pickle

class Preprocess:
    def __init__(self, image, bbox_list, background_dir):
        self.image = image
        self.bbox_list = bbox_list
        self.background = self.get_background(background_dir)

class DataGenerator:
    def __init__(self, labeling_info, original_directory, background_directory, s3_client,bucket_name,user_id, count=200):
        self.labeling_info = labeling_info
        self.original_directory = original_directory
        self.process_directory = original_directory
        self.gen_directory = os.path.join(self.process_directory, 'GENdata')
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.user_id = user_id
        os.makedirs(self.gen_directory, exist_ok=True)
        self.background_directory = background_directory
        self.count = count

class AddNewText:
    def __init__(self, image, bbox_list, word_list,  label_list):
        self.image = image
        self.korean_text_list = self.extract_korean()
