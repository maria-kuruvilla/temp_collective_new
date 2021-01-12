"""
Goal - To fo functional pca on convex hull area values
"""
import pickle
import numpy as np


in_dir1 = '../../output/temp_collective/convex_hull_area.p'

area = pickle.load(open(in_dir1, 'rb')) # 'rb is for read binary

