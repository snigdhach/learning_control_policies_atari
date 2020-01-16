from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np



class ScoreLogger:
    """Log scores
    """
    SCORES_CSV_PATH = "scores.csv"
    SCORES_PNG_PATH = "scores.png"
    SOLVED_CSV_PATH = "solved.csv"
    SOLVED_PNG_PATH = "solved.png"
    AVERAGE_SCORE_TO_SOLVE = 195
    CONSECUTIVE_RUNS_TO_SOLVE = 100

    def __init__(self, env_name):
        """Initializing Score
        """
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)
