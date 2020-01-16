"""Script to benchmark the performance of a model on Atari Cartpole
Author: Harsh Bhate
******************************************************************************
Note: The MATPLOTLIB library in this class uses LaTeX. The output would not
work if you do not have a pre-installed LaTeX compiler on your device. To use
this library without LaTeX, remove the prefix r from the statement.
For Example:
plt.ylabel(r'$L(x) = max(0,c-x)$') would plot the text in LaTeX, whereas
plt.ylabel('L(x) = max(0,c-x)') would plot the text as console output.

Also, comment out line 25.

RECOMMENDATION:
---------------
The Use of LaTeX is strongly recommended for uniform replication of plot style
irrespective of system configuration.
******************************************************************************
"""

import csv
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import pandas as pd

# Configuring plot style
rc('text', usetex=True)
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 
font = {'family' : 'serif',
        'size'   : 12}
rc('font', **font)

class benchmark:
    """Class to benchmark the performance of models. The class features:

    """

    # Defining the paths
    CSV_PATH = "deep_q_learning/scores/scoreLog.csv"
    PNG_PATH = "deep_q_learning/scores/scores.png"
    BENCHMARK_SCORE = 195

    def __init__(self):
        """Constructor"""
        if os.path.exists(self.PNG_PATH):
            os.remove(self.PNG_PATH)
        if os.path.exists(self.CSV_PATH):
            os.remove(self.CSV_PATH)
    
    def record_score(self, episode, step):
        """Function to record the step into CSV file. In the cartpole problem,
        X-axis: Episodes
        Y-Axis: Step (Or how long can the cartpole stay up)"""
        log = [episode, step]
        title = '# Episodes, # Steps'
        comment = '# Log of model performance on CartPole-v1'
        # add row to CSV file
        with open(self.CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log)
    
    def display_log(self):
        """Function to display the CSV File"""
        log = pd.read_csv(self.CSV_PATH, 
                        header=None, 
                        names = ["Episode", "Step"])
        print(log)
    
    def stats(self):
        """Function to display the log info"""
        log = np.genfromtxt(self.CSV_PATH, 
                            delimiter=',')
        episodes = log[:, 0]
        steps = log[:, 1]
        mean_step = np.mean(steps)
        std_step = np.std(steps)
        # Printing the Report
        msg = "\n\t\t STAT REPORT\n"
        print(msg)
        msg = "Number of Episodes: {}".format(np.shape(episodes)[0])
        print(msg)
        msg = "Mean Steps: %.3f, Standard Deviation: %.3f"\
            %(mean_step, std_step)
        print(msg)
        msg = "Benchmark Mean: {}".format(self.BENCHMARK_SCORE)
        print(msg)

    def plot_log(self):
        """Function to plot the graph"""
        # Read CSV
        log = np.genfromtxt(self.CSV_PATH, 
                            delimiter=',')
        episodes = log[:, 0]
        steps = log[:, 1]
        # Finding Mean Line
        mean_step = np.mean(steps)
        mean_observed = np.full(np.shape(steps), mean_step)
        # Benchmark Mean Line
        mean_benchmark = np.full(np.shape(steps), self.BENCHMARK_SCORE)
        # Estimating Trend Line
        z = np.polyfit(episodes, steps, 1)
        p = np.poly1d(z)
        # PLotting
        plt.plot(episodes, 
                steps, 
                'k', 
                label="")
        plt.plot(episodes, 
                p(episodes), 
                'g--', 
                label="Trend")
        plt.plot(episodes, 
                mean_observed, 
                'b', 
                label="Observed Mean")
        plt.plot(episodes, 
                mean_benchmark, 
                'r', 
                label="Benchmark Mean")
        # Adding Legend, Title 
        plt.xlabel(r'Episodes')
        plt.ylabel(r'Steps')
        plt.title(r"Plot of Steps vs Episode")
        # Place a legend to the right of this smaller subplot.
        plt.legend(loc=1)
        plt.savefig(self.PNG_PATH, bbox_inches='tight')
        plt.show()