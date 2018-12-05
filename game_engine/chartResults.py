import numpy as np
import matplotlib.pyplot as plt

def getValueFromLine(line):
    return line.split(":")[1].strip()

f=open("/home/josh/Desktop/0.995/log.txt", "r")
lines = f.readlines()

discount_rate = 0
tau = 0
learning_rounds = 0
mini_batch_number = 0
ppo_clip = 0
minimum_rollout_size = 0
learning_rate = 0

losses = []
scores = []
steps = []

state = ''

for line in lines:

    if line.startswith('discount'):
        discount_rate = getValueFromLine(line)
    elif line.startswith('tau'):
        tau = getValueFromLine(line)
    elif line.startswith('learning_rounds'):
        learning_rounds = getValueFromLine(line)
    elif line.startswith('mini_batch_number'):
        mini_batch_number = getValueFromLine(line)
    elif line.startswith('ppo_clip'):
        ppo_clip = getValueFromLine(line)
    elif line.startswith('minimum_rollout_size'):
        minimum_rollout_size = getValueFromLine(line)
    elif line.startswith('learning_rate'):
        learning_rate = getValueFromLine(line)
    elif state == "AllScores" and not line.startswith("AllGameSteps"):
        scores.append(float(line))
    elif state == "AllGameSteps" and not line.startswith("AllLosses"):
        steps.append(float(line))
    elif line.startswith("AllScores"):
        state = "AllScores"
    elif line.startswith("AllGameSteps"):
        state = "AllGameSteps"
    elif line.startswith("AllLosses"):
        state = ""
    elif line.startswith('~~~~~~~~~~'):
        #We just finished an iteration. Plot this.
        fig, ax = plt.subplots(1, 4)

        ax[0].plot(np.arange(len(scores)), scores) #row=0, col=0
        ax[0].set_title("Scores")
        
        avgScores = np.convolve(scores, np.ones((50,))/50, mode='valid')
        ax[1].plot(np.arange(len(avgScores)), avgScores) #row=0, col=0
        ax[1].set_title("Running Avg 50 of Score")
        
        ax[2].plot(np.arange(len(steps)), steps) #row=1, col=0
        ax[2].set_title("Steps")

        avgSteps = np.convolve(steps, np.ones((50,))/50, mode='valid')
        ax[3].plot(np.arange(len(avgSteps)), avgSteps) #row=1, col=0
        ax[3].set_title("Running Avg 50 of Steps")

        plt.show()

        steps = []
        scores = []

