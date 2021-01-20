import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import numpy as np
import re
import csv


def events_to_data(model_dir, loss='loss'):
    x = []
    y = []
    try:
        for event_file in glob.glob(model_dir + os.path.sep + "*tfevents*"):
            for summary in tf.train.summary_iterator(event_file):
                for v in summary.summary.value:
                    if v.tag == loss:
                        x.append(summary.step)
                        y.append(v.simple_value)
    except:
        print("some error during plotting")
    x = np.array(x)
    y = np.array(y)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return x,y

def events_to_pyplot(model_dir, file="loss.png", loss='loss'):

    x, y = events_to_data(model_dir, loss=loss)
    plt.plot(x, y)
    plt.ylabel('summary: '+loss)
    plt.xlabel('steps')
    plt.axhline(y=0, color="black")
    plt.grid()
    plt.yticks(np.arange(0, 0.8, 0.05))
    plt.savefig(model_dir + os.path.sep + file)
    plt.close()

def plot_multiple_losses(log_dir, loss='loss', regex=".*"):
    like = re.compile(regex, re.IGNORECASE)
    for i in os.listdir(log_dir):
        if not like.search(i):
            continue
        print(i)
        x, y = events_to_data(os.path.join(log_dir, i), loss=loss)
        plt.plot(x, y, label=i.replace('_', ' '))
    plt.ylabel(loss)
    plt.xlabel('epochs')
    plt.legend(fontsize='x-large')
    plt.axhline(y=0, color="black")
    plt.grid()
    plt.yticks(np.arange(0, 0.8, 0.05))
    plt.title(loss)
    plt.savefig(loss+".png")
    plt.close()

def clump_csv_resuts(folder, pattern='*'):
    res = {}
    max_keys = 0
    for i in os.listdir(folder):
        for j in os.listdir(os.path.join(folder, i)):
            path = os.path.join(folder, i, j, "*.csv")
            list_of_files = glob.glob(path)
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, newline='') as csvfile:
                rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
                label = next(rows)[0]
                if label not in res:
                    res[label] = []
                for row in rows:
                    res[label].append(row[0])
                    max_keys = max(max_keys, len(res[label]))
    csv_output = open("csv_result.csv", "w")
    keys = sorted(res.keys())
    print(", ".join([l for l in keys]), file=csv_output)
    for i in range(max_keys):
        print(", ".join([str(res[k][i]) if i < len(res[k]) else "" for k in keys]), file=csv_output)



