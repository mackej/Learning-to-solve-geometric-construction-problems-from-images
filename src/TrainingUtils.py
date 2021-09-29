import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import numpy as np
import re
import csv
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline


def events_to_data(model_dir, loss='loss'):
    x = []
    y = []
    try:
        for event_file in glob.glob(model_dir + os.path.sep + "*tfevents*"):
            for summary in tf.train.summary_iterator(event_file): #TF1.15
            #for summary in tf.compat.v1.train.summary_iterator(event_file): #TF 2>
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

def yolact_training_stats(model_dir, color=["blue"], loss_legend=["1"],y_range=2, precision=100):
    save_to = model_dir[0]
    if len(model_dir) > 1:
        save_to+= "_X_".join(loss_legend)
    Path(save_to).mkdir(parents=True, exist_ok=True)

    plots = []
    try:

        for i in model_dir:
            data = {}
            for event_file in glob.glob(i + os.path.sep + "*tfevents*"):
                # for summary in tf.train.summary_iterator(event_file):  # TF1.15
                for summary in tf.compat.v1.train.summary_iterator(event_file):  # TF 2>
                    for v in summary.summary.value:
                        if v.tag in ['compute/B', 'compute/M', 'compute/S', 'compute/C', 'meta/lr']:
                            if v.tag not in data:
                                data[v.tag] = []
                            data[v.tag].append(v.simple_value)
            for tag in ['compute/B', 'compute/M', 'compute/S', 'compute/C', 'meta/lr']:
                for i in range(precision - (len(data[tag]) % precision)):
                    data[tag].insert(0, data[tag][0])
                data[tag] = np.average(np.array(data[tag]).reshape(len(data[tag]) // precision, precision), axis=1)
            plots.append(data)
    except:
        print("some error during plotting")
    names ={'compute/B':'Bbox loss', 'compute/M':'Mask loss', 'compute/S':'S', 'compute/C':'Class loss', 'meta/lr':'learning rate'}
    total = [0]*len(model_dir)
    for i in ['compute/B', 'compute/M', 'compute/S', 'compute/C','meta/lr']:
        for c in range(len(model_dir)):
            y = np.array(plots[c][i])
            # x = np.array(range(len(data[i])))
            idx = np.arange(len(y))*precision

            if i != 'meta/lr':
                total[c] += y
            original_loss = plt.figure(1)
            plt.plot(idx, y, color=color[c], label=loss_legend[c])
            # smooth version:
            xnew = np.linspace(idx.min(), idx.max(), 100)
            spl = make_interp_spline(idx, y, k=3)
            y_smooth = spl(xnew)
            smoothed_loss = plt.figure(2)
            plt.plot(xnew, y_smooth, label=loss_legend[c])
        plt.figure(1)
        plt.ylabel('summary: ' + names[i])
        plt.xlabel('steps')
        plt.axhline(y=0, color="black")
        plt.grid()
        plt.gca()
        plt.legend()
        if i != 'meta/lr':
            plt.ylim(0, y_range)
            total[c] += y
        else:
            plt.yscale('log')
        plt.savefig(save_to + os.path.sep + names[i] + '_loss.png')
        plt.close()


        plt.figure(2)
        plt.ylabel('summary: ' + names[i])
        plt.xlabel('steps')
        plt.axhline(y=0, color="black")
        plt.grid()
        plt.gca()
        plt.legend()
        if i != 'meta/lr':
            plt.ylim(0, y_range)
        else:
            plt.yscale('log')
        plt.savefig(save_to + os.path.sep + names[i] + '_smooth_loss.png')
        plt.close()

    for c in range(len(model_dir)):
        idx = np.arange(len(total[c]))*precision
        plt.figure(1)
        plt.plot(idx, total[c], color=color[c], label=loss_legend[c])
        #smoothed
        plt.figure(2)
        xnew = np.linspace(idx.min(), idx.max(), 50)
        spl = make_interp_spline(idx, total[c], k=3)
        y_smooth = spl(xnew)
        smoothed_loss = plt.figure(2)
        plt.plot(xnew, y_smooth, label=loss_legend[c])

    plt.figure(1)
    plt.ylabel('summary: ' + "total")
    plt.xlabel('steps')
    plt.axhline(y=0, color="black")
    plt.grid()
    plt.ylim(0, y_range)
    plt.gca()
    plt.legend()
    plt.savefig(save_to + os.path.sep + 'total' + '_loss.png')
    plt.close()

    plt.figure(2)
    plt.ylabel('summary: ' + "total")
    plt.xlabel('steps')
    plt.axhline(y=0, color="black")
    plt.grid()
    plt.ylim(0, y_range)
    plt.gca()
    plt.legend()
    plt.savefig(save_to + os.path.sep + 'total' + 'smooth_loss.png')
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



