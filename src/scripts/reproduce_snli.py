import os
import sys

from scheduling import launch


jobs = {
    "SGD-CE":
    "python train_nli.py --opt sgd --eta 1 --loss ce --no-tqdm",

    "SGD-SVM":
    "python train_nli.py --opt sgd --eta 0.1 --loss svm --no-tqdm",

    "ADAM-SVM":
    "python train_nli.py --opt adam --eta 1e-4 --loss svm --no-tqdm",

    "ADAM-CE":
    "python train_nli.py --opt adam --eta 1e-4 --loss ce --no-tqdm",

    "DFW-SVM":
    "python train_nli.py --opt dfw --eta 1 --loss svm --no-tqdm",
}


if __name__ == "__main__":
    # change current directory to InferSent
    os.chdir('./InferSent/')
    if len(sys.argv) != 1:
        r = range(1, len(sys.argv))  # pass job codes as params
        toLaunch = [sys.argv[i] for i in r]
    else:
        toLaunch = jobs.keys()
    launch([(jobName, jobs[jobName]) for jobName in toLaunch], interval=3)
    # change current directory back to original
    os.chdir('..')
