from scheduling import launch
import sys

jobs = {

    "SGD-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt sgd --l2 5e-4 --eta 0.1 --T 60 120 160 --decay-factor 0.2 --no-tqdm""",

    "SGD-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt sgd --l2 5e-4 --eta 0.1 --T 60 120 160 --decay-factor 0.2 --no-tqdm""",

    "SGD-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt sgd --l2 1e-4 --eta 0.1 --T 150 225 --decay-factor 0.1 --no-tqdm""",

    "SGD-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt sgd --l2 1e-4 --eta 0.1 --T 150 225 --decay-factor 0.1 --no-tqdm""",

    "DFW-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt dfw --l2 1e-4 --eta 1. --loss svm --smooth --no-tqdm""",

    "DFW-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt dfw --l2 1e-4 --eta 1. --loss svm --smooth --no-tqdm""",

    "DFW-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt dfw --l2 1e-4 --eta 0.1 --loss svm --smooth --no-tqdm""",

    "DFW-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt dfw --l2 1e-4 --eta 0.1 --loss svm --smooth --no-tqdm""",

    "ADAM-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt adam --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "ADAM-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt adam --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "ADAM-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt adam --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "ADAM-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt adam --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "ADAGRAD-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt adagrad --l2 5e-4 --eta 1e-2 --no-tqdm""",

    "ADAGRAD-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt adagrad --l2 1e-4 --eta 1e-2 --no-tqdm""",

    "ADAGRAD-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt adagrad --l2 1e-4 --eta 1e-2 --no-tqdm""",

    "ADAGRAD-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt adagrad --l2 1e-4 --eta 1e-2 --no-tqdm""",

    "AMSGRAD-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt amsgrad --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "AMSGRAD-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt amsgrad --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "AMSGRAD-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt amsgrad --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "AMSGRAD-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt amsgrad --l2 1e-4 --eta 1e-3 --no-tqdm""",

    "BPGRAD-CIFAR-10-WRN":
    """python3 main.py --batch-size 256 --dataset cifar10 --wrn --opt bpgrad --l2 1e-4 --eta 0.1 --no-tqdm""",

    "BPGRAD-CIFAR-100-WRN":
    """python3 main.py --batch-size 256 --dataset cifar100 --wrn --opt bpgrad --l2 5e-4 --eta 0.1 --no-tqdm""",

    "BPGRAD-CIFAR-10-DN":
    """python3 main.py --batch-size 256 --dataset cifar10 --densenet --opt bpgrad --l2 1e-4 --eta 0.1 --no-tqdm""",

    "BPGRAD-CIFAR-100-DN":
    """python3 main.py --batch-size 256 --dataset cifar100 --densenet --opt bpgrad --l2 1e-4 --eta 0.1 --no-tqdm""",


}


if __name__ == "__main__":
    if len(sys.argv) != 1:
        r = range(1, len(sys.argv))  # pass job codes as params
        toLaunch = [sys.argv[i] for i in r]
    else:
        toLaunch = jobs.keys()
    launch([(jobName, jobs[jobName]) for jobName in toLaunch], interval=3)

