try:
    import waitGPU
except ImportError:
    print('Failed to import waitGPU --> no automatic scheduling on GPU')
    waitGPU = None
    pass
import subprocess
import time


def run_command(command, noprint=True):
    if waitGPU is not None:
        waitGPU.wait(nproc=2, interval=1, ngpu=1)  # process limit adjusted for better GPU from GCP
    command = " ".join(command.split())
    if noprint:
        command = "{} > /dev/null".format(command)
    print(command)
    subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=None, shell=True)


def launch(jobs, interval):
    i = 0
    for name, job in jobs:
        i += 1
        print("\nJob {}, {} out of {}".format(name, i, len(jobs)))
        run_command(job, noprint=False)
        time.sleep(interval)
