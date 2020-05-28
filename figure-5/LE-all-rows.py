from __future__ import division

import time
import pickle
import numpy as np
import os
from subprocess import Popen

# time.sleep(5)

MAX_PYTHON_PROCESSES = 8

def main():
    data = pickle.load(open("left-column.pickle","rb"))

    for i in data:
        h2 = i["h2"]
        (M10,M20,Q0,x0,y0) = i["IC"]

        os.system("mkdir results")
        output_file = os.path.join("results","stdout_%5.5f.txt" % h2)
        command = "python LE-any-row.py -h2 %5.5f -M10 %5.5f -M20 %5.5f -Q0 %5.5f -x0 %5.5f -y0 %5.5f" % (h2,M10,M20,Q0,x0,y0)

        test_text = "python LE-any-row.py"
        while True:
            time.sleep(2)
            stream = os.popen("pgrep -af python")
            full_output = stream.read().split("\n")

            ### UNIX
            # num_processes_running = len(list(filter(lambda s: test_text in s, full_output)))
            ### MAC-OSX
            num_processes_running = len(full_output)

            if num_processes_running < MAX_PYTHON_PROCESSES:
                break

        f = open(output_file,"w")
        Popen(command.split(),stdout=f).pid


if __name__ == "__main__":
    main()


