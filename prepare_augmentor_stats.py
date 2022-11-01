"""
Helper script to prepare latex tables from benchmark data
"""

import os
import pandas as pd
import numpy as np

def main():
    dir = "./results"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        print(dirpath)
        for filename in filenames:
            files.append(dirpath + "/" + filename)

    table_entries = []
    for file in files:
        name = file.split("/")[-1].split(".")[0]
        aug_name = name.split("-")[0]
        dataset_name = "-".join(name.split("-")[1:])
        mem_usage = []
        latencies = []
        with open(file, "r") as fh:
            lines = fh.readlines()
            for line in lines:
                if "aug(" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    mem_usage.append(float(tokens[3]))
                if "DURATION" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    latencies.append(float(tokens[1]))

        entry = {
            "augmentor": aug_name,
            "dataset": dataset_name,
            "memory": r"${} \pm {}$".format(np.round(np.mean(mem_usage),4), np.round(np.std(mem_usage),4)),
            "latency": r"${} \pm {}$".format(np.round(np.mean(latencies),4), np.round(np.std(latencies),4)),
        }
        table_entries.append(entry)

    df = pd.DataFrame(table_entries)
    df = df.sort_values(["dataset"])
    print(df.to_latex(index=False, escape=False))

if __name__ == "__main__":
    main()