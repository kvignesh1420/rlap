"""
Helper script to prepare latex tables from benchmark data
"""

import os
import pandas as pd
import numpy as np

def memory_table():
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
        dataset_name = name.split("-")[1]
        mem_usage = []
        latencies = []
        with open(file, "r") as fh:
            lines = fh.readlines()
            for line in lines:
                if "aug(" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    print(tokens)
                    mem_usage.append(float(tokens[3]))
                if "DURATION" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    print(tokens)
                    latencies.append(float(tokens[1]))

        entry = {
            "augmentor": aug_name,
            "dataset": dataset_name,
            "mean(mem)": np.mean(mem_usage),
            "std(mem)": np.std(mem_usage),
            "mean(duration)": np.mean(latencies),
            "std(duration)": np.std(latencies)
        }
        table_entries.append(entry)

    df = pd.DataFrame(table_entries)
    print(df.to_latex(index=False))

def main():
    memory_table()


if __name__ == "__main__":
    main()