"""
Helper script to prepare latex tables from benchmark data
"""

import os
import pandas as pd
import numpy as np

def main_cpu():
    directory = "./results/overheads/cpu"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        print(dirpath)
        for filename in filenames:
            files.append(dirpath + "/" + filename)

    table_entries = []
    for _file in files:
        name = _file.split("/")[-1].split(".")[0]
        aug_name = name.split("-")[0]
        dataset_name = "-".join(name.split("-")[1:])
        mem_usage = []
        latencies = []
        with open(_file, "r") as fh:
            lines = fh.readlines()
            for line in lines:
                if "aug(" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    # print(aug_name, dataset_name, tokens)
                    mem_usage.append(float(tokens[3]))
                if "DURATION" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    latencies.append(float(tokens[1]))

        entry = {
            "augmentor": aug_name,
            "dataset": dataset_name,
            "memory": r"${} \pm {}$".format(np.round(np.mean(mem_usage),4), np.round(np.std(mem_usage),4)),
            "cpu_latency": r"${} \pm {}$".format(np.round(np.mean(latencies),4), np.round(np.std(latencies),4)),
        }
        table_entries.append(entry)

    df = pd.DataFrame(table_entries)
    df = df.sort_values(["dataset", "augmentor"])
    return df


def main_gpu():
    directory = "./results/overheads/gpu"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        print(dirpath)
        for filename in filenames:
            files.append(dirpath + "/" + filename)

    table_entries = []
    for _file in files:
        name = _file.split("/")[-1].split(".")[0]
        aug_name = name.split("-")[0]
        dataset_name = "-".join(name.split("-")[1:])
        latencies = []
        with open(_file, "r") as fh:
            lines = fh.readlines()
            for line in lines:
                if "DURATION" in line:
                    tokens = line.split(" ")
                    tokens = [tok for tok in tokens if tok!=""]
                    latencies.append(float(tokens[1]))

        entry = {
            "augmentor": aug_name,
            "dataset": dataset_name,
            "gpu_latency": r"${} \pm {}$".format(np.round(np.mean(latencies),4), np.round(np.std(latencies),4)),
        }
        table_entries.append(entry)

    df = pd.DataFrame(table_entries)
    df = df.sort_values(["dataset", "augmentor"])
    return df

if __name__ == "__main__":
    df_cpu = main_cpu()
    df_gpu = main_gpu()
    df = df_cpu.merge(df_gpu, on=["augmentor", "dataset"])
    print(df.to_latex(index=False, escape=False))