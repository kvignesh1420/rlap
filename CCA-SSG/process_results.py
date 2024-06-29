"""Prepare the results table from runs"""

import os
import pandas as pd
import numpy as np
from pygrok import Grok
from collections import defaultdict


def main():
    table_entries = []
    for dataset in ["CORA", "CITESEER", "COMPUTERS"]:
        print("DATASET: ", dataset)
        directory = "./results/{}".format(dataset)
        filepaths = []
        for dirpath, dirnames, filenames in os.walk(directory):
            # print(dirpath)
            for filename in filenames:
                filepaths.append(dirpath + "/" + filename)
        aug_results = defaultdict(list)
        pattern = "Linear evaluation accuracy:%{NUMBER:acc}"
        grok = Grok(pattern)
        for filepath in filepaths:
            name = filepath.split("/")[-1].replace(".txt", "")
            aug_name = name.split("-")[0]
            with open(filepath, "r") as fh:
                lines = fh.readlines()
                acc_scores = []
                for line in lines:
                    match = grok.match(line)
                    if match is not None:
                        aug_results[aug_name].append(float(match["acc"]) * 100)
        print(aug_results)
        for aug_name in list(aug_results.keys()):
            table_entries.append(
                {
                    "augmentor": aug_name,
                    "dataset": dataset,
                    "acc": r"${} \pm {}$".format(
                        np.round(np.mean(aug_results[aug_name]), 2),
                        np.round(np.std(aug_results[aug_name]), 2),
                    ),
                }
            )

    df = pd.DataFrame(table_entries)
    df = df.sort_values(["dataset", "augmentor"])
    #     print(df.to_latex(index=False, escape=False))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
