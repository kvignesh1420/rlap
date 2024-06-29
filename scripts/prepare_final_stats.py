"""
Helper script to prepare latex tables from benchmark data
"""

import os
import pandas as pd
import numpy as np
from pygrok import Grok


def main():
    for task in tasks:
        for framework in frameworks:
            print("TASK: {} FRAMEWORK: {}".format(task, framework))
            directory = "./results/final/{}/{}".format(task, framework)
            filepaths = []
            for dirpath, dirnames, filenames in os.walk(directory):
                # print(dirpath)
                for filename in filenames:
                    filepaths.append(dirpath + "/" + filename)

            table_entries = []
            pattern = "Test run: %{NUMBER:epoch} %{DATA:extra} F1Mi=%{NUMBER:f1mi}, F1Ma=%{NUMBER:f1ma}, Acc=%{NUMBER:acc}"
            grok = Grok(pattern)
            for filepath in filepaths:
                name = filepath.split("/")[-1].replace(".txt", "")
                aug_name = name.split("-")[0]
                mode = name.split("-")[-1]
                dims = int(name.split("-")[-2])
                wd = float(name.split("-")[-3])
                lr = float(name.split("-")[-4])
                num_layers = int(name.split("-")[-5])
                dataset = "-".join(name.split("-")[1:-5])
                with open(filepath, "r") as fh:
                    lines = fh.readlines()
                    f1mi_scores = []
                    f1ma_scores = []
                    acc_scores = []
                    for line in lines:
                        match = grok.match(line)
                        if match is not None:
                            f1mi_scores.append(float(match["f1mi"]) * 100)
                            f1ma_scores.append(float(match["f1ma"]) * 100)
                            acc_scores.append(float(match["acc"]) * 100)

                    table_entries.append(
                        {
                            "augmentor": aug_name,
                            "dataset": dataset,
                            "mode": mode,
                            "num_layers": num_layers,
                            "dims": dims,
                            "lr": lr,
                            "wd": wd,
                            "F1Mi": r"${} \pm {}$".format(
                                np.round(np.mean(f1mi_scores), 2),
                                np.round(np.std(f1mi_scores), 2),
                            ),
                            "F1Ma": r"${} \pm {}$".format(
                                np.round(np.mean(f1ma_scores), 2),
                                np.round(np.std(f1ma_scores), 2),
                            ),
                            "acc": r"${} \pm {}$".format(
                                np.round(np.mean(acc_scores), 2),
                                np.round(np.std(acc_scores), 2),
                            ),
                        }
                    )

            df = pd.DataFrame(table_entries)
            df = df.sort_values(["dataset", "mode", "augmentor"])
            print(df.to_latex(index=False, escape=False))


if __name__ == "__main__":
    frameworks = ["shared", "dedicated"]
    tasks = ["node", "graph"]
    main()
