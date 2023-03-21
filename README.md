
## Setup

```bash
$ pip3 install -r requirements.txt
$ bazel build //rlap:all
```

## Benchmarks

Use the following shell script to benchmark all the augmentors on node and graph classification datasets

```bash
$ bash run_augmentor_benchmarks.sh
```

Use the following python script to prepare the latex table of benchmark results. The table will be properly filled only when CPU and GPU based benchmarks have completed. Interrupting the previous script to generate the table will lead to parsing errors for incomplete runs.

```bash
$ python3 prepare_augmentor_stats.py
```


## Experiments

Use the following shell script to run node classification experiments using the GRACE design

```bash
$ bash run_node_shared.sh
```

Use the following shell script to run node classification experiments using the MVGRL design

```bash
$ bash run_node_dedicated.sh
```

Use the following shell script to run graph classification experiments using the GraphCL design

```bash
$ bash run_graph_shared.sh
```

Use the following shell script to run graph classification experiments using the BGRL (g-l) design

```bash
$ bash run_graph_shared_g2l.sh
```

Use the following python script to prepare the latex table of results

```bash
$ python3 prepare_final_stats.py
```


## Additional Experiments

Use the following shell script to run max singular value and edge count analysis of rlap variants

```bash
$ python3 rlap_vc_spectral.py
```

Use the following shell script to plot edge counts of randomized schur complements after diffusion

```bash
$ python3 rlap_ppr_edge_plots.py
```
