Run:

```sh
conda env create -f environment.yml --prefix ./env
conda activate ./env
pip install mpipool
mpirun -n 4 --oversubscribe python de.py
```
