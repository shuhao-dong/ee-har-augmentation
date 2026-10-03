# Datasets

The datasets are not included. Download them and place them under one folder (the data root) with these names:

| Dataset | Folder under the data root | Files read | Source |
|---|---|---|---|
| PAMAP2 | `PAMAP2_Dataset/Protocol/` | `subject101.dat` ... `subject109.dat` | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring |
| USC-HAD | `USC-HAD/` | `Subject<N>/a<A>t<T>.mat` | https://sipi.usc.edu/had/ |
| DSADS | `DSADS/` | `a01`...`a19` / `p1`...`p8` / `s01.txt`...`s60.txt` | https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities |
| TORUS-HAR | `TORUS_HAR/` | `labeled_<id>/labeled_<id>_wrist.csv` | our own dataset; see the paper |

Pass the data root with `--data-root /path/to/datasets` or set `EE_HAR_DATA=/path/to/datasets`.

Windows (from `ee_har/config.py`): TORUS-HAR 2.5 s with 50 % overlap at 100 Hz (3,960 windows); PAMAP2 5 s / 50 % at
100 Hz (7,611 windows; +/-16 g accelerometer and gyroscope of the hand, chest and ankle IMUs, drop-outs interpolated);
USC-HAD 5 s / 50 % at 100 Hz (10,132 windows); DSADS 5 s / 50 % at 25 Hz (18,088 windows).
