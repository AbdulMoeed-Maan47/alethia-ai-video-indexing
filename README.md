# Project: Alethia AI Take-Home Assignment Video Indexing with YOLO11l

# Author: Abdul Moeed Maan

For each second of each video, `run.py` records which of five objects
(person, car, dog, laptop, cell phone) appear and how many people are visible.

## Install

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

No `apt-get` packages were needed on Colab (PyAV ships its own FFmpeg).

`requirements.txt` pins the CUDA 12.6 build of torch and torchvision, because the
grading machine has a CUDA 12 driver. My results were produced on Colab with
torch 2.14.1+cu130, so the pinned CUDA 12.6 build itself was not tested by me.

## Preparation command (needs internet, run once from the repo folder)

```
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

This downloads `yolo11l.pt` into the current folder. Run `run.py` from the same folder.

## Run

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional: `--sps N` sets how many frames per second of video are sent to the model
(default 5).

## What the script does

Every frame is decoded, but only about 5 frames per second (chosen by timestamp) are
converted to BGR and sent to YOLO11l (imgsz 640, conf 0.4, IoU 0.7, class-aware NMS,
five classes only, FP16, batches of 32). For each second, an object counts as present
if it appears in at least half of the sampled frames in that second, and the people
count is the median (rounded down on a tie). Every second up to the last frame gets a
line, including empty ones. A damaged video gets an error line.

## Results: PARTIAL RUN

`results.jsonl.gz` covers 6 of the 50 files (54,022 seconds of video, about 15 hours).
All six files were processed completely:

- train-00000-of-01357.parquet: 9,144 lines
- train-00006-of-01357.parquet: 9,488 lines
- train-00013-of-01357.parquet: 8,691 lines
- train-00021-of-01357.parquet: 10,254 lines
- train-00037-of-01357.parquet: 8,342 lines
- train-00042-of-01357.parquet: 8,103 lines

There are no error lines. The other 44 files were not run, because of the limited time
and the free Colab T4. Total run time for these six files: about 57 minutes
(see `HARDWARE.md`).

## Checking correctness

I made my own reference by running the baseline (every frame) on 10 videos taken from
5 different files (about 2,782 seconds). `compare.py` scores a results file against it
with the three metrics from the brief:

```
python compare.py ref.jsonl candidate.jsonl
```

The final settings (5 frames per second, FP16, batch 32) scored on that 10-video set:

- object F1: 0.989 (need 0.95)
- people exact match: 96.59% (need 85%)
- people within 1: 99.78% (need 95%)

The reference is small, and cars and cell phones had few examples in it (about 55 and
70 seconds), so their individual scores are less reliable.

## Known limits

- Rotation is only read from the old `rotate` metadata tag. Videos that store rotation
  as a display matrix would not be rotated. I checked one video and it had no rotation.
- `half=True` prints a deprecation warning in the Ultralytics version I used.
- Decoding is single-process, so decode time is likely a bottleneck. I did not measure
  decode and detect time separately.
