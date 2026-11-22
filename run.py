"""Run YOLO11l on about SPS sampled frames per second, aggregate per second, write JSONL.

Usage: python run.py --input <folder of .parquet> --output results.jsonl
"""
import argparse
import glob
import io
import json
import math
import os

import av
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

# COCO class ids -> output names
CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CLASS_IDS = list(CLASSES.keys())
ORDER = ["person", "car", "dog", "laptop", "cell phone"]
BATCH = 32
SPS = 5  # frames sampled per second of video; set with --sps


def get_rotation(stream):
    """Clockwise degrees needed to make the frame upright (0/90/180/270).

    Only reads the legacy 'rotate' metadata tag. Newer files store a
    display matrix instead: check a few videos with ffprobe and extend
    this function if you find any (see notes).
    """
    try:
        r = int(float(stream.metadata.get("rotate", 0)))
    except (TypeError, ValueError):
        r = 0
    return r % 360


def rotate(img, deg):
    if deg == 90:
        return np.rot90(img, k=-1)  # clockwise
    if deg == 180:
        return np.rot90(img, k=2)
    if deg == 270:
        return np.rot90(img, k=1)
    return img


def frames_of(mp4_bytes):
    """Yield (timestamp_seconds, BGR ndarray) for each decoded frame."""
    with av.open(io.BytesIO(mp4_bytes)) as container:
        stream = container.streams.video[0]
        deg = get_rotation(stream)
        step = 1.0 / SPS
        next_t = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            # Sampling: keep a frame only when its time reaches the next
            # sample time. The first frame is always kept. If the video has
            # fewer than SPS frames per second, every frame is kept.
            if next_t is not None and t < next_t - 1e-6:
                continue
            next_t = t + step if next_t is None else next_t + step
            while next_t <= t:  # skip ahead after timestamp gaps
                next_t += step
            img = frame.to_ndarray(format="bgr24")
            yield t, np.ascontiguousarray(rotate(img, deg))


def detect(model, imgs):
    """Return a list of (people_count, set_of_object_names) per image."""
    results = model.predict(
        imgs, imgsz=640, conf=0.4, iou=0.7, classes=CLASS_IDS,
        agnostic_nms=False, half=True, verbose=False,
    )
    out = []
    for r in results:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = {CLASSES[c] for c in cls}
        out.append((int((cls == 0).sum()), names))
    return out


def median_floor(values):
    """Median, rounded down when it falls halfway between two numbers."""
    return int(math.floor(float(np.median(values))))


def process_video(model, mp4_bytes, video_id, fout):
    # per second: [n_frames, [people per frame], {object: frames_with_it}]
    secs = {}
    err = None
    try:
        first_t = None
        buf_t, buf_img = [], []

        def flush():
            for t, (people, names) in zip(buf_t, detect(model, buf_img)):
                s = int(math.floor(t - first_t))
                if s < 0:
                    s = 0
                rec = secs.setdefault(s, [0, [], {}])
                rec[0] += 1
                rec[1].append(people)
                for n in names:
                    rec[2][n] = rec[2].get(n, 0) + 1
            buf_t.clear()
            buf_img.clear()

        for t, img in frames_of(mp4_bytes):
            if first_t is None:
                first_t = t
            buf_t.append(t)
            buf_img.append(img)
            if len(buf_img) == BATCH:
                flush()
        if buf_img:
            flush()
        if first_t is None:
            err = "no video frames"
    except Exception as e:  # damaged video: keep what we have, add error line
        err = f"{type(e).__name__}: {e}"[:200]

    if secs:
        for s in range(0, max(secs) + 1):  # every second, even empty ones
            n, people, objs = secs.get(s, [0, [], {}])
            if n == 0:
                line = {"video": video_id, "second": s, "objects": [], "people": 0}
            else:
                present = [o for o in ORDER if objs.get(o, 0) * 2 >= n]
                line = {"video": video_id, "second": s, "objects": present,
                        "people": median_floor(people)}
            fout.write(json.dumps(line) + "\n")
    if err:
        fout.write(json.dumps({"video": video_id, "error": err}) + "\n")
    fout.flush()


def main():
    global SPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sps", type=int, default=SPS,
                    help="frames sampled per second of video")
    args = ap.parse_args()
    SPS = args.sps

    model = YOLO("yolo11l.pt")
    files = sorted(glob.glob(os.path.join(args.input, "**", "*.parquet"), recursive=True))

    with open(args.output, "w") as fout:
        for path in files:
            fname = os.path.basename(path)
            row = 0
            try:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    video_id = f"{fname}:{row}"
                    try:
                        data = batch.column("mp4")[0].as_py()
                        process_video(model, data, video_id, fout)
                    except Exception as e:
                        fout.write(json.dumps({"video": video_id,
                                               "error": f"{type(e).__name__}: {e}"[:200]}) + "\n")
                    row += 1
            except Exception as e:  # unreadable parquet file
                fout.write(json.dumps({"video": f"{fname}:{row}",
                                       "error": f"parquet read failed: {e}"[:200]}) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()