# Hardware used for the results

Platform: Google Colab (free tier)

- GPU: Tesla T4, 15360 MiB
- CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, 1 vCPU
- RAM: 12Gi total
- Disk: 113G total (Colab virtual disk, type not reported)
- Operating system: Ubuntu 24.04.4 LTS
- Python packages: see requirements.txt (the run used torch 2.14.1+cu130, Python 3.13)

## Run time

- Start: about 17:02 UTC (22:02 PKT), noted by hand
- End: 17:59:27 UTC (22:59 PKT), the time of the last line written
- Total: about 57 minutes
- Files covered: train-00000, train-00006, train-00013, train-00021, train-00037,
  train-00042 (6 of 50, all complete, 54,022 seconds of video)

The command used was:
python run.py --input /content/finevideo/data --output results.jsonl --sps 5

## Note

The machine had only 1 vCPU, so decoding and the rest of the Python work shared one
CPU core with feeding the GPU. On the grading machine (32 threads) this part should
not be as limiting.
