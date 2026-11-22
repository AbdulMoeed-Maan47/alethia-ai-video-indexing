# My Timings:

The first version of my program took 29m for processing 10 videos of combined time 2782 seconds.
The final version of my program took 3m for processing 10 videos of combined time 2782 seconds.
In the improved version, batch size, quantization and sampling all were implemented at the same time, so all combined to give the improved results.

# My Pipeline:

My pipeline runs the YOLO model, then opens each parquet file one by one, and opens each mp4 file and process each video by decoding frames, keeping uniform sampling and converting each video to BGR. It then collects the frames in the batch and then executes the batch restricted to just 5 objects. It then returns objects and people count in each frame. A json line is written for every second output, even if there is no output.

# Where the time goes:

The major time goes in the YOLO inference, because each batch contains 32 frames at 640x640 resolution, which is very expensive as compared to other I/O operations. I measured it by wrapping each section of the code in time.perf_counter() and then calculated the time.

# Checking correctness:

I ran both versions on the same videos and compared people count and objects arrays directly for every second to catch any discrepancies. I found almost no discrepancies.

# Working with AI:

The AI suggested me to wait for the process to end, but I decided to keep checking the collab, because by not doing so, my collab runtime would time out and I'll had to start over. I found out about this because I had experience working with collab. Moreover, it was my decision to go with the increased batch size, since you can benefit from GPU using parallel processing via batches.

# Next steps:

I would look into how to optimize video decoding, and sampling. I'd try to improve speed by experimenting with batch sizes and precisions, and optimizing the pipeline so that nothing sits idle.
