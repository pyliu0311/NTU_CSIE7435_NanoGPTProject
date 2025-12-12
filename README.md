# NTU_CSIE7435_NanoGPTProject

## Project Description

In this project, you will explore the implementation of GPT-2 through the lightweight framework NanoGPT, with the goal of connecting mathematical definitions to practical code and improving the implementation.

Your first task is to connect the mathematical definition of multi-head attention (see Section Transformer blocks and other details in the slides) with its implementation in NanoGPT (see model.py in the NanoGPT).
Locating the computation of Query, Key, and Value: Find where queries, keys, and values are generated in the NanoGPT code, and compare this process with how they are introduced in the slides. What differences do you notice? What efficiency or memory benefits do these differences bring?
Head Splitting and Tensor Shapes: Identify where the code separates the attention heads and reshapes tensors. How does the reshaping in code relate to the mathematical definition in the slides? What reasons might explain the reshaping?

Your second task is to implement and evaluate KV Caching to NanoGPT with a focus on inference efficiency.
KV Caching: We introduce KV caching in the slides (see Section Prediction in the slides). Implement KV caching for the autoregressive prediction in NanoGPT (see sample.py in the NanoGPT) and compare inference time with the original NanoGPT. For reference, you can check the implementation of KV caching in the official GPT-2 repository.

## This Code
Modify NanoGPT's `model.py` to implement a key-value cache for inference and analyze the time consumption of each computational component. Also implement `experiment.py` for experimental purposes.

## How to run

1. Clone the official nanopt repository and import my two extra files (the `model.py` file should be overwritten). (NanoGPT: https://github.com/karpathy/nanoGPT)

2. Find a dataset that interests you, organize it according to the instructions in the official NanoGPT GitHub repository, and train your model.

3. Modify the experiment configuration in `experiment.py` to compare the inference speed with and without caching.

4. When running `sample.py` normally, you can also set `use_cache=True` to use the cache to speed up inference.
