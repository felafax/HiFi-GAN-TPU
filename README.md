# HiFiGAN + FastPitch on TPU

Run NVIDIA's FastPitch and HiFiGAN text-to-speech models on Google Cloud TPUs using PyTorch/XLA, achieving real-time factors (RTF) of up to 160x.

## Performance

| TPU Version | End-to-End Latency | RTF (approx.) |
|-------------|-------------------|---------------|
| v3          | 230ms             | 125x          |
| v5e         | 50-60ms           | 140x          |
| v5p         | 40-50ms           | 160x          |

*Benchmarks performed on ~7-8 second audio generations including padding

## Overview

This repository demonstrates how to run NVIDIA's FastPitch (text-to-spectrogram) and HiFiGAN (spectrogram-to-audio) models on Google Cloud TPUs. The implementation:

- Uses PyTorch/XLA for TPU acceleration
- Includes TPU-optimized inference code
- Achieves real-time performance with minimal latency
- Supports dynamic input lengths
- Provides easy-to-use Jupyter notebook interface

## Quick Start

1. **Set up TPU VM**
   - Go to [Felafax AI Dashboard](https://app.felafax.ai/tpu-vms)
   - Spin up a TPU v5e or better instance
   - Connect to the instance

2. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/hifigan_tpu.git
   cd hifigan_tpu
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Inference**
   - Open `main.ipynb` in Jupyter
   - Run all cells
   - Use the provided inference functions to generate audio

## Implementation Details

The are several optimizations for TPU inference:
- Static shape compilation for XLA
- BFloat16 precision where appropriate
- Efficient text parsing
- Minimized host-device transfers
- Compilation cache management

The end-to-end pipeline includes:
1. Text parsing and tokenization
2. FastPitch inference (text → mel-spectrogram)
3. HiFiGAN inference (mel-spectrogram → audio)

## Notes

- First inference may be slower due to compilation
- Compilation cache is maintained in `./compilation_cache/`
- Text parsing currently runs on CPU and is often the bottleneck (~50-60% of total time)
- Further optimizations possible by moving text parsing to Rust/C++

