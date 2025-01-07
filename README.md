# hifigan_tpu

## Purpose
This repository runs the FastPitch + HifiGAN model on TPUs using Torch:XLA. FastPitch is for text-to-speech synthesis, and HifiGAN generates high-fidelity audio from mel-spectrograms. Torch:XLA allows PyTorch to run on TPUs, providing significant performance improvements.

## Getting Started

### Prerequisites
Before you begin, ensure you have met the following requirements:
- You have a Google Cloud account.
- You have access to the TPU dashboard at https://app.felafax.ai/tpu-vms.

### Steps to Run the Repository
Follow these steps to set up and run the repository:

1. **Spin up a TPU**
   - Go to the [TPU dashboard](https://app.felafax.ai/tpu-vms).
   - Spin up a TPU v5e or better.

2. **Clone the Repository**
   - Open a terminal.
   - Run the following command to clone the repository:
     ```bash
     git clone https://github.com/felafax/hifigan_tpu.git
     ```

3. **Run the Jupyter Notebook**
   - Navigate to the cloned repository:
     ```bash
     cd hifigan_tpu
     ```
   - Open the `main.ipynb` file in Jupyter Notebook.
   - Select "Run all cells" to execute the notebook.

## Technical Details

### FastPitch Model
FastPitch is a text-to-speech model that generates mel-spectrograms from text. It is based on the FastSpeech model but includes pitch prediction, which improves the naturalness and expressiveness of the generated speech.

### HifiGAN
HifiGAN is a neural vocoder that converts mel-spectrograms into high-fidelity audio. It is known for its high-quality audio generation and fast inference speed.

### Torch:XLA
Torch:XLA is a library that allows PyTorch to run on TPUs. It provides a seamless integration with PyTorch, enabling significant performance improvements for deep learning models.

## Limitations
One limitation of this repository is the parsing limitation mentioned in the notebook. The current implementation may not handle certain edge cases in text parsing, which could affect the quality of the generated speech. Future work will focus on improving the parsing algorithm to handle a wider range of text inputs.

Additionally, some text parsing and string manipulation in Python is slow. Rewriting in Rust could significantly improve performance.
