# LLM Factoscope

We are excited to share the code and datasets from our study on the LLM Factoscope, making them publicly available for further research and development. Our repository, containing all necessary materials, is accessible at [this link](https://github.com/JenniferHo97/llm_factoscope).

## Models Used
Our experiments leverage several Large Language Models (LLMs) from Huggingface, a reputable platform hosting a diverse array of LLMs. The specific models utilized in our study include:
- **GPT2-XL**: [View on Huggingface](https://huggingface.co/openai-community/gpt2-xl)
- **Llama-2-7B**: [View on Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Vicuna-7B-v1.5**: [View on Huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- **Llama-2-13B**: [View on Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- **Vicuna-13B-v1.5**: [View on Huggingface](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- **StableLM-7B**: [View on Huggingface](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)

## Computational Environment
For our computational experiments, we used the following major packages and versions:
- **Python**: `3.9.18`
- **Captum**: `0.6.0`
- **Ecco**: `0.1.2`
- **H5py**: `3.6.0`
- **Huggingface**: `0.17.3`
- **PyTorch**: `1.10.2`

## Dataset and Codebase
Inside the GitHub repository, the dataset is organized in the `dataset_train` directory in a clear and accessible manner. The dataset comprises various files, catering to different aspects of our experiments.

### Code Files
- **Construct dataset (`construct_data.py`)**: Uses the prompts and answers in the factual datasets to query the LLMs for collecting inner states and labels.
- **Prepare (`prepare_save_data_h5.py`)**: Preprocesses and transforms the inner states for effective learning.
- **Train (`train_nn_tri_test.py`)**: Trains the LLM Factoscope.

### Replicating Our Experiments
To replicate our experiments, follow these steps:

1. **Fill the Target Path**: Specify the directory where you wish the datasets and models to be stored.
2. **Run Construct**: Execute `python construct_data.py` in the terminal.
3. **Execute Prepare**: Run `python prepare_save_data_h5.py` in the terminal.
4. **Train the Model**: Execute `python train_nn_tri_test.py --model_name gpt2-xl --support_size 100` in the terminal.

Following these steps will guide you through the process of constructing the dataset, preparing the data, and training the LLM Factoscope on your dataset.
