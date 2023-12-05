# cosmosage

## Introduction

Large language models are emerging as powerful tools for many natural language tasks. Very large parameter counts of 1e11 or more are needed for a general-purpose model such as GPT-4. However, even small models can be extremely powerful if the application is sufficiently narrow.

cosmosage is an attempt to fine-tune a relatively modest large language model on cosmology-specific datasets with the goal of making a general-purpose natural-language assistant for cosmologists.

## Author

Tijmen de Haan <tijmen.dehaan@gmail.com>

## Project Structure

The project includes the following files, as well as some more helper files.

1. `cosmosage.ipynb`: This notebook walks through a several-step process for fine-tuning the language model on cosmology-specific datasets. It goes through steps for data collection, preprocessing, model training, and evaluation.

2. `extract_textbooks.ipynb`: This notebook is an example of how to use `extract_textbooks.py` to extract next-token prediction training data for fine-tuning a base model. It extracts text from astro-related textbooks. This data is then cleaned and saved in JSON format for further use. 

3. `arxiv.py`: Helper file adapted from Karpathy's excellent `arxiv-sanity-lite`, this is used to interface with the arXiv API.

4. `fine_tune.py`: This contains the actual `pytorch` training loop.

## Syntax, Code Style, Tools Used

The .py files are kept consistently formatted with `black` on its default settings.

The codebase was written with the use of Pylance, GitHub Copilot, GPT-4, and VSCode fork `cursor`.

## Usage

To get started with cosmosage:
- Ensure you have Jupyter Notebook and the required dependencies
- Open and follow the steps in `cosmosage.ipynb` for a guide to training and using the model.

## Contributing

If you'd like to get involved, please contact me at <tijmen.dehaan@gmail.com>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
