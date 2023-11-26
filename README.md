# cosmosage

## Introduction

Large language models are emerging as powerful tools for many natural language tasks. Very large parameter counts of 1e11 or more are needed for a general-purpose model such as GPT-4. However, even small models can be extremely powerful if the application is sufficiently narrow.

cosmosage is an attempt to fine-tune a relatively modest large language model on cosmology-specific datasets with the goal of making a general-purpose natural-language assistant for cosmologists.

## Author

Tijmen de Haan 
Email: <tijmen.dehaan@gmail.com>

## Project Structure

The project includes the following files, as well as some more helper files.

1. `cosmosage.ipynb`: This notebook contains the main code for fine-tuning the language model on cosmology-specific datasets. It goes through steps for data collection, preprocessing, model training, and evaluation.

2. `textbooks.ipynb`: This notebook is used for processing and preparing the dataset extracted from public-domain astro-related textbooks. The data is cleaned and saved in JSON format for further use. See the notebook for more details. 

3. `arxiv.py`: Adapted from Karpathy's excellent `arxiv-sanity-lite`, this is used to interface with the arXiv API.

4. `fine_tune.py`: methods for interacting with `pytorch`` and doing the actual deep learning


## Syntax, Code Style, Tools Used

The .py files are kept consistently formatted with `black` on its default settings.

The codebase was written with the use of Pylance, GitHub Copilot, GPT-4, and VSCode fork `cursor`.

Pylint shows no warnings as of the writing of this README file.

## Usage

To get started with cosmosage:
- Ensure you have Jupyter Notebook and the required dependencies
- Open and follow the steps in `cosmosage.ipynb` for a guide to training and using the model.

## History

 - 2023 Nov 21 - project start
 - 2023 Nov 25 - completed fine tune "cosmosage_v1" based on arxiv papers and physics QA pairs, taking ~10 hours on 1x A6000
 - 2023 Nov 26 - collected open-access astro textbooks, began 2nd training run expected to take ~48 hours

## To-Do List

- **Data Improvement**: Explore more diverse datasets, experiment with token chunking, and extend the arXiv dataset.
- **Hyperparameter Optimization**: Fine-tune hyperparameters like learning rate and gradient clipping.
- **Validation Data**: Set aside a portion of data for validation purposes.
- **Deployment Strategy**: Decide on the front-end platform and hardware options, and determine access permissions.

## Contributing

If you'd like to get involved, please contact me at <tijmen.dehaan@gmail.com>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
