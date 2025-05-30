# Update 2025-05

This repository documents some of the code used in January 2024 for https://doi.org/10.1016/j.ascom.2025.100934 It may still contain some useful snippets, but it is largely out of date and I would not recommend using this as a starting point for modern LLM fine-tuning work.

# cosmosage

## Introduction

Large language models are emerging as powerful tools for many natural language tasks. Very large parameter counts of 1e11 or more are needed for a general-purpose model such as GPT-4. However, even small models can be extremely powerful if the application is sufficiently narrow.

cosmosage is an attempt to fine-tune a relatively modest large language model on cosmology-specific datasets with the goal of making a general-purpose natural-language assistant for cosmologists.

## Author

Tijmen de Haan <tijmen.dehaan@gmail.com>

I gave a colloquium on cosmosage, which you can watch here https://www.youtube.com/watch?v=azwfG2UTNEY

I also wrote an article on cosmosage which is available as preprint at https://arxiv.org/abs/2407.04420

## Project Structure

A walkthrough of the project is given in iPython Notebook format in `cosmosage.ipynb`. This notebook walks through the several-step process for fine-tuning the language model on cosmology-specific datasets. It goes through steps for data collection, preprocessing, model training, and evaluation.

## Syntax, Code Style, Tools Used

The .py files are kept consistently formatted with `black` on its default settings.

The codebase was written with the use of Pylance, GitHub Copilot, GPT-4, and VSCode fork `cursor`.

## Usage

To get started with training cosmosage:
- Ensure you have Jupyter Notebook and the required dependencies
- Open and follow the steps in `cosmosage.ipynb` for a guide to training and using the model.

If you'd like to run a version of cosmosage that I've trained, head over to https://huggingface.co/Tijmen2 where I have and will continue to post versions.

## Contributing

If you'd like to get involved, please contact me at <tijmen.dehaan@gmail.com>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
