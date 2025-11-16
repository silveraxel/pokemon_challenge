# pokemon_challenge
FDS challenge

1. Create a Virtual Environment

To work properly with this project, you should create a Python virtual environment and activate it.

python3 -m venv venv
source venv/bin/activate

2. Install Dependencies

A requirements.txt file is already included in the project.
Install all required packages using:

pip install -r requirements.txt

3. Input Files

The project expects the dataset files to be placed inside the input/ directory.

Required files:

input/train.jsonl

input/test.jsonl

Make sure these files are correctly placed before running the project.

4. Running the Project

Once the environment is activated and the dependencies are installed, you can run the project normally. For example:

python main.py






Project File and Folder Description

Toolbox.py

This file contains helper functions that are used by the Kaggle notebook.
Its purpose is to provide reusable utilities that simplify preprocessing, evaluation, and other operations performed in the notebook environment.

Libreries.py

This file includes all the functions required for the main script to run correctly.
It acts as the core logic layer of the project, providing the necessary methods for data processing, model training, predictions, and other internal operations used by main.py.

attempted_trials/

This folder contains several experimental approaches that were initially explored but later abandoned because they did not produce satisfactory or reliable results.
These files serve as archived prototypes or alternative strategies that were tested during development but are not part of the final workflow.