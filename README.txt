1. Make sure Python 3.8 or later is installed.
   Check with:  python --version

2. (Recommended) Create and activate a virtual environment:

   python -m venv venv

   Windows:
     venv\Scripts\activate

   Mac / Linux:
     source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the project.

   Option A — run all portions at once (from the project root):
     python run_all.py

   Option B — run each portion individually in order:
     python src/portion1_data.py
     python src/portion2_perceptron.py
     python src/portion3_delta.py
     python src/portion4_activations.py
     python src/portion5_lr_tuning.py
     python src/portion6_comparison.py

   Always run from the project root folder so that relative paths
   to the outputs/ directory resolve correctly.


Output Files
------------

After a successful run, the outputs/ folder will contain:

  X_train.npy               Scaled training features
  X_test.npy                Scaled test features
  y_train.npy               Training labels
  y_test.npy                Test labels
  perc_errors.npy           Perceptron misclassification history
  gd_loss.npy               Delta Rule loss history

  portion1_exploration.png  Dataset scatter plots
  portion2_perceptron.png   Perceptron training results
  portion3_delta.png        Delta Rule training results
  portion4_activations.png  Activation function comparison
  portion5_lr_tuning.png    Learning rate sweep results
  portion6_comparison.png   Final side-by-side comparison


Execution Order Note
--------------------

portion1 must run before all others because it creates the
preprocessed .npy files that every subsequent script loads.
If a later portion fails, re-run portion1 first to regenerate
the data files, then retry the failing script.


Dataset
-------

Source : UCI Machine Learning Repository — Iris Dataset
URL    : https://archive.ics.uci.edu/dataset/53/iris
Loaded via scikit-learn's built-in load_iris() function.
