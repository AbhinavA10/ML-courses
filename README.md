# ML-courses
Data Science Courses

Courses:
1. Data Analysis with Python: https://www.coursera.org/learn/data-analysis-with-python 
2. Data Visualization with Python https://www.coursera.org/learn/python-for-data-visualization
3. Machine Learning with Python https://www.coursera.org/learn/machine-learning-with-python 



## Steps to run:
- `conda env create --file environment.yml`
- `conda activate ml`
- `jupyter labextension install @jupyter-widgets/jupyterlab-manager` [Source 1](https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab), [Source 2](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension)
- `jupyter-lab`

## To update conda env from `.yml` file:

```
conda activate ml
conda env update --file environment.yml
```