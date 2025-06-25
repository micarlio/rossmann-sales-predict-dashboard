from setuptools import setup, find_packages

setup(
    name="rossmann-sales-dashboard",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "dash==2.14.2",
        "dash-bootstrap-components==1.5.0",
        "pandas==2.1.4",
        "plotly==5.18.0",
        "numpy==1.26.2",
        "statsmodels==0.14.1",
        "scikit-learn==1.3.2",
        "gunicorn==21.2.0",
    ],
) 