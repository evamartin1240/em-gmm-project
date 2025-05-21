from setuptools import setup, find_packages

setup(
    name='emgmm',
    version='0.1',
    description='Expectation-Maximization for Gaussian Mixture Models from scratch',
    author='Eva Martín, Jonàs Salat, Albert Vidal',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'plotly',
        'tqdm'
    ],
)

