from setuptools import setup, find_packages

setup(
    install_requires=['pandas', 'opencv_python', 'opencv_contrib_python', 'matplotlib', 'sklearn', 'datetime', 'seaborn',
                      'plotly', 'statsmodels', 'wget', 'keras', 'tensorflow', 'folium', 'cython', 'pystan==2.17.1',
                      'convertdate', 'fbprophet', 'pmdarima', 'requests', 'bs4'],
    name='covid_19_analysis',
    version='1.0.0',
    description='ANALYSIS OF COVID-19 CORONAVIRUS PANDEMIC',
    long_description='Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory '
                     'syndrome coronavirus 2 (SARS-CoV-2). The disease was first identified in 2019 in Wuhan, '
                     'the capital of Hubei, China, and has since spread globally, '
                     'resulting in the 2019–20 coronavirus pandemic.',
    url='',
    author='Sveta Raboy',
    author_email='',
    license='',
    classifiers=[],
    keywords={'COVID-19', 'CORONAVIRUS'},
    setup_requires=[],
    package_data={},
    packages=find_packages(),
    python_requires='>=3.5.0',
)
