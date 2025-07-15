from setuptools import setup, find_packages

setup(
    name='llm4ad',
    version='1.0',
    author='Optima',
    description='Large Language Model for Algorithm Design Platform ',
    packages=find_packages(),  
    package_dir={'': '.'}, 
    python_requires='>=3.9,<=3.11',
    install_requires=[
        'numpy<2',
        'tensorboard',
        'scipy',
        'tqdm',
        'requests',
        'openai',
        'pytz',
        'matplotlib',
        'python-docx',
        'gym',
        'ttkbootstrap'
    ]
)