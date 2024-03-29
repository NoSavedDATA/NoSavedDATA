from setuptools import setup, find_packages

print(f"{find_packages()}")

setup(
    name='nosaveddata',
    description = "High level tools from my neural network projects",
    version='0.12',
    author='Augusto Seben da Rosa',
    author_email='snykralafk@gmail.com',
    packages=find_packages(),
    install_requires=[
        #'transformers==4.31.0',
        #'gradio==3.50.2',
        'diffusers',
        'wandb',
        'einops',
        'opencv-python==4.8.0.74',
        'matplotlib',
        'pandas'
    ],
    zip_safe=False,
)