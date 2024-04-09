from setuptools import setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name='conflare',
    version='0.1.3',    
    description='conformal retreival augmented generation with LLMs',
    url='https://github.com/Mayo-Radiology-Informatics-Lab/conflare',
    author='Moein Shariatnia',
    author_email='moein.shariatnia@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir = {"conflare": "conflare",
                   "conflare.conformal": "conflare/conformal",
                   "conflare.models": "conflare/models",
                   "conflare.utils": "conflare/utils",
                   "conflare.augmented_retrieval": "conflare/augmented_retrieval"},
    packages = ['conflare', 'conflare.conformal', 'conflare.models', 'conflare.utils', 'conflare.augmented_retrieval'],
    install_requires=[  
                        'pypdf==4.1.0',
                        'torch>=2.0.0',
                        'transformers==4.37.2',
                        'bitsandbytes==0.42.0',
                        'accelerate==0.26.1',
                        'openai==1.12.0',
                        'sentence-transformers==2.3.1',
                        'chromadb==0.4.22'
                    ],

    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)