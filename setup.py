from setuptools import setup, find_packages

setup(
    name="agent_bench",
    version="0.1.0",
    description="Use this modules to evaluate easily your agents, compare versions or test models",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.10.0",
        "numpy==2.2.3",
        "reportlab==4.3.1",
        "rich==13.9.4",
        "setuptools==75.8.2",
        "tqdm==4.67.1",
        "json-tricks==3.17.3",
        'langgraph==0.2.75'
    ],
    extras_require={
        'prebuilt': [
            'sentence_transformers==3.4.1',
            'langchain_aws==0.2.13',
        ],
        'aws': [
            'langchain_aws==0.2.13',
            'langgraph==0.2.75',
        ],
        'all': [
            'sentence_transformers==3.4.1',
            'langchain_aws==0.2.13',
            'langgraph==0.2.75',
        ]
    },
    python_requires=">=3.7",
)