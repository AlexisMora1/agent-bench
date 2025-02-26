from setuptools import setup, find_packages

setup(
    name="agent_bench",
    version="0.1.0",
    description="Use this modules to evaluate easily your agents, compare versions or test models",
    packages=find_packages(),  # Encuentra automáticamente todos los submódulos
    install_requires=[
        "langchain_aws==0.2.13",
        "langgraph==0.2.75",
        "matplotlib==3.10.0",
        "numpy==2.2.3",
        "reportlab==4.3.1",
        "rich==13.9.4",
        "sentence_transformers==3.4.1",
        "setuptools==75.8.2",
        "tqdm==4.67.1",
    ],  # Agrega dependencias si es necesario
    python_requires=">=3.7",
)