# ⚔️ Getting started

The evaluation of multi- or single-agent architectures is still a complicated issue with most frameworks, and to overcome this repetitive task, this toolkit was developed to offer you the flexibility to adjust to your use case.

Before starting the installation make sure you have **Python 3.10+** installed, as well as having created your virtual environment.

## 🔨 Instalation

### Using pip

```bash
pip install git+https://github.com/AlexisMora1/agent-bench.git
``` 

There is no more options to install, thank you!

## 🔑 Key Concepts

Before you start using the package we suggest you read this section to get a better understanding of how to best perform your evaluations.

#### 1. 🧪 **Experiments**: 

The evaluations in Agent Bench are based on the experiment concept. Each time you run an evaluation you can provide an experiment_id that will be the key to recognize your evaluation. 

An experiment can have several configurations of architectures and from it the necessary groupings will be made in the report.

#### 2. 🔧 **Configurations**: 

A configuration is a dictionary containing, for each key, another dictionary with the keys "nodes" (list of functions representing the LangGraph nodes), "edges" (list of tuples representing the connection between nodes if it is a 2-valued tuple, and a conditional edge if it is a 3-valued tuple), and "eval_config". 

```python
configurations = {
    "rag": {
        "nodes": rag_nodes,
        "edges": rag_edges,
        "eval_config": rag_eval_config,
    },
    ...
}
``` 

Sometimes with LangGraph it is complex to construct multiple architectures to test each one, so assuming you have a set of functions representing your nodes and edges, you can define this configuration dictionary to evaluate all configurations together

#### 3. 📋 **Evaluation Configuration**

By modifying the connections between nodes or the logic behind them, you can experiment with the capabilities of the architecture, but sometimes it is also necessary to evaluate the performance of the **large language models** involved in the agent's decisions.

So the evaluation configuration is a dictionary containing the node or edge name and its particular configuration.

#### 4. 🔬 **Evaluators**

An **Evaluator** is the other key part of the Graph Evaluator class. To make a good evaluation most of the time you may need to create your own evaluators that can assign a value to the execution results.

When you add an evaluator to an experiment, you can also add the custom_plot() method or the default_plot() attribute to enable the results to be plotted in the final evaluation report.