# 🧬 Graph Evaluator with Evaluation Datasets

The Graph Evaluator object is the centre of the whole project, it contains the necessary stuff to:

1. Dynamically build your graph from configurations
2. Evaluate your configurations by dataset
3. Evaluate your configurations by conversational setup

## 📦 Evaluation Dataset.

When working with agentic systems, most of the time we need the agent to classify an input into a certain label, and this process could be realised with only one turn of interaction, it means an input is given and we expect the model output to be a defined value.

A **dataset** is an object that contains execution inputs with their corresponding output examples.

If your evaluation fits this type of task description, you can pass a dataset when using the **evaluate()** method and configure a custom evaluator to measure the accuracy of the architecture at this task.

## 📡 evaluate() method

The **evaluate()** method is one of the main methods for the GraphEvaluator class. 

It is designed to evaluate a configuration graph in an evaluation dataset and returns an evaluation dataset containing the labels provided before execution, updated with the results and the metrics calculated by the evaluators provided during evaluation.

## 🧰 Basic Setup

```python
from agent_bench.graph_evaluator import GraphEvaluator
from agent_bench.evaluators.prebuilt import SimilarityEvaluator

# Initialize evaluators
similarity_evaluator = SimilarityEvaluator(
    state_key="generation",
    aggregation=np.mean,
    default_plot=True
)

# Define your configurations
configurations = {
    "rag": {
        ...
    },
}

# Initialize the evaluator
graph_evaluator = GraphEvaluator(
    configurations=configurations,
    start_node="retrieve",
    evaluators=[similarity_evaluator],
)
```

## 🖋 Dataset Format
Your dataset should be a list of dictionaries with this structure:
```python
dataset = [
    {
        "input": {"key": "value"},
        "output": {
            "key1": "expected_value1",
            "key2": "expected_value2"
        }
    },
    # More examples...
]
```

## ▶️ Running Evaluation

```python
results = graph_evaluator.evaluate(
    dataset=dataset,
    graph_state_class=GraphState,
    batch_size=10,  # Optional: for parallel processing
    experiment_name="First Test"  # Optional: to identify different runs
)
```

## Generating Reports

```python
# Generate PDF report with plots and metrics
results.generate_report()

# Save results to file
results.save_experiment_as("results.json")  # or "results.csv"
```

## 🗝 Key Parameters

- `configurations`: Dictionary containing different agent architectures to evaluate
- `start_node`: Initial node in your graph
- `evaluators`: List of BaseEvaluator instances
- `batch_size`: Number of examples to process in parallel
- `experiment_name`: Identifier for the evaluation run

## Output

The evaluation generates:
- Execution metrics (time, memory usage, CPU usage)
- Custom metrics from evaluators
- Plots for metrics with default_plot=True
- PDF report with all results
