# Agent-Bench Evaluators Documentation

## 📝 Overview

Evaluators in Agent-Bench are essential components that measure different aspects of agent performance. The evaluator system is built around the `BaseEvaluator` class, which provides a flexible framework for creating custom evaluation metrics.

## 🏗 BaseEvaluator Class

The `BaseEvaluator` class serves as the foundation for all evaluators in the system.

### ⛓ Key Components

```python
class BaseEvaluator:
    def __init__(
        self,
        state_key: str = None,
        aggregation: Callable = None,
        **kwargs
    ):
        self.state_key = state_key
        self.aggregation = aggregation
        self.default_plot = kwargs.get("default_plot", False)
```

- `state_key`: Specifies which part of the agent's output to evaluate
- `aggregation`: Function to aggregate multiple evaluation results
- `default_plot`: Boolean flag to enable automatic plotting of results

### Core Methods

1. **extract_from_state**
```python
def extract_from_state(self, state: Dict):
    if self.state_key:
        return state.get(self.state_key, None)
    return state
```
Extracts relevant data from the agent's state using the configured state_key.

2. **evaluate**
```python
def evaluate(self, model_output, output_data):
    raise NotImplementedError
```
Abstract method that must be implemented by concrete evaluators.

## Pre-built Evaluators

### 1. SimilarityEvaluator

Measures semantic similarity between model outputs and expected outputs using sentence embeddings.

```python
class SimilarityEvaluator(BaseEvaluator):
    def __init__(
        self,
        state_key,
        aggregation: Callable,
        **kwargs
    ):
        super().__init__(state_key, aggregation, **kwargs)
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
```

🌌 **Usage Example:**
```python
similarity_evaluator = SimilarityEvaluator(
    state_key="generation",
    aggregation=np.mean,
    default_plot=True
)
```

**Key Features:**
- Uses multilingual sentence transformers
- Computes cosine similarity between embeddings
- Supports automatic plotting of similarity distributions

### 2. AccuracyEvaluator

Evaluates binary classification accuracy and generates confusion matrices.

```python
class AccuracyEvaluator(BaseEvaluator):
    def __init__(self, state_key):
        super().__init__(state_key)

```

🌌 **Usage Example:**
```python
accuracy_evaluator = AccuracyEvaluator(
    state_key="model_response"
)
```

**Key Features:**
- Binary classification evaluation
- Custom confusion matrix plotting
- Support for statistical tests

## BaseConversationalEvaluator

Extended evaluator class for conversational agents.

```python
class BaseConversationalEvaluator(BaseEvaluator):
    def __init__(
        self,
        state_key: str = None,
        iterations: int = 1,
        conversational_agent: Callable = None,
        **kwargs
    ):
        super().__init__(state_key, **kwargs)
        self.iterations = iterations
        self.conversational_agent = conversational_agent
```

**Key Features:**
- Supports multi-turn conversation evaluation
- Manages conversation state across iterations
- Customizable conversation flow

## Creating Custom Evaluators

### Basic Template

```python
class CustomEvaluator(BaseEvaluator):
    def __init__(self, state_key, **kwargs):
        super().__init__(state_key, **kwargs)
        # Initialize any required models or resources

    def evaluate(self, model_output, output_data):
        # Extract relevant data
        model_data = self.extract_from_state(model_output)
        expected_data = self.extract_from_state(output_data)
        
        # Implement evaluation logic
        return evaluation_score

    def custom_plot(self, dataset: Dict, file_prefix: str):
        # Optional: Implement custom visualization
        return ["path/to/generated/plot.png"]
```

### Best Practices

1. **State Management**
   - Always use `extract_from_state` to access data
   - Handle missing data gracefully

2. **Error Handling**
   - Implement robust error handling in evaluate()
   - Return sensible default values on failure

3. **Visualization**
   - Implement `custom_plot` for specialized visualizations
   - Use `default_plot=True` for standard metrics

4. **Performance**
   - Initialize heavy resources (like models) in __init__
   - Cache computations when possible

## Integration with GraphEvaluator

Evaluators are used in GraphEvaluator to assess agent performance:

```python
graph_evaluator = GraphEvaluator(
    configurations=configurations,
    start_node="retrieve",
    evaluators=[
        SimilarityEvaluator(
            state_key="generation",
            aggregation=np.mean,
            default_plot=True
        ),
        AccuracyEvaluator(state_key="model_response")
    ]
)
```

The evaluators will automatically:
- Process each agent response
- Aggregate results across experiments
- Generate visualizations in the final report