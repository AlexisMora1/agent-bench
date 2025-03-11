# 🧬 Graph Evaluator with Conversational Setup

The Graph Evaluator object is the centre of the whole project, it contains the necessary stuff to:

1. Dynamically build your graph from configurations
2. Evaluate your configurations by dataset
3. Evaluate your configurations by conversational setup

## 🔍 What means to evaluate an architecture by a conversational setup?

Some of the times when we build an agent to solve a problem, performing the task may take more than one turn of human-agent interaction to reach the solution, for this kind of tasks we have implemented the **evaluate_with_conversational_setup()** method.

This method allows you not only to run a complete conversation defining the maximum number of turns, but also to customize your type of interaction and evaluation for each setup by implementing an **agent_handler()**.

## 📚 Conversational Setup.

In order to make the evaluation of the graph as flexible as possible, it was decided to use a list of dictionaries in the conversational setup, containing in each key the following important and necessary values

- `iterations`: Number of conversation iterations - `invoke_state`: Initial state for the conversation - `preserving_keys`: State keys to preserve between turns - `conversational_agent`: Agent that automatically handles conversation evaluation with your current graph configuration

## 🧰 Basic Setup

```python
from agent_bench.graph_evaluator import GraphEvaluator
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Define your conversational agent
def conversational_agent(messages: List[AnyMessage]):
    system = SystemMessage(content="Your system prompt")
    try:
        response = llm.invoke([system] + messages)
        messages.append(response)
    except Exception as e:
        print(f"Error: {e}")
        messages.append(AIMessage(content=''))
    return messages

# Define agent setup
agent_setup = {
    "iterations": 3,  # Number of conversation turns
    "invoke_state": {
        "messages": [HumanMessage(content="Initial message")],
    },
    "preserving_keys": ["messages"],  # State keys to preserve between iterations
    "conversational_agent": conversational_agent
}

# Initialize evaluator
graph_evaluator = GraphEvaluator(
    configurations=configurations,
    start_node="retrieve",
    evaluators=[]
)
```

## ▶️ Running Evaluation

```python
results = graph_evaluator.evaluate_with_conversational_config(
    experiment_name="Test1",
    graph_state_class=GraphState,
    agent_handler=AWSProvider.bedrock_agent_handler,  # or your custom handler
    agent_setup=[agent_setup],
    batch_size=3
)
```

## 🗝 Key Components


### Agent Handler

An **Agent Handler** is a function that will allow you to manage the turn-based execution of your architecture.

Currently as an example we have **bedrock_agent_handler** in the **integrations** module, which will be useful to evaluate architectures that manage the history of the conversation through a list of messages in a key **“messages”**.

## Evaluators

For conversational evaluation, create evaluators that inherit from BaseEvaluator:

```python
class ConversationalEvaluator(BaseEvaluator):
    def evaluate(self, conversation_history):
        # Implement conversation evaluation logic
        return score
```

## Output

The evaluation generates:
- Conversation metrics
- System resource usage
- PDF report with evaluation result