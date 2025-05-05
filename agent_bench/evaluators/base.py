from typing import Dict, Callable, Any, List
from langchain_core.messages import HumanMessage


class BaseEvaluator:
    """This Base Evaluator class will help you build reusable evaluators with any network.
    Dont forget to add the evaluate and custom_plot methods if you want the evaluation to be plotted
    """

    def __init__(self, state_key: str = None, aggregation: Callable = None, **kwargs):
        """
        Initializes the evaluator with an optional key to extract data from the state.

        Args:
            state_key (str): key to extract from the state. If None, the complete state is passed.
        """
        self.state_key = state_key
        self.aggregation = aggregation
        self.default_plot = kwargs.get("default_plot", False)

    def extract_from_state(self, state: Dict):
        """
        Extracts the value of the state using the configured key.

        Args:
            state (dict): The state returned by the network.

        Returns:
            The value extracted from the state or the entire state if no key is defined.
        """
        # print(f"State received in extract_from_state: {state}")
        if self.state_key:
            return state.get(self.state_key, None)
        return state

    def evaluate(self, model_output, output_data):
        """
        Base method for evaluation. Must be implemented by the evaluators.

        Args:
            graph (StateGraph): The graph that was executed.
            state (dict): The output of the graph with metadata as execution time.

        Returns:
            Result of the evaluation.
        """
        raise NotImplementedError(
            "El método evaluate() debe ser implementado por las clases hijas."
        )


class BaseConversationalEvaluator(BaseEvaluator):

    def __init__(
        self,
        state_key: str = None,
        iterations: int = 1,
        conversational_agent: Callable = None,
        **kwargs
    ):
        """
        Initializes the conversational evaluator.

        Args:
            state_key (str): Key to extract from the state.
            iterations (int): Number of iterations for the conversation.
        """
        super().__init__(state_key, **kwargs)
        self.iterations = iterations
        self.conversational_agent = conversational_agent

    def evaluate_conversation(self) -> Dict:
        """
        Base method for evaluating a conversation. Must be implemented by subclasses.

        Args:
            **kwargs: Additional arguments that may be needed for evaluation.

        Raises:
            NotImplementedError: If not implemented by subclasses.

        Returns:
            Dict: Result of the conversation evaluation.
        """
        raise NotImplementedError(
            "The evaluate_conversation() method must be implemented by subclasses."
        )
