from typing import Dict

class BaseEvaluator:
    """This Base Evaluator class will help you build reusable evaluators with any network.
        Dont forget to add the evaluate and custom_plot methods if you want the evaluation to be plotted
    """

    def __init__(self, state_key=None, test=None):
        """
        Initializes the evaluator with an optional key to extract data from the state.
        
        Args:
            state_key (str): key to extract from the state. If None, the complete state is passed.
        """
        self.state_key = state_key

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
        raise NotImplementedError("El método evaluate() debe ser implementado por las clases hijas.")



