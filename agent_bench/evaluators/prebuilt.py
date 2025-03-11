from .base import BaseEvaluator, BaseConversationalEvaluator
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Callable, Any
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(vec1, vec2):
    """Calcula la similitud coseno entre dos vectores NumPy."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

class SimilarityEvaluator(BaseEvaluator):
    """
    Evaluates the similarity between the model output and the true output using cosine similarity.
    """

    def __init__(self, state_key, aggregation: Callable, **kwargs):
        """
        Initializes the SimilarityEvaluator with a state key and an optional test type.

        Args:
            state_key (str): The key to extract the relevant state from the model output.
            test (str): The type of test to be used. Defaults to "anova".
        """
        super().__init__(state_key, aggregation, **kwargs)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def evaluate(self, model_output, output_data):
        """
        Evaluates the similarity between the input and output of the graph.

        Args:
            model_output (Dict): The output from the model.
            output_data (Dict): The true output data.

        Returns:
            float: The cosine similarity score between the model output and the true output.
        """
        model_output = self.extract_from_state(model_output)
        true_output = self.extract_from_state(output_data)
        embeddings = self.model.encode([model_output, true_output]) # Obtener los embeddings

        return cosine_similarity(embeddings[0], embeddings[1])


class AccuracyEvaluator(BaseEvaluator):
    """
    Evaluates the accuracy of the model by comparing the model output with the true output.
    """

    def __init__(self, state_key, test="mcnemar"):
        super().__init__(state_key)
        self.test=test

    def evaluate(self, model_output, output_data):
        """
        Evaluates the accuracy of the model's response for binary outputs.

        Args:
            model_output (Dict): The output from the model.
            output_data (Dict): The true output data.

        Returns:
            int: 1 if the model output matches the true output, 0 otherwise.
        """
        model_output = self.extract_from_state(model_output)
        output_data = self.extract_from_state(output_data)

        return 1 if model_output == output_data else 0
    
    def custom_plot(self, dataset: Dict, file_prefix: str):
        """
        Generates confusion matrix plots for different configurations.

        Args:
            dataset (Dict): The dataset containing evaluation results.
            file_prefix (str): The prefix for the output plot files.

        Returns:
            List[str]: A list containing the file paths of the generated plots.
        """
        configs = set(dataset.get("config"))
        file_name_list = []
        # Extraer los valores reales y predichos
        for config in configs:
            y_true = np.array([dataset.get("dataset_output", [])[i].get("model_response") for i in range(len(dataset.get("config"))) if dataset.get("config")[i] == config])
            y_pred = np.array([dataset.get("model_output", [])[i].get("model_response") for i in range(len(dataset.get("config"))) if dataset.get("config")[i] == config])

            classes = [False, True]  

            # Crear matriz de confusión vacía
            conf_matrix = np.zeros((2, 2), dtype=int)

            # Llenar la matriz de confusión
            for true_label, pred_label in zip(y_true, y_pred):
                if true_label == None or pred_label == None:
                    continue
                true_index = classes.index(true_label)
                pred_index = classes.index(pred_label)
                conf_matrix[true_index, pred_index] += 1

            # Graficar la matriz de confusión con Matplotlib
            _, ax = plt.subplots(figsize=(6, 5))
            cax = ax.imshow(conf_matrix, cmap="Blues", aspect="auto")

            # Agregar anotaciones en cada celda
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

            # Configurar ejes
            ax.set_xticks(range(2))
            ax.set_yticks(range(2))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel("Predictions")
            ax.set_ylabel("Real Values")
            ax.set_title(f"{config} Confusion Matrix")

            # Agregar barra de color
            file_name = f"./images/{file_prefix}_{config}.png"
            plt.colorbar(cax)
            plt.savefig(file_name)
            plt.close()
            file_name_list.append(file_name)
        
        return file_name_list
             