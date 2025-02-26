from typing import Literal, List, Annotated, Tuple, Dict, Callable, Any
from langchain_aws import (
    ChatBedrock,
    ChatBedrockConverse,
)
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt

SECTION_NAME = "Configuration"

def get_llm(model_id: str, chat_model: Literal["ChatBedrock", "ChatBedrockConverse"]) -> Callable:
    if chat_model == "ChatBedrock":
        return ChatBedrock(
            model_id= model_id,
            model_kwargs=dict(temperature=0),
            region="us-east-1",
            max_tokens=500,
        )
    elif chat_model == "ChatBedrockConverse":
        return ChatBedrockConverse(
            model= model_id,
            temperature=0,
            max_tokens=500,
        )


def prepare_data_for_anova(data: dict, key: str) -> dict:
    """
    Reorganiza los datos para una prueba ANOVA.

    Args:
        data (dict): Diccionario con los datos.
        key (str): Clave del valor a analizar en la prueba ANOVA.

    Returns:
        dict: Diccionario con listas de valores por configuración.
    """
    grouped_data = defaultdict(list)
    
    for config, value in zip(data["config"], data[key]):
        grouped_data[config].append(value)
    
    return {key: list(grouped_data.values())}


def prepare_data_for_mcnemar(data: dict, key: str) -> np.ndarray:
    """
    Reorganiza los datos en una matriz de contingencia para la prueba de McNemar.

    Args:
        data (dict): Diccionario con los datos.
        key (str): Clave del valor binario a analizar.

    Returns:
        np.ndarray: Matriz de contingencia de tamaño (n x 2).
    """
    contingency_table = []
    unique_configs = sorted(set(data["config"]))

    for config in unique_configs:
        values = [data[key][i] for i in range(len(data["config"])) if data["config"][i] == config]
        count_1 = sum(1 for v in values if v == 1)
        count_0 = sum(1 for v in values if v == 0)
        contingency_table.append([count_0, count_1])  # Filas: configs, Columnas: [0s, 1s]

    return np.array(contingency_table)


def process_default_data(dataset: Dict):
    """Procesa el dataset para obtener estadísticas clave por configuración."""
    processed_data = {}
    
    configs = set(dataset.get("config"))
    for config in configs:
        times = [dataset.get("execution_time")[i] for i in range(len(dataset.get("config"))) if dataset.get("config")[i] == config]
        errors = [dataset.get("error")[i] for i in range(len(dataset.get("config"))) if dataset.get("config")[i] == config]
        
        mean_time = sum(times) / len(times)
        std_dev = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))
        worst_case = max(times)
        error_prob = sum(errors) / len(errors)
        error_count = sum(errors)
        
        processed_data[config] = {
            "exec_time": mean_time,
            "exec_std": std_dev,
            "worst_case": worst_case,
            "error_prob": error_prob,
            "error_count": error_count
        }
    
    return processed_data


def generate_plots(output_dataset: Dict):
    """Genera gráficos de tiempo de ejecución y errores."""

    dataset = process_default_data(output_dataset)

    configs = list(dataset.keys())
    exec_times = [dataset[c]['exec_time'] for c in configs]
    exec_stds = [dataset[c]['exec_std'] for c in configs]
    worst_cases = [dataset[c]['worst_case'] for c in configs]
    error_probs = [dataset[c]['error_prob'] for c in configs]
    error_counts = [dataset[c]['error_count'] for c in configs]
    
    # Gráfico de tiempo de ejecución
    plt.figure()
    plt.bar(configs, exec_times, yerr=exec_stds, capsize=5)
    plt.xlabel(SECTION_NAME)
    plt.ylabel("Execution time (s)")
    plt.title("Execution time by Configuration")
    plt.savefig("./images/exec_time.png")
    
    # Gráfico de peores casos
    plt.figure()
    plt.bar(configs, worst_cases, color='red')
    plt.xlabel(SECTION_NAME)
    plt.ylabel("Worst Case (s)")
    plt.title("Worst case by Configuration")
    plt.savefig("./images/worst_case.png")
    
    # Gráfico de errores
    plt.figure()
    plt.bar(configs, error_probs)
    plt.xlabel(SECTION_NAME)
    plt.ylabel("Error probability")
    plt.title("Error probability by Configuration")
    plt.savefig("./images/error_prob.png")
    
    plt.figure()
    plt.bar(configs, error_counts, color='gray')
    plt.xlabel(SECTION_NAME)
    plt.ylabel("Error Count")
    plt.title("Error count by Configuration")
    plt.savefig("./images/error_count.png")


