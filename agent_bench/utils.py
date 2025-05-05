from typing import Literal, List, Annotated, Tuple, Dict, Callable, Any
from langchain_aws import (
    ChatBedrock,
    ChatBedrockConverse,
)
from cycler import cycler
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from typing import Dict, List, Any, Tuple
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap
import psutil
import os
from agent_bench.evaluators import BaseEvaluator
from langgraph.graph import StateGraph, START, END


SECTION_NAME = "Configuration"


def build_graph(
    start_node: str,
    nodes: List[Callable],
    edges: List[Tuple],
    graph_state: Dict,
    eval_config: Dict,
):
    """
    Dynamically builds the graph with nodes and connections.

    Args:
        nodes (List[Callable]): A list of node functions.
        edges (List[Tuple]): A list of edges connecting the nodes.
        graph_state (Dict): The initial state of the graph.
        eval_config (Dict): Configuration for the evaluation.

    Returns:
        StateGraph: The compiled state graph.
    """
    agent = StateGraph(graph_state)

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]🔧 Building graph...", total=len(nodes) + len(edges)
        )

        for node in nodes:
            node_name = node.__name__
            config = {"function": node}
            agent.add_node(
                node_name,
                create_graph_component(config, eval_config.get(node_name, {})),
            )
            progress.update(task, advance=1)

        agent.add_edge(START, start_node)

        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                if edge[1] == "END":
                    agent.add_edge(edge[0], END)
                    progress.update(task, advance=1)
                    continue
                if isinstance(edge[1], str):
                    target = edge[1]
                    agent.add_edge(edge[0], target)
                else:
                    target = create_graph_component(
                        {"function": edge[1]}, eval_config.get(edge[1].__name__, {})
                    )
                    agent.add_conditional_edges(edge[0], target)

            elif isinstance(edge, tuple) and len(edge) == 3:
                target = (
                    edge[1]
                    if isinstance(edge[1], str)
                    else create_graph_component(
                        {"function": edge[1]}, eval_config.get(edge[1].__name__, {})
                    )
                )
                agent.add_conditional_edges(edge[0], target, edge[2])
            progress.update(task, advance=1)

    return agent.compile()


def create_graph_component(config, eval_config):
    def component_function(state):
        return (
            config["function"](state, **eval_config)
            if eval_config
            else config["function"](state)
        )

    return component_function


def truncate_cell_content(content: str, max_length: int = 100) -> str:
    """
    Trunca el contenido de una celda si excede el largo máximo.

    Args:
        content (str): Contenido de la celda
        max_length (int): Longitud máxima permitida

    Returns:
        str: Contenido truncado si excede max_length
    """
    if len(content) <= max_length:
        return content

    # Reservamos caracteres para "..."
    segment_length = (max_length - 3) // 2
    return f"{content[:segment_length]}...{content[-segment_length:]}"


def display_results(results: Dict, console: Console, max_cell_length: int = 100):
    """ "
    Displays the evaluation results in a table.
    """
    console.clear()
    table = Table(title="Evaluation Results", style="white")

    # Mantener un orden consistente en las columnas
    columns = sorted(results.keys())

    # Agregar columnas a la tabla
    for column in columns:
        table.add_column(column, style="white")

    # Asegurar que todas las claves tengan el mismo número de elementos
    num_rows = min(len(values) for values in results.values())

    if num_rows > 10:
        # Mostrar las primeras 5 filas
        for i in range(5):
            row_data = [
                truncate_cell_content(str(results[key][i]), max_cell_length)
                for key in columns
            ]
            table.add_row(*row_data)

        # Agregar una fila de separación (opcional, para claridad)
        table.add_row(*["..."] * len(columns))

        # Mostrar las últimas 5 filas
        for i in range(num_rows - 5, num_rows):
            row_data = [
                truncate_cell_content(str(results[key][i]), max_cell_length)
                for key in columns
            ]
            table.add_row(*row_data)
    else:
        # Agregar todas las filas si no son más de 10
        for i in range(num_rows):
            row_data = [
                truncate_cell_content(str(results[key][i]), max_cell_length)
                for key in columns
            ]
            table.add_row(*row_data)

    console.print(table)
    console.print("[bold white]Evaluation completed![/bold white] ✅🎉\n")


def add_page_header(page_title="", c: canvas.Canvas = None):
    """Añade encabezado consistente a cada página"""
    width, height = letter
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Evaluation Report")
    if page_title:
        c.setFont("Helvetica", 16)
        c.drawString(50, height - 80, page_title)
    c.setFont("Helvetica", 10)
    c.drawString(width - 150, height - 30, f"Generated: {time.strftime('%Y-%m-%d')}")
    c.line(50, height - 90, width - 50, height - 90)


def add_section_title(title, y_position, c: canvas.Canvas = None):
    """Añade título de sección con formato consistente"""
    width, _ = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, title)
    c.line(50, y_position - 5, width - 50, y_position - 5)
    return y_position - 30


def add_evaluator_descriptions(
    evaluators: List[BaseEvaluator], y_pos, c: canvas.Canvas = None
):
    _, height = letter
    for evaluator in evaluators:
        evaluator_name = evaluator.__class__.__name__
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, evaluator_name)
        y_pos -= 20

        c.setFont("Helvetica", 10)
        doc_text = evaluator.__doc__ or "No description available"
        # Wrap text para múltiples líneas
        for line in doc_text.split("\n"):
            wrapped_text = textwrap.fill(line, width=80)
            for wrapped_line in wrapped_text.split("\n"):
                c.drawString(70, y_pos, wrapped_line)
                y_pos -= 15

        c.drawString(70, y_pos, f"State Key: {evaluator.state_key or 'None'}")
        y_pos -= 15
        c.drawString(
            70, y_pos, f"Default Plot: {'Yes' if evaluator.default_plot else 'No'}"
        )
        y_pos -= 30

        if y_pos < 100:  # Nueva página si no hay espacio
            c.showPage()
            add_page_header("Evaluators Description (cont.)")
            y_pos = height - 150


def measure_system_resources():
    """Measure current system resources."""
    process = psutil.Process(os.getpid())
    _ = process.cpu_percent(interval=0.1)  # Primera llamada para evitar el 0%

    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,  # Convertir a MB
        "cpu_percent": process.cpu_percent(
            interval=0.1
        ),  # Segunda llamada con intervalo
    }


def plot_paired_data(
    metric_name: str, data: List[List], configuration_names: List[str] = []
) -> str:
    _, ax = plt.subplots()
    n = len(data[0])
    m = len(data)

    x_positions = np.linspace(1, m, m)

    if not configuration_names:
        configuration_names = [f"Config {i}" for i in range(len(data))]

    custom_cycler = cycler(color=["c", "m", "y", "k"])
    ax.set_prop_cycle(custom_cycler)

    for i, (x, y, config_name) in enumerate(
        zip(x_positions, data, configuration_names)
    ):
        ax.scatter([x] * n, y, label=config_name)

    for i in range(n):
        ax.plot(
            x_positions,
            [data[j][i] for j in range(m)],
            color="gray",
            linestyle="--",
            linewidth=0.7,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(configuration_names)
    ax.set_ylabel("value")
    ax.set_title(f"{metric_name} by configuration")
    ax.legend()

    filename = f"./images/{metric_name}_paired.png"
    plt.savefig(filename)
    plt.close()

    return filename


def plot_violin(
    metric_name: str, data: List[List], configuration_names: List[str] = []
) -> str:
    _, ax = plt.subplots()
    n = len(data[0])
    m = len(data)

    if not configuration_names:
        configuration_names = [f"Config {i}" for i in range(len(data))]

    custom_cycler = cycler(color=["c", "m", "y", "k"])
    ax.set_prop_cycle(custom_cycler)

    x_positions = np.linspace(1, m, m)

    for x, element, config_name in zip(x_positions, data, configuration_names):
        ax.violinplot(element, positions=[x], side="high", showmeans=True)

    for i, (x, y, config_name) in enumerate(
        zip(x_positions, data, configuration_names)
    ):
        ax.scatter([x - 0.2] * n, y, label=config_name)

    # Etiquetas y título
    ax.set_xticks(x_positions)
    ax.set_xticklabels(configuration_names)
    ax.set_ylabel("value")
    ax.set_title(f"{metric_name} by configuration")
    ax.legend(loc="upper left", ncols=m)

    # Mostrar gráfico
    filename = f"./images/{metric_name}_violin_.png"
    plt.savefig(filename)
    plt.close()

    return filename


def plot_bar_data(
    metric_name: str, data: List, configuration_names: List[str] = []
) -> str:
    width = 0.25

    m = len(data)
    _, ax = plt.subplots(layout="constrained")

    if not configuration_names:
        configuration_names = [f"Config {i}" for i in range(len(data))]

    custom_cycler = cycler(color=["c", "m", "y", "k"])
    ax.set_prop_cycle(custom_cycler)

    x_positions = np.linspace(1, m, m)

    for x, element, config_name in zip(x_positions, data, configuration_names):

        rects = ax.bar(x, element, width, label=config_name)
        ax.bar_label(rects, padding=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Length (mm)")
    ax.set_title(f"{metric_name} by configuration")
    ax.set_xticks(x_positions, configuration_names)
    ax.legend(loc="upper left", ncols=3)
    # ax.set_ylim(0, 250)

    filename = f"./images/{metric_name}_bar_.png"
    plt.savefig(filename)
    plt.close()

    return filename
