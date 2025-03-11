from typing import Literal, List, Annotated, Tuple, Dict, Callable, Any
from langchain_aws import (
    ChatBedrock,
    ChatBedrockConverse,
)
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


SECTION_NAME = "Configuration"

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


def add_evaluator_descriptions(evaluators: List[BaseEvaluator], y_pos, c: canvas.Canvas = None):
    _, height = letter
    for evaluator in evaluators:
        evaluator_name = evaluator.__class__.__name__
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, evaluator_name)
        y_pos -= 20
        
        c.setFont("Helvetica", 10)
        doc_text = evaluator.__doc__ or "No description available"
        # Wrap text para múltiples líneas
        for line in doc_text.split('\n'):
            wrapped_text = textwrap.fill(line, width=80)
            for wrapped_line in wrapped_text.split('\n'):
                c.drawString(70, y_pos, wrapped_line)
                y_pos -= 15
        
        c.drawString(70, y_pos, f"State Key: {evaluator.state_key or 'None'}")
        y_pos -= 15
        c.drawString(70, y_pos, f"Default Plot: {'Yes' if evaluator.default_plot else 'No'}")
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
        "cpu_percent": process.cpu_percent(interval=0.1),  # Segunda llamada con intervalo
        "num_threads": process.num_threads()
    }