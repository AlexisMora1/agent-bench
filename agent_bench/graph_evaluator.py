from langgraph.graph import StateGraph, START, END
from .evaluators.base import BaseEvaluator
from .utils import generate_plots
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
import time
import os
import json
import csv
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

console = Console()

class GraphEvaluator:
    """GraphEvaluator will allow you to build agents for evaluation from a dictionary of configurations, a dataset and a series of evaluators (by default ErrorProbability and ExecutionTime)."""
    
    def __init__(self, configurations: Dict[str, Dict], start_node: str, evaluators: List[BaseEvaluator], results_file: str = None):
        """
        Initializes the evaluator with multiple configurations.

        Args:
            configurations (Dict[str, Dict]): A dictionary of configurations for the graph.
            start_node (str): The starting node of the graph.
            evaluators (List[BaseEvaluator]): A list of evaluators to be used for evaluation.
        """
        self.configurations = configurations
        self.start_node = start_node
        self.results = defaultdict(list)
        self.evaluators = evaluators
        self.results_file = results_file
        self.relevant_output_keys = set(eval.state_key for eval in evaluators if hasattr(eval, "state_key"))

        if results_file:
            self.load_results(results_file)

        console.print(f"[bold green] GraphEvaluator initialized[/bold green] with {len(configurations)} configurations.")

    def build_graph(self, nodes: List[Callable], edges: List[Tuple], graph_state: Dict, eval_config: Dict):
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
        chat_agent = StateGraph(graph_state)
        
        with Progress() as progress:
            task = progress.add_task("[cyan]🔧 Building graph...", total=len(nodes) + len(edges))
            
            def create_graph_component(config, eval_config):
                def component_function(state):
                    return config["function"](state, **eval_config) if eval_config else config["function"](state)
                return component_function

            for node in nodes:
                node_name = node.__name__
                config = {"function": node}
                chat_agent.add_node(node_name, create_graph_component(config, eval_config.get(node_name, {})))
                progress.update(task, advance=1)
            
            chat_agent.add_edge(START, self.start_node)
            for edge in edges:
                if isinstance(edge, tuple) and len(edge) == 2:
                    if edge[1] == "END":
                        chat_agent.add_edge(edge[0], END)
                        progress.update(task, advance=1)
                        continue
                    if isinstance(edge[1], str):
                        target = edge[1]
                        chat_agent.add_edge(edge[0], target)
                    else:
                        target =create_graph_component({"function": edge[1]}, eval_config.get(edge[1].__name__, {}))
                        chat_agent.add_conditional_edges(edge[0], target)

                elif isinstance(edge, tuple) and len(edge) == 3:
                    target = edge[1] if isinstance(edge[1], str) else create_graph_component({"function": edge[1]}, eval_config.get(edge[1].__name__, {}))
                    chat_agent.add_conditional_edges(edge[0], target, edge[2])
                progress.update(task, advance=1)

        return chat_agent.compile()


    def evaluate(self, dataset: List[Dict], graph_state_class, generate_report: bool = True, batch_size: int = None, default_values: Dict = {}, experiment_name: str = None):
        """
        Evaluates the graph with a dataset and evaluators.

        Args:
            dataset (List[Dict]): The dataset to be used for evaluation.
            graph_state_class: The class representing the state of the graph.
            generate_report (bool): Whether to generate a report after evaluation. Defaults to True.
            batch_size (int): The size of batches for evaluation. Defaults to None. If None evaluates example by example, in other case it will parallelize evaluations.
            default_values (Dict): Default values to be used in case of errors. Defaults to {}.

        Returns:
            Dict: The results of the evaluation.
        """
        if experiment_name is None:
            experiment_name = str(uuid.uuid4())

        console.print(f"\n[bold cyan] Evaluating {len(self.configurations)} configurations[/bold cyan] on dataset of size {len(dataset)}\n")

        total_evaluation = len(dataset)*len(self.configurations)

        with tqdm(total=total_evaluation, desc="🔍 Evaluating dataset", unit="example") as pbar:
            results = defaultdict(list)
            for config_name, config in self.configurations.items():
                nodes = config["nodes"]
                edges = config["edges"]
                eval_config = config.get("eval_config", {})
                self.graph = self.build_graph(nodes, edges, graph_state_class, eval_config)
                
                if batch_size:
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        j=0
                        for i in range(0, len(dataset), batch_size):
                            console.print(f"\n[bold cyan] Evaluating {j} batch examples [/bold cyan] on configuration [bold cyan] {config_name} [/bold cyan]\n")
                            batch = dataset[i:i + batch_size]
                            futures.extend(executor.submit(self.process_example, example, config_name, default_values, experiment_name) for example in batch)
                            j+=1

                        for future in as_completed(futures):
                            result = future.result()
                            for key, value in result.items():
                                results[key].append(value)
                            pbar.update(1)
                            console.clear()
                
                else:
                    i=0
                    for example in dataset:
                        console.print(f"\n[bold cyan] Evaluating example {i} [/bold cyan] on configuration [bold cyan] {config_name} [/bold cyan]\n")
                        result = self.process_example(example, config_name, default_values, experiment_name)
                        for key, value in result.items():
                            results[key].append(value)
                        pbar.update(1)
                        console.clear()
                        i+=1

            for key, value in results.items():
                self.results[key].extend(value)

        self._display_results()

        if self.results_file:
            self.save_experiment_as(self.results_file)

        if generate_report:
            self.generate_report()

        return self.results


    def load_results(self, filename: str):
        """
        Loads previous results from a JSON file.

        Args:
            filename (str): The name of the file to load results from.
        """
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            console.print(f"[bold green]Loaded previous results from {filename}[/bold green]")
        else:
            console.print(f"[bold red]No previous results found in {filename}[/bold red]")


    def _display_results(self):
        """"
        Displays the evaluation results in a table.
        """
        console.clear()
        table = Table(title="Evaluation Results", style="white")

        # Mantener un orden consistente en las columnas
        columns = sorted(self.results.keys())

        # Agregar columnas a la tabla
        for column in columns:
            table.add_column(column, style="white")

        # Asegurar que todas las claves tengan el mismo número de elementos
        num_rows = min(len(values) for values in self.results.values())

        if num_rows > 10:
            # Mostrar las primeras 5 filas
            for i in range(5):
                row_data = [str(self.results[key][i]) for key in columns]
                table.add_row(*row_data)

            # Agregar una fila de separación (opcional, para claridad)
            table.add_row(*["..."] * len(columns))

            # Mostrar las últimas 5 filas
            for i in range(num_rows - 5, num_rows):
                row_data = [str(self.results[key][i]) for key in columns]
                table.add_row(*row_data)
        else:
            # Agregar todas las filas si no son más de 10
            for i in range(num_rows):
                row_data = [str(self.results[key][i]) for key in columns]
                table.add_row(*row_data)

        console.print(table)
        console.print("[bold white]Evaluation completed![/bold white] ✅🎉\n")


    def process_example(self, example: Dict, config_name: str, default_values: Dict = {}, experiment_name: str = None):   
        """
        Processes a single example from the dataset.

        Args:
            example (Dict): The example to be processed.
            config_name (str): The name of the configuration being used.
            default_values (Dict): Default values to be used in case of errors. Defaults to {}.

        Returns:
            Dict: The results of processing the example.
        """ 
        eval_results = {}
        input_data = example.get("input")
        output_data = example.get("output")
        start_time = time.time()
        try:
            model_output = self.graph.invoke(input_data)
            error = False   
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            error = True
            model_output = default_values
        end_time = time.time()

        example_results = {
            "config": config_name,
            "dataset_input": input_data,
            "model_output": {k: model_output.get(k, None) for k in self.relevant_output_keys},
            "dataset_output": output_data,
            "execution_time": end_time - start_time,
            "error": error,
            **({"experiment_id": experiment_name} if experiment_name is not None else {})
        }

        for eval in self.evaluators:
            eval_results[eval.__class__.__name__] = eval.evaluate(model_output, output_data)

        return example_results | eval_results


    def save_experiment_as(self, filename: str, encoding: str = "utf-8"):
        """
        Saves a list of dictionaries in JSON or CSV format based on the file extension.

        Args:
            filename (str): The name of the file with extension (.json, .csv).
            encoding (str): The encoding to be used for the file. Defaults to "utf-8".

        Raises:
            ValueError: If the file extension is not valid.
        """
        if not self.results:
            raise ValueError("La lista de datos está vacía.")

        file_ext = filename.split(".")[-1].lower()

        if file_ext == "json":
            with open(filename, "w", encoding=encoding) as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)

        elif file_ext == "csv":
            keys = self.results[0].keys()  # Obtiene las claves del primer diccionario
            with open(filename, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)

        else:
            raise ValueError("Formato no soportado. Usa .json o .csv.")

        console.print(f"[bold white]Experiment saves as {filename}![/bold white] ✅🎉\n")


    def generate_report(self):
        """
        Generates a PDF report with evaluation graphs.
        If no evaluators were provided it will plot only default evaluators, if you created a new evaluator be sure to add the custom_plot() method.
        """
        os.makedirs("./images", exist_ok=True)
        experiment_names = set(self.results.get("experiment_name", ["default"]))

        c = canvas.Canvas("./evaluation_report.pdf", pagesize=letter)
        _, height = letter
        total_bar = (len(self.evaluators) + 2)*len(experiment_names)  # Para la barra de progreso

        with Progress() as progress:
            task = progress.add_task("[cyan]📝 Generating report...", total=total_bar)

            for experiment in experiment_names:
                filtered_results = {key: [value for i, value in enumerate(values) if self.results.get("experiment_name")[i] == experiment] for key, values in self.results.items()}

                # Título del reporte
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, height - 50, "Reporte de Evaluación de Configuraciones")

                # Tamaño y posiciones para una cuadrícula 2x2
                img_width, img_height = 280, 180
                x1, x2 = 50, 320  # Columnas
                y1, y2 = height - 250, height - 450  # Filas

                # Dibujar las 4 gráficas principales en la primera página
                generate_plots(filtered_results)
                c.drawImage("./images/exec_time.png", x1, y1, width=img_width, height=img_height)
                c.drawImage("./images/worst_case.png", x2, y1, width=img_width, height=img_height)
                c.drawImage("./images/error_prob.png", x1, y2, width=img_width, height=img_height)
                c.drawImage("./images/error_count.png", x2, y2, width=img_width, height=img_height)

                c.showPage()  # Nueva página para los evaluadores
                progress.update(task, advance=2)

                # Agregar gráficos personalizados organizados en cuadrícula
                for evaluator in self.evaluators:
                    if hasattr(evaluator, "custom_plot"):
                        evaluator_name = evaluator.__class__.__name__
                        generated_files = evaluator.custom_plot(self.results, evaluator_name)

                        # Título del evaluador en la nueva página
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(100, height - 50, f"Evaluación: {evaluator_name}")

                        x_positions = [50, 320]
                        y_positions = [height - 250, height - 450, height - 630]
                        img_idx = 0

                        for filename in generated_files:
                            x = x_positions[img_idx % 2]
                            y = y_positions[img_idx // 2]

                            c.drawImage(filename, x, y, width=img_width, height=img_height)
                            img_idx += 1

                            # Cada 4 imágenes (2x2) pasamos a una nueva página
                            if img_idx % 5 == 0:
                                c.showPage()
                                c.setFont("Helvetica-Bold", 12)
                                c.drawString(100, height - 50, f"Evaluación: {evaluator_name} (cont.)")
                                img_idx = 0

                        c.showPage()  # Nueva página después de cada evaluación

                    progress.update(task, advance=1)

        c.save()
        console.print("[bold green]Report created successfully![/bold green] ✅🎉\n")

