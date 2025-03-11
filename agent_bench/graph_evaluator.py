from langgraph.graph import StateGraph, START, END
from .evaluators.base import BaseEvaluator
from .utils import display_results, add_page_header, add_evaluator_descriptions, measure_system_resources
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from typing import List, Dict, Tuple, Callable, Any
from collections import defaultdict
import time
import os
import json
import json_tricks
import csv
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

console = Console()


class EvaluationDataset:
    """
    ## Evaluation Dataset class

    Stores the data of an evaluation across several experiments and allows you to load previous results to continue your evaluation.

    Its main method is **generate_report()** which will generate a report with default plots if:
        - evaluators have activated the **default_plot** option
    or:
        - A custom plot, in case they have the **custom_plot** method.
    """

    def __init__(self, experiments_results: Dict, evaluators: List[BaseEvaluator] = None):
        self.raw_results = experiments_results
        self.evaluators = evaluators
        # Procesar datos solo una vez al inicializar
        self.processed_data = self._process_data_efficiently()
        self.experiments = sorted(set(experiments_results["experiment_id"]))
        self.configs = sorted(set(experiments_results["config"]))


    def _process_data_efficiently(self) -> Dict:
        """
        Proccess the metrics with evaluators with default_plot=True for plotting and metrics calculation
        """
        metrics = ["execution_time", "error", "memory_usage_mb", "cpu_percent", "num_threads"]
        if self.evaluators:
            metrics.extend(evaluator.__class__.__name__ for evaluator in self.evaluators)

        processed = {metric: {} for metric in metrics}
        
        # Crear índices para acceso rápido
        for metric in metrics:
            if metric not in self.raw_results:
                continue
                
            for i, (exp_id, config, value) in enumerate(zip(
                self.raw_results["experiment_id"],
                self.raw_results["config"],
                self.raw_results[metric]
            )):
                if exp_id not in processed[metric]:
                    processed[metric][exp_id] = {}
                if config not in processed[metric][exp_id]:
                    processed[metric][exp_id][config] = []
                    
                processed[metric][exp_id][config].append(value)
        
        return processed


    def calculate_metrics(self) -> Dict:
        """
        Use the aggregation function provided in the evaluator to calculate the metrics.

        It wirks only in the metrics that are in the processed_data (those that were instantiated with default_plot=True)
        """
        calculation_metric = {
            "execution_time": np.mean,
            "error": sum,
            "memory_usage_mb":np.max, 
            "cpu_percent":np.mean, 
            "num_threads":np.median,
        }
        if self.evaluators:
            calculation_metric.update({
                evaluator.__class__.__name__: evaluator.aggregation 
                for evaluator in self.evaluators
                if evaluator.aggregation is not None  # Solo actualizar si hay función de agregación
            })

        results = {}
        for metric, exp_data in self.processed_data.items():
            results[metric] = {}
            agg_func = calculation_metric.get(metric, np.mean)
            
            for config in self.configs:
                results[metric][config] = [
                    agg_func(exp_data.get(exp_id, {}).get(config, [0]))
                    for exp_id in self.experiments
                ]
                
        return results


    def plot_metrics(self):
        """
        Plot by the metric and experiment the configurations results for the evaluation.

        If cuistom_plot() method was provided in the evaluator it is used instead of the default plot.
        """
        x = np.arange(len(self.experiments))
        width = 0.25
        resulting_files = []

        custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) +
                        cycler(lw=[1, 2, 3, 4]))

        # Determinar qué métricas plotear
        metrics_to_plot = {
            "default": ["execution_time", "error", "memory_usage_mb", "cpu_percent", "num_threads"],
            "custom": {}
        }
        
        if self.evaluators:
            for evaluator in self.evaluators:
                if evaluator.default_plot:
                    metrics_to_plot["default"].append(evaluator.__class__.__name__)
                elif hasattr(evaluator, "custom_plot"):
                    metrics_to_plot["custom"][evaluator.__class__.__name__] = evaluator

        transformed_data = self.calculate_metrics()

        # Plotear métricas por defecto
        for metric in metrics_to_plot["default"]:
            if metric not in transformed_data:
                continue
                
            _, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            
            for i, config in enumerate(self.configs):
                values = transformed_data[metric][config]
                offset = width * i
                rects = ax.bar(x + offset, values, width, 
                              label=f"{metric} ({config})", capsize=5)
                ax.bar_label(rects, padding=3, label_type='center', color='white')
                ax.set_prop_cycle(custom_cycler)
                
            ax.set_ylabel("Value")
            ax.set_title(f"{metric} by Experiment")
            ax.set_xticks(x + width / 2, self.experiments)
            ax.legend(loc='upper left')
            
            filename = f"./images/{metric}.png"
            resulting_files.append(filename)
            plt.savefig(filename)
            plt.close()

        # Manejar plots personalizados
        for metric, evaluator in metrics_to_plot["custom"].items():
            resulting_files.extend(
                evaluator.custom_plot(self.raw_results, "test")
            )

        return resulting_files


    def generate_report(self):
        """
        Generates a PDF report with evaluation graphs.

        If no evaluators were provided it will plot only default evaluators, if you created a new evaluator be sure to add the custom_plot() method.
        """
        os.makedirs("./images", exist_ok=True)
        generated_files = self.plot_metrics()

        c = canvas.Canvas("./evaluation_report.pdf", pagesize=letter)
        _, height = letter
        total_bar = len(generated_files)  # Para la barra de progreso

        with Progress() as progress:
            task = progress.add_task("[cyan]📝 Generating report...", total=total_bar)

            add_page_header("Evaluation Report", c)

            y_pos = height - 150
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_pos, "Evaluation Summary")
            c.setFont("Helvetica", 10)
            y_pos -= 20

            summary_text = [
                f"Total Configurations: {len(self.configs)}",
                f"Total Experiments: {len(self.experiments)}",
                f"Evaluators Used: {len(self.evaluators) if self.evaluators else 0}",
                "Default Metrics: execution_time, error"
            ]
            
            for text in summary_text:
                c.drawString(70, y_pos, f"• {text}")
                y_pos -= 15

            progress.update(task, advance=1)
            c.showPage()

            if self.evaluators:
                add_page_header("Evaluators Description", c)
                y_pos = height - 150

                add_evaluator_descriptions(evaluators=self.evaluators, y_pos=y_pos, c=c)
                c.showPage()
            progress.update(task, advance=1)

            add_page_header("Evaluation Results", c)
            y_pos = height - 30

            # Tamaño y posiciones para una cuadrícula 2x2
            img_width, img_height = 250, 170
            x_positions = [50, 320]
            y_positions = [y_pos - 250, y_pos - 450, y_pos - 630]
            img_idx = 0

            for filename in generated_files:
                x = x_positions[img_idx % 2]
                y = y_positions[img_idx // 2]

                c.drawImage(filename, x, y, width=img_width, height=img_height)
                progress.update(task, advance=1)
                img_idx += 1

                # Cada 4 imágenes (2x2) pasamos a una nueva página
                if img_idx % 6 == 0:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 12)
                    # c.drawString(100, height - 50, f"Evaluación: {evaluator_name} (cont.)")
                    img_idx = 0
  # Nueva página para los evaluadores

        c.setFont("Helvetica", 10)
        y_pos -= 650 
        c.drawString(50, y_pos, "This report was automatically generated by Agent-Bench.")
        y_pos -= 20
        c.drawString(50, y_pos, f"Total evaluation time: {time.strftime('%H:%M:%S')}")
        c.showPage()
        
        progress.update(task, advance=1)
            

        c.save()
        console.print("[bold green]Report created successfully![/bold green] ✅🎉\n")


    def save_experiment_as(self, filename: str, encoding: str = "utf-8"):
        """
        Saves a list of dictionaries in JSON or CSV format based on the file extension.

        Args:
            filename (str): The name of the file with extension (.json, .csv).
            encoding (str): The encoding to be used for the file. Defaults to "utf-8".

        Raises:
            ValueError: If the file extension is not valid.
        """
        if not self.experiments_results:
            raise ValueError("La lista de datos está vacía.")

        file_ext = filename.split(".")[-1].lower()

        if file_ext == "json":
            with open(filename, "w", encoding=encoding) as f:
                json_tricks.dump(
                    self.experiments_results, f, ensure_ascii=False, indent=4
                )

        elif file_ext == "csv":
            keys = self.experiments_results[
                0
            ].keys()  # Obtiene las claves del primer diccionario
            with open(filename, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.experiments_results)

        else:
            raise ValueError("Formato no soportado. Usa .json o .csv.")

        console.print(
            f"[bold white]Experiment saves as {filename}![/bold white] ✅🎉\n"
        )


class GraphEvaluator:
    """
    GraphEvaluator will allow you to build agents for evaluation from a dictionary of configurations, a dataset and a series of evaluators (by default ErrorProbability and ExecutionTime).
    """

    def __init__(
        self,
        configurations: Dict[str, Dict],
        start_node: str,
        evaluators: List[BaseEvaluator],
        results_file: str = None,
    ):
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
        self.relevant_output_keys = set(
            eval.state_key for eval in evaluators if hasattr(eval, "state_key")
        )

        if results_file:
            self.load_results(results_file)

        console.print(
            f"[bold green] GraphEvaluator initialized[/bold green] with {len(configurations)} configurations."
        )


    def build_graph(
        self,
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
        chat_agent = StateGraph(graph_state)

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]🔧 Building graph...", total=len(nodes) + len(edges)
            )

            def create_graph_component(config, eval_config):
                def component_function(state):
                    return (
                        config["function"](state, **eval_config)
                        if eval_config
                        else config["function"](state)
                    )

                return component_function

            for node in nodes:
                node_name = node.__name__
                config = {"function": node}
                chat_agent.add_node(
                    node_name,
                    create_graph_component(config, eval_config.get(node_name, {})),
                )
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
                        target = create_graph_component(
                            {"function": edge[1]}, eval_config.get(edge[1].__name__, {})
                        )
                        chat_agent.add_conditional_edges(edge[0], target)

                elif isinstance(edge, tuple) and len(edge) == 3:
                    target = (
                        edge[1]
                        if isinstance(edge[1], str)
                        else create_graph_component(
                            {"function": edge[1]}, eval_config.get(edge[1].__name__, {})
                        )
                    )
                    chat_agent.add_conditional_edges(edge[0], target, edge[2])
                progress.update(task, advance=1)

        return chat_agent.compile()


    def evaluate(
        self,
        dataset: List[Dict],
        graph_state_class,
        batch_size: int = None,
        default_values: Dict = {},
        experiment_name: str = None,
    ):
        """
        Evaluates the graph with a dataset and evaluators.

        Args:
            dataset (List[Dict]): The dataset to be used for evaluation.
            graph_state_class: The class representing the state of the graph.
            generate_report (bool): Whether to generate a report after evaluation. Defaults to True.
            batch_size (int): The size of batches for evaluation. Defaults to None. If None evaluates example by example, in other case it will parallelize evaluations.
            default_values (Dict): Default values to be used in case of errors. Defaults to {}.

        Returns:
            EvaluationDataset: An object containing the results of the evaluation. It can also save the results in json or generate a pdf report.
        """
        if experiment_name is None:
            experiment_name = str(uuid.uuid4())

        console.print(
            f"\n[bold cyan] Evaluating {len(self.configurations)} configurations[/bold cyan] on dataset of size {len(dataset)}\n"
        )

        total_evaluation = len(dataset) * len(self.configurations)

        with tqdm(
            total=total_evaluation, desc="🔍 Evaluating dataset", unit="example"
        ) as pbar:
            results = defaultdict(list)
            for config_name, config in self.configurations.items():
                nodes = config["nodes"]
                edges = config["edges"]
                eval_config = config.get("eval_config", {})
                self.graph = self.build_graph(
                    nodes, edges, graph_state_class, eval_config
                )

                if batch_size:
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for i in range(0, len(dataset), batch_size):
                            batch = dataset[i : i + batch_size]
                            futures.extend(
                                executor.submit(
                                    self.process_example,
                                    example,
                                    config_name,
                                    default_values,
                                    experiment_name,
                                )
                                for example in batch
                            )

                        console.print(
                                f"\n[bold cyan] Evaluating {len(futures)} batch examples [/bold cyan] on configuration [bold cyan] {config_name} [/bold cyan]\n"
                            )

                        for future in as_completed(futures):
                            result = future.result()
                            for key, value in result.items():
                                results[key].append(value)
                            pbar.update(1)

                else:
                    i = 0
                    for example in dataset:
                        console.print(
                            f"\n[bold cyan] Evaluating example {i} [/bold cyan] on configuration [bold cyan] {config_name} [/bold cyan]\n"
                        )
                        result = self.process_example(
                            example, config_name, default_values, experiment_name
                        )
                        for key, value in result.items():
                            results[key].append(value)
                        pbar.update(1)
                        i += 1

            for key, value in results.items():
                self.results[key].extend(value)

        self._display_results()

        return EvaluationDataset(
            experiments_results=self.results, evaluators=self.evaluators
        )


    def evaluate_with_conversational_config(self, experiment_name: str, graph_state_class, agent_handler: Callable, agent_setup: List[Dict], batch_size: int = None):
        """Use this method to evaluate architectures iteratively (conversationally).

        It will only works if the evaluators are of type BaseConversationalEvaluator and have the evaluate_conversation method implemented."""

        if experiment_name is None:
            experiment_name = str(uuid.uuid4())

        console.print(
            f"\n[bold cyan] Evaluating {len(self.configurations)} configurations[/bold cyan]\n"
        )

        total_evaluation = len(agent_setup)*len(self.configurations)

        with tqdm(
            total=total_evaluation, desc="🔍 Evaluating with conversational agent", unit="setup"
        ) as pbar:
            results = defaultdict(list)
            for config_name, config in self.configurations.items():
                nodes = config["nodes"]
                edges = config["edges"]
                eval_config = config.get("eval_config", {})
                self.graph = self.build_graph(nodes, edges, graph_state_class, eval_config)

                if batch_size:
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for i in range(0, len(agent_setup), batch_size):
                            batch = agent_setup[i : i + batch_size]
                            futures.extend(
                                executor.submit(self.process_setup, setup, agent_handler, config_name, experiment_name) for setup in batch
                            )

                        for future in as_completed(futures):
                            batch_results = future.result()
                            for key, value in batch_results.items():
                                # print(results)
                                results[key].append(value)
                            pbar.update(1)

                else:
                    for setup in agent_setup:
                        setup_results = self.process_setup(setup, agent_handler, config_name, experiment_name)
                        for key, value in setup_results.items():
                            results[key].append(value)
                        pbar.update(1)

            for key, value in results.items():
                self.results[key].extend(value)

        self._display_results()

        return EvaluationDataset(
            experiments_results=self.results, evaluators=self.evaluators
        )


    def load_results(self, filename: str):
        """
        Loads previous results from a JSON file.

        Args:
            filename (str): The name of the file to load results from.
        """
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            console.print(
                f"[bold green]Loaded previous results from {filename}[/bold green]"
            )
        else:
            console.print(
                f"[bold red]No previous results found in {filename}[/bold red]"
            )


    def _display_results(self):
        """ "
        Displays the evaluation results in a table.
        """
        display_results(self.results, console)
      
        
    def process_example(
        self,
        example: Dict,
        config_name: str,
        default_values: Dict = {},
        experiment_name: str = None,
    ):
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
            "model_output": {
                k: model_output.get(k, None) for k in self.relevant_output_keys
            },
            "dataset_output": output_data,
            "execution_time": end_time - start_time,
            "error": error,
            **(
                {"experiment_id": experiment_name}
                if experiment_name is not None
                else {}
            ),
        }

        for eval in self.evaluators:
            eval_results[eval.__class__.__name__] = eval.evaluate(
                model_output, output_data
            )

        return example_results | eval_results


    def process_setup(self, setup: Dict, agent_handler: Callable, config_name: str, experiment_name: str):
        """Process a single agent setup."""
        start_time = time.time()
        start_resources = measure_system_resources()
        try:
            results = agent_handler(self.graph, setup)
            error = False
        except Exception as e:
            results = {}
            error = True
        end_time = time.time()
        end_resources = measure_system_resources()

        setup_results = {
            "config": config_name,
            "execution_time": end_time - start_time,
            "error": error,
            "experiment_id": experiment_name,
            "memory_usage_mb": end_resources["memory_usage_mb"] - start_resources["memory_usage_mb"],
            "cpu_percent": end_resources["cpu_percent"] - start_resources["cpu_percent"],
            "num_threads": end_resources["num_threads"],  # Este es un valor instantáneo
        }

        for eval in self.evaluators:
            eval_results = eval.evaluate(results)
            setup_results[eval.__class__.__name__] = eval_results

        return setup_results
