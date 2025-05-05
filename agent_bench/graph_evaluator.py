from langgraph.graph.state import CompiledGraph
from .evaluators.base import BaseEvaluator
from .utils import (
    display_results,
    add_page_header,
    add_evaluator_descriptions,
    plot_paired_data,
    plot_bar_data,
    plot_violin,
    build_graph,
)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import polars as pl
from typing import List, Dict, Tuple, Callable, Any, Optional, Literal
from collections import defaultdict
import time
import os
import json
import json_tricks
import csv
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress
import uuid
import asyncio
import copy

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

    def __init__(
        self,
        experiments_results: Dict,
        evaluation_type: Literal["from_dataset", "from_conversation"] = "from_dataset",
        evaluators: List[BaseEvaluator] = None,
    ):
        self.raw_results = dict(experiments_results)
        self.evaluators = evaluators
        # Procesar datos solo una vez al inicializar
        self.evaluation_type = evaluation_type
        self.experiments = sorted(set(experiments_results["experiment_id"]))
        self.configs = sorted(set(experiments_results["config"]))

    def calculate_metrics(
        self, evaluators, group_by: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Calculate metrics with flexible grouping.

        Args:
            group_by: List of columns to group by. Default ['experiment_id', 'config']
        """
        if group_by is None:
            group_by = ["experiment_id", "config"]

        calculation_metric = {
            "execution_time": pl.mean,
            "error": pl.sum,
        }

        if evaluators:
            calculation_metric.update(
                {
                    evaluator.__class__.__name__: evaluator.aggregation
                    for evaluator in evaluators
                    if evaluator.aggregation is not None
                }
            )

        # Convert to polars DataFrame
        df = pl.DataFrame(self.raw_results, strict=False)

        grouped_in_list = None
        result_list = {}

        grouped_in_list = (
            df.group_by(group_by)
            .agg(**{metric: pl.col(metric) for metric, _ in calculation_metric.items()})
            .sort(group_by)
        )
        result_list = {
            col: grouped_in_list[col].to_list() for col in grouped_in_list.columns
        }

        grouped_agg = (
            df.group_by(group_by)
            .agg(**{metric: agg(metric) for metric, agg in calculation_metric.items()})
            .sort(group_by)
        )

        result_agg = {col: grouped_agg[col].to_list() for col in grouped_agg.columns}

        return result_agg, result_list

    def plot_metrics(self):
        """
        Plot by the metric and experiment the configurations results for the evaluation.

        If cuistom_plot() method was provided in the evaluator it is used instead of the default plot.
        """
        metrics_to_plot = {"default": [], "custom": {}}

        eval_type = self.evaluation_type

        if self.evaluators:
            for evaluator in self.evaluators:
                if evaluator.default_plot:
                    metrics_to_plot["default"].append(evaluator)
                elif hasattr(evaluator, "custom_plot"):
                    metrics_to_plot["custom"][evaluator.__class__.__name__] = evaluator

        agg_data, list_data = self.calculate_metrics(
            metrics_to_plot["default"],
            ["config"] if eval_type == "from_dataset" else ["config", "setup"],
        )

        # print(agg_data)
        resulting_files = []

        configuration_names = list_data.pop("config")
        if self.evaluation_type == "from_conversation":
            agg_data.pop("setup")
            list_data.pop("setup")

        resulting_files.extend(
            [
                plot_violin(metric_name, data, configuration_names)
                for metric_name, data in list_data.items()
            ]
        )

        if self.evaluation_type == "from_dataset":
            resulting_files.extend(
                [
                    plot_paired_data(metric_name, data, configuration_names)
                    for metric_name, data in list_data.items()
                ]
            )

        configuration_names = agg_data.pop("config")
        resulting_files.extend(
            [
                plot_bar_data(metric_name, data, configuration_names)
                for metric_name, data in agg_data.items()
            ]
        )

        # Manejar plots personalizados
        for metric, evaluator in metrics_to_plot["custom"].items():
            resulting_files.extend(evaluator.custom_plot(self.raw_results, metric))

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
                "Default Metrics: execution_time, error, memory_usage, cpu_percent",
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
        c.drawString(
            50, y_pos, "This report was automatically generated by Agent-Bench."
        )
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
        if not self.raw_results:
            raise ValueError("La lista de datos está vacía.")

        file_ext = filename.split(".")[-1].lower()

        if file_ext == "json":
            with open(filename, "w", encoding=encoding) as f:
                json_tricks.dump(self.raw_results, f, ensure_ascii=False, indent=4)

        elif file_ext == "csv":
            keys = self.raw_results[
                0
            ].keys()  # Obtiene las claves del primer diccionario
            with open(filename, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.raw_results)

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
        agents: List[Any],
        agents_names:List[str],
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
        self.agents = agents
        self.agents_names = agents_names
        self.results = defaultdict(list)
        self.evaluators = evaluators
        self.results_file = results_file
        self.relevant_output_keys = set(
            eval.state_key for eval in evaluators if hasattr(eval, "state_key")
        )

        if results_file:
            self.load_results(results_file)

        console.print(
            "[bold green] GraphEvaluator initialized[/bold green] configurations."
        )

    @classmethod
    def from_agent(
        cls,
        agents: List[CompiledGraph],
        agents_names: List[str],
        evaluators: List[BaseEvaluator],
        results_file: str = None,
    ):
        """
        # Description

        Use this method when you are using a compiled agent to measure model performance.

        ## Args:
            - agents (List[CompiledStateGraph]): A list of the compiled agents you are going to evaluate.
            - evaluators (List[BaseEvaluator]): List of the evaluators designed for your experiment.
            - results_file (str): Path of the file with previous results or in wich you decided to save the results of the evaluation. Defaults to None.
        """
        return cls(
            agents=agents,
            agents_names=agents_names,
            evaluators=evaluators,
            results_file=results_file,
        )

    @classmethod
    def from_config(
        cls,
        configurations: Dict[str, Dict],
        graph_state_class: Dict,
        start_node: str,
        evaluators: List[BaseEvaluator],
        results_file: str = None,
    ):
        """
        # Description

        Use this method if you are trying to evaluate a set of agents from a configurations dict. Inside the configurations dict it is important to include:

        1. A "nodes" key with a List[Callable] representing the nodes you are considering in the agents.
        2. An "edges" key with a List[Tuple] representing the connections between your nodes.
        3. "eval_config" key, with the configuration that your nodes will receive as kwargs.

        This method is the most flexible, because you allow a set of nodes and edges to dynamically construct all the configurations you want.

        ## Args:
            - configurations (Dict[str, Dict]) The configurations dict described below
            - graph_state_class (Dict): Dict representing the state of your graph.
            - start_node (str): Name of the first node you are considering to constuct your graph.
            - evaluators (List[BaseEvaluator]): List of the evaluators designed for your experiment.
            - results_file (str): Path of the file with previous results or in wich you decided to save the results of the evaluation. Defaults to None.
        """
        agents = [
            build_graph(
                start_node,
                config["nodes"],
                config["edges"],
                graph_state_class,
                config.get("eval_config", {}),
            )
            for config in configurations.values()
        ]

        agents_names = [
            config_names for config_names in configurations.keys()
        ]

        return cls(
            agents=agents,
            agents_names=agents_names,
            evaluators=evaluators,
            results_file=results_file,
        )


    def evaluate_from_dataset(
        self,
        dataset: List[Dict],
        batch_size: int = 10,
        default_values: Dict = {},
        keep_keys: bool = False,
        experiment_name: str = None,
    ):
        """
        Evaluates the graph with a dataset and evaluators.

        ## Args:
            - dataset (List[Dict]): The dataset to be used for evaluation.
            - graph_state_class: The class representing the state of the graph.
            - generate_report (bool): Whether to generate a report after evaluation. Defaults to True.
            - batch_size (int): The size of batches for evaluation. Defaults to 10.
            - default_values (Dict): Default values to be used in case of errors. Defaults to {}.

        ## Returns:
            - EvaluationDataset: An object containing the results of the evaluation. It can also save the results in json or generate a pdf report.
        """

        experiment_name = (
            str(uuid.uuid4()) if experiment_name is None else experiment_name
        )

        console.print(
            f"[bold cyan] Evaluating {len(self.agents)} configurations[/bold cyan] with {len(dataset)} examples!"
        )

        total_evaluation = len(dataset) * len(self.agents)

        with tqdm(
            total=total_evaluation, desc="🔍 Evaluating dataset", unit="example"
        ) as pbar:
            results = defaultdict(list)
            for idx, agent in enumerate(self.agents):

                agent_results = asyncio.run(
                    self._evaluate_dataset_examples(
                        dataset,
                        agent,
                        idx,
                        batch_size,
                        pbar,
                        default_values,
                        keep_keys,
                        experiment_name,
                    )
                )

                for key, value in agent_results.items():
                    results[key].extend(value)

            self.results = results

        self._display_results()

        return EvaluationDataset(
            experiments_results=self.results,
            evaluation_type="from_dataset",
            evaluators=self.evaluators,
        )


    async def _evaluate_dataset_examples(
        self,
        inputs: List[Dict],
        agent: CompiledGraph,
        idx: int,
        batch_size: int,
        pbar: tqdm,
        default_values: Dict = {},
        keep_keys: bool = False,
        experiment_name: str = None,
    ):
        """
        Process asynchronously the dataset examples by batch size.
        """
        results = defaultdict(list)

        for i in range(0, len(inputs), batch_size):
            input_batch = inputs[i : i + batch_size]
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._process_example(
                            input, agent, idx, default_values, keep_keys, experiment_name
                        )
                    )
                    for input in input_batch
                ]

            for completed_task in tasks:
                result = completed_task.result()
                for key, value in result.items():
                    results[key].append(value)
                pbar.update(1)

        return results


    async def _process_example(
        self,
        example: Dict,
        agent: CompiledGraph,
        idx: int,
        default_values: Dict = {},
        keep_keys: bool = False,
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
            model_output = await agent.ainvoke(copy.deepcopy(input_data))
            error = 0
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")  # Usando __str__()
            error = 1
            model_output = default_values
        end_time = time.time()

        if not keep_keys:
            model_output = {
                k: model_output.get(k, None) for k in self.relevant_output_keys
            }

        example_results = {
            "config": self.agents_names[idx],
            "dataset_input": input_data,
            "model_output": model_output,
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


    def evaluate_with_conversational_config(
        self,
        experiment_name: str,
        agent_handler: Callable,
        agent_setup: List[Dict],
        conversation_size: int = 6,
        repeat_configuration: int = 2,
    ):
        """Use this method to evaluate architectures iteratively (conversationally).

        It will only works if the evaluators are of type BaseConversationalEvaluator and have the evaluate_conversation method implemented.
        """

        experiment_name = (
            str(uuid.uuid4()) if experiment_name is None else experiment_name
        )

        console.print(
            f"[bold cyan] Evaluating {len(self.agents)} configurations[/bold cyan] with {len(agent_setup)} agent setups!"
        )

        total_evaluation = len(agent_setup) * len(self.agents) * repeat_configuration

        with tqdm(
            total=total_evaluation, desc="🔍 Evaluating dataset", unit="example"
        ) as pbar:
            results = defaultdict(list)
            for agent in self.agents:
                agent_results = asyncio.run(
                    self._evaluate_setup(
                        agent,
                        agent_setup,
                        agent_handler,
                        conversation_size,
                        repeat_configuration,
                        experiment_name,
                        pbar
                    )
                )

                for key, value in agent_results.items():
                    results[key].extend(value)

            for key, value in results.items():
                self.results[key].extend(value)

        self._display_results()

        return EvaluationDataset(
            experiments_results=self.results,
            evaluation_type="from_conversation",
            evaluators=self.evaluators,
        )


    async def _evaluate_setup(
        self,
        agent: CompiledGraph,
        agent_setup: List[Dict],
        agent_handler: Callable,
        conversation_size: int,
        repeat_configuration: int,
        experiment_name: str,
        pbar: tqdm
    ):
        results = defaultdict(list)
        for setup in agent_setup:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._process_setup(
                            setup,
                            agent_handler,
                            agent,
                            conversation_size,
                            experiment_name,
                        )
                    )
                    for _ in range(repeat_configuration)
                ]

            for completed_task in tasks:
                result = completed_task.result()
                for key, value in result.items():
                    results[key].append(value)
                pbar.update(1)
        
        return results


    async def _process_setup(
        self,
        setup: Dict,
        agent_handler: Callable,
        agent: CompiledGraph,
        conversation_size: int,
        experiment_name: str,
    ):
        """Process a single agent setup."""
        start_time = time.time()
        try:
            results = await agent_handler(agent, copy.deepcopy(setup), conversation_size)
            error = 0
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")  # Usando __str__()
            results = {"messages":[]}
            error = 1
        end_time = time.time()

        setup_results = {
            "config": agent.__name__,
            "setup": setup.get("name", ""),
            "execution_time": end_time - start_time,
            "error": error,
            "experiment_id": experiment_name,
            "model_output": results,
        }

        for eval in self.evaluators:
            eval_results = eval.evaluate(results)
            setup_results[eval.__class__.__name__] = eval_results

        return setup_results


    def _load_results(self, filename: str):
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
