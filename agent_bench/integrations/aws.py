from typing import Literal, Callable, Dict, Tuple, Any
from .base import BaseModelProvider
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, AIMessage


class AWSProvider(BaseModelProvider):
    @staticmethod
    def get_llm(
        model_id: str,
        chat_model: Literal["ChatBedrock", "ChatBedrockConverse"],
        temperature: int = 0,
    ) -> Callable:
        """
        Use this function to enable llm model and temperature be configurable.
        """
        try:
            from langchain_aws import ChatBedrock, ChatBedrockConverse
        except ImportError:
            raise ImportError(
                "Para usar AWSProvider necesitas instalar langchain-aws. "
                "Puedes instalarlo con: pip install agent_bench[aws]"
            )

        if chat_model == "ChatBedrock":
            return ChatBedrock(
                model_id=model_id,
                model_kwargs=dict(temperature=temperature),
                region="us-east-1",
                max_tokens=500,
            )
        elif chat_model == "ChatBedrockConverse":
            return ChatBedrockConverse(
                model=model_id,
                temperature=temperature,
                max_tokens=500,
            )

    @staticmethod
    def bedrock_agent_handler(graph: CompiledGraph, setup: Dict):
        """
        Use this agent_handler for evaluate_with_conversational_agent.

        Args:
            graph: Compiled LangGraph StateGraph (it will be passed during execution)
            setup: Dict, It has to contain number of iterations per conversation, setup for the first version of state,
                        preserving_keys of the state during executions and finally the conversational_agent that as an output should have a list.

        Consider that it only works on graphs that implement messages as a List in the state key "messages".

        """
        try:
            from langchain_aws import ChatBedrock, ChatBedrockConverse
        except ImportError:
            raise ImportError(
                "Para usar AWSProvider necesitas instalar langchain-aws. "
                "Puedes instalarlo con: pip install agent_bench[aws]"
            )

        max_iterations = setup.get("iterations", 2)
        invoke_state = setup.get("invoke_state", {})
        preserving_keys = setup.get("preserving_keys", [])
        conversational_agent = setup.get("conversational_agent")
        agent_messages = []
        working_state = invoke_state.copy()

        for _ in range(max_iterations):
            pre_actualized_state = graph.invoke(working_state)

            agent_messages.append(
                HumanMessage(content=pre_actualized_state["messages"][-1].content)
            )

            agent_response = conversational_agent(agent_messages)
            pre_actualized_state["messages"].append(
                HumanMessage(content=agent_response[-1].content)
            )

            for key in preserving_keys:
                working_state[key] = pre_actualized_state[key]

        return working_state
