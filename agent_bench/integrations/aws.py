from typing import Literal, Callable
from .base import BaseModelProvider

class AWSProvider(BaseModelProvider):
    @staticmethod
    def get_llm(
        model_id: str, 
        chat_model: Literal["ChatBedrock", "ChatBedrockConverse"], 
        temperature: int = 0
    ) -> Callable:
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