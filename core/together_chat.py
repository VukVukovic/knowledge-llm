from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from typing import Optional, Dict, List, Any
from langchain_core.outputs import ChatGeneration, ChatResult
import requests

TOGETHER_MESSAGE_MAPPING = {
    "system" : "system",
    "human" : "user",
    "ai" : "assistant"
}

class ChatTogether(BaseChatModel):
    base_url: str = "https://api.together.xyz/v1/chat/completions"
    together_api_key: SecretStr
    model: str

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Dict] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["together_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        )
        return values
    
    @property
    def _llm_type(self) -> str:
        return "together-chat"
    
    @property
    def _model_params(self) -> Dict[str, Any]:
        return {
            "temperature" : self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format
        }
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{"model_name": self.model}, **self._model_params}
    
    @staticmethod
    def _message_to_dict(message: BaseMessage):
        type_str = message.type

        if type_str not in TOGETHER_MESSAGE_MAPPING:
            raise ValueError("Message type {type_str} not supported.")
    
        return {"role" : TOGETHER_MESSAGE_MAPPING[type_str], "content" : message.content}
    
    @staticmethod
    def _to_chat_result(response: Dict) -> ChatResult:
        chat_generations = []

        for g in response["choices"]:
            chat_generation = ChatGeneration(
                message=
                    AIMessage(content=g["message"]["content"]), 
                    generation_info={"finish_reason": g["finish_reason"]}
                )
            chat_generations.append(chat_generation)

        return ChatResult(generations=chat_generations)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.together_api_key.get_secret_value()}"
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop

        payload: Dict[str, Any] = {
            "model" : self.model,
            **self._model_params,
            "messages": list(map(self._message_to_dict, messages)),
            "stop": stop_to_use,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(url=self.base_url, json=payload, headers=headers)

        if response.status_code >= 500:
            raise Exception(f"Together Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"Together received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"Together returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()
        return self._to_chat_result(data)