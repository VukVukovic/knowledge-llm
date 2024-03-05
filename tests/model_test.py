import unittest
import datetime

from core.model import ModelClient

from unittest.mock import patch
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

def create_chat_completion(response: str, model: str) -> ChatCompletion:
    return ChatCompletion(
        id="foo",
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=response,
                    role="assistant",
                ),
            )
        ],
        created=int(datetime.datetime.now().timestamp()),
    )

def create_embedding_response(model: str, embeddings: [float]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[
            Embedding(embedding=e, index=i, object="embedding")
            for i, e in enumerate(embeddings)
        ],
        model=model,
        object="list",
        usage=Usage(prompt_tokens=len(embeddings), total_tokens=len(embeddings))
    )

class TestCachedModel(unittest.TestCase):
    @patch("openai.resources.chat.Completions.create")
    def test_completion(self, openai_create):
        expected_response = "Mock response!"
        model_name = "gpt-4-0125-preview"
        openai_create.return_value = create_chat_completion(expected_response, model_name)

        client = ModelClient(openai_key="ski-openaitest")
        response = client.get_completion(system_prompt="Be nice.", 
                                         user_prompt="What is the speed of light?",
                                         model=model_name)

        assert response == expected_response

    @patch("openai.resources.chat.Completions.create")
    def test_completion_params(self, openai_create):
        expected_response = '{{"response" : "Hello!"}}'
        model_name = "gpt-4-0125-preview"
        openai_create.return_value = create_chat_completion(expected_response, model_name)

        client = ModelClient(openai_key="ski-openaitest")
        response = client.get_completion(system_prompt="Give answer as JSON.", 
                                         user_prompt="Hello!",
                                         model=model_name,
                                         temperature=0.7, top_p=0.1, top_k=40,
                                         max_tokens=100, response_format = {"type": "json_object"})

        assert response == expected_response

    @patch("openai.resources.chat.Completions.create")
    def test_no_key_completion(self, openai_create):
        expected_response = "Mock response!"
        model_name = "gpt-4-0125-preview"
        openai_create.return_value = create_chat_completion(expected_response, model_name)

        client = ModelClient()
        with self.assertRaises(ValueError):
            client.get_completion(system_prompt="Be nice.", 
                                  user_prompt="What is the speed of light?",
                                  model=model_name)


    @patch("openai.resources.chat.Completions.create")
    def test_unsupported_model_completion(self, openai_create):
        expected_response = "Mock response!"
        model_name = "gpt-unknown"
        openai_create.return_value = create_chat_completion(expected_response, model_name)

        client = ModelClient()
        with self.assertRaises(ValueError):
            client.get_completion(system_prompt="Be nice.", 
                                  user_prompt="What is the speed of light?",
                                  model=model_name)

    @patch("openai.resources.embeddings.Embeddings.create")
    def test_embedding(self, openai_create):
        expected_embeddings = [[0.01]*512, [0.02]*512]
        model_name = "text-embedding-ada-002"

        openai_create.return_value = create_embedding_response(model_name, expected_embeddings)

        client = ModelClient(openai_key="ski-openaitest")
        response = client.get_embedding(["Hello"], model=model_name)

        assert response == expected_embeddings

    @patch("openai.resources.embeddings.Embeddings.create")
    def test_unsupported_embedding(self, openai_create):
        expected_embeddings = [[0.01]*512, [0.02]*512]
        model_name = "text-embedding-test" 

        openai_create.return_value = create_embedding_response(model_name, expected_embeddings)

        client = ModelClient(openai_key="ski-openaitest")
        with self.assertRaises(ValueError):
            client.get_embedding(["Hello"], model=model_name)

if __name__ == '__main__':
    unittest.main()