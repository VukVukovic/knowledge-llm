import unittest
import os
import tempfile

from unittest.mock import patch
from core.cached_model import CachedModelClient, ModelClient
from .model_test import create_chat_completion, create_embedding_response

class TestCachedModel(unittest.TestCase):
    @patch("openai.resources.chat.Completions.create")
    def test_completion_api_and_cache(self, openai_create):
        model_name = "gpt-4-0125-preview"
        temp_dir = tempfile.TemporaryDirectory()

        client = ModelClient(openai_key="ski-openaitest")
        cache_client = CachedModelClient(client, cache_path=os.path.join(temp_dir.name, "cache.pickle"))

        expected_response = "Response 1"
        openai_create.return_value = create_chat_completion(expected_response, model_name)

        # Caching is happening
        response1 = cache_client.get_completion(system_prompt="Be nice.", 
                                                user_prompt="What is the speed of light?",
                                                model=model_name)
        assert response1 == expected_response

        # API is now returning different response
        uexpected_response = "Response 2"
        openai_create.return_value = create_chat_completion(uexpected_response, model_name)

        # We are getting the first response (from cache)
        assert expected_response == cache_client.get_completion(system_prompt="Be nice.", 
                                                user_prompt="What is the speed of light?",
                                                model=model_name)
        # API is returning new response
        assert uexpected_response == client.get_completion(system_prompt="Be nice.", 
                                                user_prompt="What is the speed of light?",
                                                model=model_name)

        temp_dir.cleanup()

    @patch("openai.resources.embeddings.Embeddings.create")
    def test_embedding(self, openai_create):
        model_name = "text-embedding-ada-002"
        temp_dir = tempfile.TemporaryDirectory()

        client = ModelClient(openai_key="ski-openaitest")
        cache_client = CachedModelClient(client, cache_path=os.path.join(temp_dir.name, "cache.pickle"))

        # Caching is happening
        expected_embeddings = [[0.1] * 512]
        openai_create.return_value = create_embedding_response(model_name, expected_embeddings)
        assert expected_embeddings == cache_client.get_embedding(["Test"], model_name)

        # Cached client should return from cache, direct client returns new embeddings
        unexpected_embeddings = [[0.2] * 512]
        openai_create.return_value = create_embedding_response(model_name, unexpected_embeddings)
        assert expected_embeddings == cache_client.get_embedding(["Test"], model_name)
        assert unexpected_embeddings == client.get_embedding(["Test"], model_name)

        temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()