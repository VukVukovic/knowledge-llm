import pickle
import os
from pathlib import Path

from core.model import ModelClient, BaseModelClient

class CachedModelClient(BaseModelClient):
	def __init__(self, model_client : ModelClient, cache_path : Path):
		self.model_client = model_client
		self.cache = {}
		
		self.cache_path = cache_path
		if os.path.isfile(self.cache_path):
			with open(self.cache_path, "rb") as f:
				self.cache = pickle.load(f)

	def get_embedding(self, texts : list[str], model : str):
		# Get embeddings not present in cache
		texts_to_get = list(filter(lambda text : not model + text in self.cache, texts))
		if len(texts_to_get) > 0:
			embeddings = self.model_client.get_embedding(texts_to_get, model)
			for t, e in zip(texts_to_get, embeddings):
				self.cache[model + t] = e
			self.flush()

		return [self.cache[model + text] for text in texts]

	def get_completion(self, system_prompt : str, user_prompt : str, model : str, **kwargs):
		kwargs_key = ", ".join(f"{key}={value}" for key, value in kwargs.items())
		cache_key = model + kwargs_key + system_prompt + user_prompt

		if cache_key in self.cache:
			return self.cache[cache_key]
		
		completion = self.model_client.get_completion(system_prompt, user_prompt, model)
		self.cache[cache_key] = completion
		self.flush()
		return completion
	
	def flush(self):
		with open(self.cache_path, "wb") as f:
			pickle.dump(self.cache, f)