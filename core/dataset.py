import json
import random
from pathlib import Path

class SwisscomDataset:
	def __init__(self, dataset_path: Path):
		with open(dataset_path) as f:
			self.swisscom_dataset = json.load(f)
		
		self.id2entry = {d["metadata"]["id"] : d for d in self.swisscom_dataset}
	
	def get_eval_sample(self, size: int = 350, seed : int = 53):
		random.seed(seed)
		return random.sample(self.swisscom_dataset, size)
	
	def get_by_id(self, id):
		return self.id2entry[id]

