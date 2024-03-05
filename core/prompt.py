class Prompt:
	def __init__(self, instruction: str, input_keys: list[str], output_key: str, examples: list[dict] = None, output_type: str = "json"):
		self.instruction = instruction
		self.input_keys = input_keys
		self.output_key = output_key
		self.examples = examples
		self.output_type = output_type

	def get_system_prompt(self) -> str:
		return self.instruction
	
	def get_user_prompt(self) -> str:
		prompt_template = ""

		if self.examples:
			for i, example in enumerate(self.examples):
				prompt_template += f"Example {i+1}:\n"
				for key, value in example.items():
					prompt_template += f"### {key}: {value}\n"
				prompt_template += "\n"
			prompt_template += "Your turn now:\n"

		if self.input_keys:
			for input_key in self.input_keys:
				prompt_template += f"### {input_key}: {{{input_key}}}\n"
		
		if self.output_key:
			prompt_template += f"### {self.output_key}: "

		return prompt_template
	
	def format_user_prompt(self, **kwargs):
		if set(self.input_keys) != set(kwargs.keys()):
			raise ValueError(
                f"Input variables {self.input_keys} do not match with the given parameters {list(kwargs.keys())}"
            )
		return self.get_user_prompt().format(**kwargs)