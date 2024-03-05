from openai import OpenAI
from transformers import AutoTokenizer

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"

class BaseModelClient:
    def get_embedding(self, texts : list[str], model : str):
        pass

    def get_completion(self, system_prompt : str, user_prompt : str, model : str, **kwargs):
        pass

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
EMBEDDING_BATCH_SIZE = 100

class ModelClient(BaseModelClient):
    # map with model -> client
    def __init__(self, openai_key = None, togetherai_key = None, anyscale_key = None):
        self.model_clients = {}

        if openai_key:
            self.client_openai = OpenAI()

            self.model_clients.update({
                "gpt-4-0125-preview" : self.client_openai,
                "gpt-3.5-turbo-0125" : self.client_openai,
                "text-embedding-ada-002" : self.client_openai,
                "text-embedding-3-large" : self.client_openai,
                "text-embedding-3-small" : self.client_openai
            })

        if anyscale_key:
            self.client_anyscale = OpenAI(
                api_key = anyscale_key,
                base_url=ANYSCALE_BASE_URL
            )

            self.model_clients.update(dict.fromkeys([
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                "meta-llama/Llama-2-70b-chat-hf",
                "BAAI/bge-large-en-v1.5",
                "thenlper/gte-large"],
                self.client_anyscale))
        
        if togetherai_key:
            self.client_togetherai = OpenAI(
                api_key=togetherai_key,
                base_url=TOGETHER_BASE_URL
            )

            self.model_clients.update(dict.fromkeys([
                "mistralai/Mistral-7B-Instruct-v0.2",
                "zero-one-ai/Yi-34B-Chat",
                "sentence-transformers/msmarco-bert-base-dot-v5",
                "togethercomputer/m2-bert-80M-2k-retrieval"
            ], self.client_togetherai))

            if not anyscale_key:
                self.model_clients.update(dict.fromkeys([
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "BAAI/bge-large-en-v1.5",
                ], 
                self.client_togetherai))

        TRUNCATE_MODELS = ["BAAI/bge-large-en-v1.5", "thenlper/gte-large",
                           "togethercomputer/m2-bert-80M-2k-retrieval",
                           "sentence-transformers/msmarco-bert-base-dot-v5"]
        
        self.truncation_tokenizers = {}
        for model in TRUNCATE_MODELS:
            if model in self.model_clients:
                self.truncation_tokenizers[model] = AutoTokenizer.from_pretrained(model)
    
    def _truncate(self, texts: list[str], model:str):
        tokenizer = self.truncation_tokenizers[model]
        encoded_input = tokenizer(texts, padding=False, truncation=True)
        return tokenizer.batch_decode(encoded_input["input_ids"], skip_special_tokens=True)

    def get_embedding(self, texts : list[str], model : str):
        if model not in self.model_clients:
            raise ValueError(f"Requested embedding model `{model}` is not supported.")

        client = self.model_clients[model]

        embeddings_results = []
        for b in batch(texts, EMBEDDING_BATCH_SIZE):
            if model in self.truncation_tokenizers:
                b = self._truncate(b, model)
            response = client.embeddings.create(input = b, model=model)
            embeddings_results.extend([d.embedding for d in response.data])
        return embeddings_results
    
    def get_completion(self, system_prompt : str, user_prompt : str, model : str, **kwargs):
        if model not in self.model_clients:
            raise ValueError(f"Requested language model `{model}` is not supported.")
        
        client = self.model_clients[model]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )

        return response.choices[0].message.content