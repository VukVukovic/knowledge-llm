import sys
from pathlib import Path
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from core.langchain_models import CachedModelFactory

context = """If you stay near the Swiss border you mobile phone may automatically switch to a foreign mobile phone network. This can also happen some distance from the border, in the Lake Geneva area or in mountain regions (such as Jura, Alpstein, Grisons or Valais Alps). Since much lower radiation limits apply in Switzerland than in the EU, EU networks extend much further into Switzerland than vice versa."""

creative_params = {
    "temperature": 1.2,
    "top_p": 0.99,
    "max_tokens" : 512
}

precise_params = {
    "temperature": 0.5,
    "top_p": 0.5,
    "max_tokens" : 512
}

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")

precise_model = model_factory.get_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", **precise_params)
creative_model = model_factory.get_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", **creative_params)

prompt_questiongen = ChatPromptTemplate.from_messages([
    SystemMessage("Think of a short question that a user would ask and that can be answered by using the following context. Do not include parts of answer in the question."),
    HumanMessagePromptTemplate.from_template("Context: {context}\nGenerated question: ")
])

'''
precise_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("Extract statements from the provided context. Output the statements as a JSON list of strings."),
    HumanMessagePromptTemplate.from_template("Context: {context}\nExtracted statements: ")
])
'''

chain_1 = prompt_questiongen | creative_model | StrOutputParser()
chain_2 = prompt_questiongen | precise_model | StrOutputParser()
print(chain_1.invoke({"context" : context}))
print(chain_2.invoke({"context" : context}))

