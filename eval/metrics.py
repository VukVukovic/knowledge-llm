import json
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from core.prompt_bank import STATEMENTS_EXAMPLES, STATEMENT_SYSTEM, STATEMENT_USER
from core.prompt_bank import NLI_EXAMPLES, NLI_SYSTEM, NLI_USER
from core.prompt_bank import FACTUALITY_EXAMPLES, FACTUALITY_SYSTEM, FACTUALITY_USER

def check_schema(variable, schema):
    # Dict
    if isinstance(schema, dict) and isinstance(variable, dict):
        for key, value_type in schema.items():
            if key not in variable:
                return False
            if not check_schema(variable[key], value_type):
                return False
        return True

    # List
    if isinstance(schema, list) and isinstance(variable, list):
        element_type = schema[0]
        for e in variable:
            if not check_schema(e, element_type):
                return False
        return True

    # Primitive types
    return isinstance(variable, schema)

def get_prompt_template(system_prompt, examples, human_template, output_key):
    # System instruction for the task
    messages = [SystemMessage(system_prompt)]
    human_message_template = HumanMessagePromptTemplate.from_template(human_template)

    # Few-shot examples
    for e in examples:
        messages.extend(human_message_template.format_messages(**e))
        messages.append(AIMessage(json.dumps(e[output_key])))

    # Final instruction
    messages.append(human_message_template)
    return ChatPromptTemplate.from_messages(messages)

class RAGMetric(ABC):
    @abstractmethod
    def compute_score(self, data):
        pass

class Faithfulness(RAGMetric):
    def __init__(self, llm) -> None:
        super().__init__()
        self.llm = llm

    def _extract_statements(self, data):
        statements_prompt = get_prompt_template(STATEMENT_SYSTEM, STATEMENTS_EXAMPLES, STATEMENT_USER, "statements")
        statements_chain = statements_prompt | self.llm | SimpleJsonOutputParser()

        statements = []
        for i, row in tqdm(data.iterrows(), total=len(data), 
                           desc="Extracting statements from answer"):
            try:
                statements_list = statements_chain.invoke({
                    "question" : row["question"],
                    "answer" : row["answer"]
                })
                if type(statements_list) != list:
                    raise Exception("Statements are not a list")
                for s in statements_list:
                    if type(s) != str:
                        raise Exception("Statement list element is not a string")
                statements.append(statements_list)
            except Exception as e:
                statements_list.append([])
                print(f"Skipped statement extraction for {i}")
        return statements
    
    def _perform_nli(self, data, statements):
        nli_prompt = get_prompt_template(NLI_SYSTEM, NLI_EXAMPLES, NLI_USER, "answer")
        nli_chain = nli_prompt | self.llm | SimpleJsonOutputParser()

        scores = []
        for s, (i, row) in tqdm(zip(statements, data.iterrows()), total=len(data), 
                           desc="Performing NLI"):
            so = s
            s = list(filter(lambda t:type(t)==str, s))
            if len(s) == 0:
                print(f"No statements in {i}")
                print(so)
                scores.append(0.0)
                continue
            statements_str = "\n".join(s)
            try:
                verdicts = nli_chain.invoke({
                    "context" : row["context"],
                    "statements" : statements_str
                })
                if type(verdicts) == dict:
                    verdicts = [verdicts]

                if type(verdicts) != list:
                    raise Exception("NLI verdicts is not a list")
                
                correct = 0
                for ver in verdicts:
                    if type(ver) != dict or not "verdict" in ver:
                        raise Exception("There is no verdict key")
                    correct += 1 if ver["verdict"].strip().lower() == "yes" else 0
                scores.append(correct/len(s))
            except Exception as e:
                scores.append(0.0)
                print(f"Skipping NLI for {i}")
                print(e)
                print(nli_chain.invoke({
                    "context" : row["context"],
                    "statements" : statements_str
                }))
        return np.mean(scores)


    def compute_score(self, data):
        statements = self._extract_statements(data)
        return self._perform_nli(data, statements)
    
class AnswerCorrectness(RAGMetric):
    def __init__(self, llm, embeddings) -> None:
        super().__init__()
        self.llm = llm
        self.embeddings = embeddings

    def _factuality(self, data):
        factuality_prompt = get_prompt_template(FACTUALITY_SYSTEM, FACTUALITY_EXAMPLES, 
                                                FACTUALITY_USER, "statements")
        factuality_chain = factuality_prompt | self.llm | SimpleJsonOutputParser()
        print(factuality_prompt)
        factuality_statements = []

        for i, row in tqdm(data.iterrows(), total=len(data), 
                           desc="Classifying factuality statements from answer"):
            try:
                classified_statements = factuality_chain.invoke({
                    "question" : row["question"],
                    "answer" : row["answer"],
                    "ground_truth" : row["ground_truth"]
                })

                if not check_schema(classified_statements, {"TP" : [str], "FP" : [str], "FN" : [str]}):
                    raise Exception("Object does not satisfy schema.")
                
                factuality_statements.append(classified_statements)
            except Exception as e:
                print(f"{i}: ", e)
                factuality_statements.append({"TP" : [], "FP" : [], "FN" : []})

        return factuality_statements

    def compute_score(self, data):
        return 0.0