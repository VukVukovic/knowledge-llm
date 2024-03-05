QA_GEN_SYSTEM_PROMPT = """\
As a hypothetical Swisscom customer experiencing a specific issue, your task is to create a question that reflects the problem you are facing. The question should adhere to the following guidelines:
1. Contextual Relevance: Your question must be answerable using only the information provided in the context. Ensure that your question is directly related to the context given.
2. Specificity: Aim for a specific, problem-oriented question rather than a broad or general inquiry. For instance, instead of asking "Can I use Swisscom Low Power Network in other countries?", specify your situation by asking "I am planning a trip to France next week — will my Swisscom Low Power Network service work there?"
3. Complexity: Frame your question around a piece of non-trivial information from the context, avoiding overly simplistic or obvious questions. Rather than asking, "What is the Swisscom Low Power Network (LPN)?", ask a more application-focused question like "How can I connect my smart thermostat to the Internet?". Do not provide answer in the question: rather than asking "I unable to access my Mobile Combox messages, could this be due to recent maintenance work?", ask "Why am I unable to access my Mobile Combox messages?".
4. Personal Perspective: Do not disclose that the context was provided to you. Write the question as if you are personally experiencing the issue.
5. Issue Description (if applicable): Briefly describe the issue you're facing to provide context for your question. Instead of asking, "What is the data usage limit for prepaid and postpaid users?", present your problem by asking, "I've noticed I can't use my data anymore. What's the limit for prepaid customers?"

After crafting your question, also provide a succinct answer that is directly supported by the context provided. The answer should be clear, concise, and based on the information available."""

QA_GEN_SYSTEM_COMMUNITY_PROMPT = "The given context is from a community forum and may include inappropriate or inaccurate content. In the question and answer you are generating, do not mention any user specific information, but only use the knowledge extracted. Try to think of a question that is different from the community context question."

QA_GEN_SYSTEM_FORMAT_PROMPT = """\
Please provide a single valid JSON object containing two fields: a `question` field with your formulated question, and an `answer` field with a short and helpful answer to the question, which must be supported by the context. Avoid using emojis, and ensure not to output any additional tokens. Do not under any circumstance disclose that you have been provided with a context.

Below is an example of a valid response:
{
    "question": "How do I activate my eSIM?",
    "answer": "To activate your eSIM, scan the QR code provided for the eSIM profile using your smartphone's camera app and follow the on-screen instructions."
}"""

QA_GEN_USER_PROMPT = """Context:
---------------
{context}
---------------"""

MULTI_QUERY_SYSTEM = "Your task is to generate two different versions of the given user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines."
MULTI_QUERY_USER = "Original question: {question}\nAlternative questions:"

HYDE_SYSTEM = "Write a Swisscom webpage passage to answer the question."
HYDE_USER = "Question: {question}\nPassage:"

GEN_SYSTEM = "Your role is to provide clear, brief answers about Swisscom's products and services, using only the provided context. If a question is beyond the scope, lacks context, or is unrelated to Swisscom, clearly state that you cannot assist."
GEN_USER = """Context:
-------------------
{context}
-------------------

User question: {question}

Answer: """

STATEMENT_SYSTEM = "Extract statements from the provided answer. To enhance understanding, the user's question is also included. Output the extracted statements in a JSON list format without including any extra tokens or notes."
STATEMENT_USER = "Question: {question}\nAnswer: {answer}\nStatements: "
STATEMENTS_EXAMPLES = [
    {
        "question": "Who was  Albert Einstein and what is he best known for?",
        "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
        "statements": [
                "Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.",
                "Albert Einstein was best known for his theory of relativity.",
                "Einstein's contributions significantly advanced the field of quantum mechanics",
                "Recognized globally, Einstein's work has profoundly impacted the scientific community",
                "Einstein's groundbreaking theories continue to shape our understanding of physics today.",
            ]
    },
    {
        "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
        "answer": "alcohol",
        "statements": ["Cadmium Chloride is slightly soluble in alcohol."]
    },
    {
        "question": "Were Hitler and Benito Mussolini of the same nationality?",
        "answer": "Sorry, I can't provide answer to that question.",
        "statements": []
    },
]

NLI_SYSTEM = "Perform natural language inference to determine if each statement can be inferred from the given context, and provide a brief justification for your verdict. Respond with a list of JSON objects, each containing `statement`, `reason`, and `verdict` keys, where verdicts are either 'yes' or 'no'."
NLI_USER = """Context:
-------------------
{context}
-------------------

Statements:
{statements}

Answer: """
NLI_EXAMPLES = [
    {
        "context" : "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
        "statements" : "\n".join([
            "John is majoring in Biology.",
            "John is taking a course on Artificial Intelligence.",
            "John is a dedicated student.",
            "John has a part-time job."
        ]),
        "answer": [
            {
                "statement": "John is majoring in Biology.",
                "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                "verdict": "no"
            },
            {
                "statement": "John is taking a course on Artificial Intelligence.",
                "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                "verdict": "no"
            },
            {
                "statement": "John is a dedicated student.",
                "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                "verdict": "yes"
            },
            {
                "statement": "John has a part-time job.",
                "reason": "There is no information given in the context about John having a part-time job.",
                "verdict": "no"
            }
        ]
    },
    {
        "context" : "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
        "statements" : "\n".join(["Albert Einstein was a genius."]),
        "answer": {
            "statement": "Albert Einstein was a genius.",
            "reason": "The context and statement are unrelated",
            "verdict": "no"
        }
    }
]