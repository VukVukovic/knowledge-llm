import unittest

from core.prompt import Prompt

class TestPrompt(unittest.TestCase):

    def test_system(self):
        prompt = Prompt("Be a helpful assistant.", input_keys=["question"], output_key="answer")
        self.assertEqual(prompt.get_system_prompt(), "Be a helpful assistant.")
    
    def test_user_prompt(self):
        prompt = Prompt("Be a helpful assistant.", input_keys=["question"], output_key="answer")
        self.assertEqual(prompt.get_user_prompt(), "### question: {question}\n### answer: ")
    
    def test_few_shot(self):
        prompt = Prompt("As a virtual assistant, help the user.", input_keys=["question"], output_key="answer",
                        examples=[
                            {"question": "What is the speed of light?",
                             "answer": "The speed of light is 299 792 458 m/s."}
                        ])
        self.assertEqual(prompt.get_user_prompt(),
                         "Example 1:\n"
                         "### question: What is the speed of light?\n"
                         "### answer: The speed of light is 299 792 458 m/s.\n\n"

                         "Your turn now:\n"
                         "### question: {question}\n### answer: ")
    
    def test_format(self):
        prompt = Prompt("Be helpful to answer user question.", input_keys=["user"], output_key="answer")
        self.assertEqual(prompt.format_user_prompt(user="What is the speed of light?"),
                         "### user: What is the speed of light?\n"
                         "### answer: ")

if __name__ == '__main__':
    unittest.main()