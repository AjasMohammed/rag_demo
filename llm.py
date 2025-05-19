from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.messages import BaseMessage

class LLM:
    """
    A class to interact with a language model for generating AI responses based on user queries
    and relevant documents.
    """
    def __init__(self, model, api_key=None, history=False):
        """
        Initialize the LLM with model details, API key, and chat history preference.
        """
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
        self.__history = history
        self.__inital_prompt = [
            ("system", (
                "You are an AI assistant that helps people find information. "
                "Answer the following questions as best you can. "
                "If you don't know the answer just say that you don't know. "
                "Use three sentences maximum. "
                "Keep the answer as short as possible. "
                "Only answer the question. "
            ))
        ]
        self.__chat_history = []

    def __get_prompt_template(self) -> ChatPromptTemplate:
        """
        Generate a chat prompt template based on the initial prompt and chat history.
        """
        if self.__history:
            prompt = self.__inital_prompt.copy()
            prompt.extend(self.__chat_history)
        else:
            prompt = self.__inital_prompt.copy()
        prompt.append(("human", "Relevant Data: {docs}\nQuery: {query}"))
        prompt_template = ChatPromptTemplate.from_messages(prompt)
        return prompt_template

    def invoke(self, query: str, relevant_docs: str) -> BaseMessage:
        """
        Invoke the language model with a query and relevant documents to get a response.
        """
        try:
            prompt = self.__get_prompt_template()
            chain: Runnable = prompt | self.llm
            response: BaseMessage = chain.invoke({"query": query, "docs": relevant_docs})
            self.__chat_history.append(("ai", response.content))
            return response
        except Exception as e:
            print("Error occurred while calling llm!")
            print(e)
