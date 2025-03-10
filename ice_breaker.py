from dotenv import load_dotenv

# import os
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser

information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his key roles in Tesla, Inc., SpaceX, and Twitter (which he rebranded as X). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk is the wealthiest person in the world; as of March 2025, Forbes estimates his net worth to be US$343 billion.
"""

if __name__ == "__main__":
    load_dotenv()

    # print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    print("Hello Langchain!")

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = ChatOllama(model="mistral")
    llm = ChatVertexAI(
        model="gemini-1.5-pro-001",
        temperature=0.2,  # Adjust temperature for desired randomness
        max_output_tokens=100,  # Set a limit for the response length
    )

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)

    pass
