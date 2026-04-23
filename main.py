from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

#model = OllamaLLM(model="llama3.2")
model = ChatOllama(model="llama3.2")

tempplate = """
You are an expert in answering questions abou a pizza resturant

Here are some relevant reviews: {reviews}

here is the question: {question}

"""

prompt = ChatPromptTemplate.from_template(tempplate)

chain = prompt | model
while True:
    try:
        print("\n\n============================================\n\n")
        user_input = input("Enter your question (OR Q t quit):   ")
        if user_input.lower() == 'q':
            break

        reviews = retriever.invoke(user_input)
        result = chain.invoke({"reviews":reviews,"question":user_input})
        print("\n\n============================================\n\n")
        print("Result: ",result.content)
    except Exception as e:
        print("\n\n============================================\n\n")
        print("Error: ", e)
        print("\n\n============================================\n\n")

