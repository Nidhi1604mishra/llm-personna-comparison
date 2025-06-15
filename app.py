from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import pandas as pd

load_dotenv()

QUERIES = {
    "Customer Support": [
        "My laptop is overheating. What should I do?",
        "I can't connect to Wi-Fi. Can you help?",
        "My phone screen is cracked. What are my options?"
    ],
    "Technical Expert": [
        "How do I reset my router?",
        "What’s the difference between RAM and ROM?",
        "How can I improve my laptop’s performance?"
    ],
    "Creative Writer": [
        "Write a short story about a robot discovering emotions.",
        "Describe a sunset using metaphors.",
        "Create a poem about the ocean."
    ]
}    


PERSONAS = {
    "Customer Support": """
    You are a friendly customer support agent for a tech company. 
    Respond politely, avoid jargon, and never blame the user.
    """,
    "Technical Expert": """
    You are an IT specialist. Provide step-by-step solutions to technical problems. 
    Use simple language and assume the user is a beginner.
    """,
    "Creative Writer": """
    You are a creative writer. Respond with imaginative and engaging content. 
    Use metaphors and storytelling to explain concepts.
    """
}


MODELS = [
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-saba",
    "cohere/command-r",
    "google/gemini-pro",
    "deepseek/deepseek-r1-zero:free"
]

def test_model(model_name, persona, query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", PERSONAS[persona]),
        ("human", "{query}")
    ])
    
    llm = ChatOpenAI(
        model = model_name, 
        temperature= 0.5,
        openai_api_key=os.getenv("Open_api_key"),
        base_url="https://openrouter.ai/api/v1")

    chain = prompt | llm
    
    start_time = time.time()
    response = chain.invoke({"query": query})
    end_time = time.time()
    
    time_taken = end_time - start_time
    
    return response.content, time_taken

def generate_response(persona, query, model):
    response, time_taken = test_model(model, persona, query)
    return f"Response: {response}\n\nTime Taken: {time_taken:.2f} seconds"


with gr.Blocks() as demo:
    gr.Markdown("# Persona and Model Comparison Demo")
    with gr.Row():
        persona = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            label="Select Persona",
            value="Customer Support"
        )
        model = gr.Dropdown(
            choices=MODELS,
            label="Select Model",
            value="openai/gpt-3.5-turbo"
        )
    query = gr.Textbox(label="Enter Your Query", lines=3, placeholder="Type your query here...")
    output = gr.Textbox(label="Response", lines=10)
    submit_button = gr.Button("Generate Response")

    # Link the button to the function
    submit_button.click(
        generate_response,
        inputs=[persona, query, model],
        outputs=output
    )

# Launch the Gradio app
demo.launch(share = True)






results = []

for persona, queries in QUERIES.items():
    #print(f"=== Testing Persona: {persona} ===")
    for query in queries:
        #print(f"Query: {query}")
        for model in MODELS:
            #print(f"=== Testing {model} ===")
            response, time_taken = test_model(model, persona, query)
            #print(f"Response: {response}")
            #print(f"Time taken: {time_taken:.2f} seconds")
            #print("\n" + "=" * 50 + "\n")   
             
       
            results.append({
                            "Persona": persona,
                            "Query": query,
                            "Model": model,
                            "Response": response,
                            "Time Taken": time_taken
                        })
    
            
