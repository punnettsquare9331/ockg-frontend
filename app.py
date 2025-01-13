import gradio as gr
import requests 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

key = os.getenv("API_KEY")

def fetch_graphql_data(prompt):
    endpoint = "https://okcg-ockg.hypermode.app/graphql"
    query = f"""
    query {{
        executeGeneratedQuery(prompt: "{prompt}") {{
            item1
            item2
        }}  
    }}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    try:
        response = requests.post(endpoint, json={"query": query}, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx and 5xx)
        print("Raw response content:", response.text)  # Debugging line
        return response.json()
    except requests.RequestException as e:
        print("Request failed:", e)
        return {"error": str(e)}


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    search_kg=False,
):
    # Initialize the message context
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    # Handle Knowledge Graph Search
    if search_kg:
        graphql_data = fetch_graphql_data(message)  # Use user input as the prompt

        # Check for errors in GraphQL response
        if "error" in graphql_data:
            yield f"An error occurred while querying the knowledge graph: {graphql_data['error']}"
            return

        # Extract query and result from GraphQL response
        query = graphql_data.get("data", {}).get("executeGeneratedQuery", {}).get("item1", "No query found")
        result = graphql_data.get("data", {}).get("executeGeneratedQuery", {}).get("item2", "No result found")

        # Add query and result to the context for chat completion
        kg_context = (
            f"The following information was retrieved from the knowledge graph:\n\n"
            f"**Query**:\n{query}\n\n**Result**:\n{result}\n\n"
        )
        messages.append({"role": "system", "content": kg_context})

    # Generate response using chat completion
    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

    if kg_context:
        response += kg_context
        yield response
"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a scientific Chatbot used to traverse an Ovarian Cancer Immunology Knowledge Graph", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Checkbox(label="Search Knowledge Graph", value=False),  # New checkbox
    ],
)


if __name__ == "__main__":
    demo.launch()
