import gradio as gr
from huggingface_hub import InferenceClient
import requests
import os

# Set the API URL and Authorization Header
OSHWA_API_URL = "https://certificationapi.oshwa.org/api/projects"
OSHWA_API_KEY = os.getenv("OSHWA_API_KEY", "your_default_api_key")  # Replace with your default key or set as an environment variable
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3ODFjNWI5MjY5YmUyMDAxNDFiOTRhZiIsImlhdCI6MTczNjU2MTM2NCwiZXhwIjoxNzQ1MjAxMzY0fQ.t-lAHj3Tordi9nEgvOaJsitFJnrFrla_uvrM1OkhRIc"
}

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Function to fetch projects from OSHWA API
def fetch_projects(api_url, headers, limit=10, offset=0):
    params = {"limit": limit, "offset": offset}
    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        print(f"Error fetching projects: {response.status_code} - {response.text}")
        return []

# Build structured context with project details
def build_context_with_guidance(projects):
    guidance = (
        "Below is a list of open-source hardware projects certified by OSHWA. "
        "Each entry includes the project name and a description of what it does. "
        "Answer questions based only on this information."
    )

    project_details = "\n".join([
        f"Project Name: {project.get('projectName', 'Unnamed Project')}.\n"
        f"Description: {project.get('projectDescription', 'No description provided.')}\n"
        for project in projects
    ])

    return f"{guidance}\n\n{project_details}"

# Fetch OSHWA context for the chatbot
projects = fetch_projects(OSHWA_API_URL, HEADERS)
oshwa_context = build_context_with_guidance(projects) if projects else "No OSHWA project data available."

# Respond function for chatbot
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Add OSHWA project context to the system message
    system_message_with_context = (
        f"{system_message}\n\nHere is the list of OSHWA-certified projects. "
        f"Answer questions based on this list:\n\n{oshwa_context}"
    )

    messages = [{"role": "system", "content": system_message_with_context}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Include user's query
    messages.append({"role": "user", "content": message})

    print("Messages sent to the model:", messages)

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

# Gradio Chat Interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()
