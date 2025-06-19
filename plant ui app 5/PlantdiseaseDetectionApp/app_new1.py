import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key using the loaded environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel with the specified model name, configuration, and safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Function to read image data from a file object
def read_image_data(file_obj):
    try:
        # If file_obj has a 'read' attribute, it's a file-like object
        if hasattr(file_obj, 'read'):
            file_obj.seek(0)
            return {"mime_type": "image/jpeg", "data": file_obj.read()}
        # If file_obj is a string (file path), open and read the file
        elif isinstance(file_obj, str):
            with open(file_obj, 'rb') as f:
                return {"mime_type": "image/jpeg", "data": f.read()}
        # If file_obj has a 'name' attribute and is a NamedString, treat as path
        elif hasattr(file_obj, 'name'):
            with open(file_obj.name, 'rb') as f:
                return {"mime_type": "image/jpeg", "data": f.read()}
        else:
            raise ValueError("Unsupported file object type for image upload.")
    except Exception as e:
        print(f"Error reading image data: {e}")
        raise

# Function to generate a response based on a prompt and an image file object
def generate_gemini_response(prompt, file_obj):
    try:
        image_data = read_image_data(file_obj)
        response = model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return f"Error: {e}"

# Initial input prompt for the plant pathologist
input_prompt = """
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

**Analysis Guidelines:**

1. **Disease Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

2. **Detailed Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

3. **Next Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

5. **Important Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

**Disclaimer:**
*"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
"""

# Function to process uploaded files and generate a response
def process_uploaded_files(files):
    try:
        file_obj = files[0] if files else None
        if file_obj is not None:
            response = generate_gemini_response(input_prompt, file_obj)
            return file_obj.name, response
        else:
            return None, "No file uploaded."
    except Exception as e:
        print(f"Error in process_uploaded_files: {e}")
        return None, f"Error: {e}"

# Gradio interface setup
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    # Upload button for user to provide images
    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
    )
     # Set up the upload button to trigger the processing function
    upload_button.upload(process_uploaded_files, upload_button, combined_output)

# Launch the Gradio interface with debug mode enabled
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
