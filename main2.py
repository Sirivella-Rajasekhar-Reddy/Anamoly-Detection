from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import base64
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load environment variables
load_dotenv()

google_api_key=os.getenv("GEMINI_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)


examples = [
    {"system": "Example Analysis: In the uploaded image, an anomaly was detected in the revenue trend where a sudden spike is observed around Q3 2023. This spike might indicate a data entry error or a significant event that needs further investigation."},
    {"system": "Another Example: The image shows a decline in sales data with an unusual dip at the end of the period. This could suggest a potential issue in sales reporting or an actual drop in performance."},
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("ai", "{system}"),
    ]
)

#Define a few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

#Final prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analyzing Power BI reports for anomalies. Analyze the uploaded image for any anomalies, including spikes and other irregular patterns. Provide a detailed summary of any detected anomalies."),
    few_shot_prompt,
    ("human", [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
        },
    ]),
])

app.config['UPLOAD_PATH'] = 'static/uploads'
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        response=""
        if request.method == "POST":
            f = request.files["load_image"]
            print(f.filename)
            filename="temp_file.jpg"
            f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            # Open and read the image file as binary
            with open("static/uploads/"+filename, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            #invoke the model with the prompt and image data
            chain=prompt | llm
            response = chain.invoke({"image_data": image_data}).content
            print(response)
            return jsonify({"status": "success", "response": response, "image_url": f"/static/uploads/temp_file.jpg"})
        return render_template("index.html")
    except Exception as e:
        print("The Error is : ", e)

if __name__ == "__main__":
    app.run(debug=True)
