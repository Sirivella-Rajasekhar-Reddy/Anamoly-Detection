from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import base64
import os, json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load environment variables
load_dotenv()

google_api_key=os.getenv("GEMINI_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)

# {"system": "Example Analysis: In the uploaded image, an anomaly was detected in the revenue trend where a sudden spike is observed around Q3 2023. This spike might indicate a data entry error or a significant event that needs further investigation."},
#     {"system": "Another Example: The image shows a decline in sales data with an unusual dip at the end of the period. This could suggest a potential issue in sales reporting or an actual drop in performance."},

examples = [{
    "system": """[{
                "analysis" : "The provided image shows a Power BI report on key leave patterns. The report presents data on sick leave per FTE, monthly annual leave, and a breakdown of leave by department and employee.",
                "anomalies" : [{
                                    "location" : "The anomaly is located in the *Over Time Analysis (Sick Leave per FTE)* section of the report",
                                    "description" : "No anomalies: This chart appears to show a stable trend with no significant spikes or dips.",
                                },
                               {
                                    "location" : "The anomaly is located in the *Over Time Analysis (Monthly Annual Leave)* section of the report, specifically the *Year Over Year Growth (Red Line)* line",
                                    "description" : "The red line representing "Year Over Year Growth" shows a sharp decline from July to August 2017. This is a significant drop and could indicate a potential issue with employee sick leave patterns during that period. Further investigation is needed to understand the reasons behind this sudden decline.",
                                }]
                "additional-observation" : "* The report shows a relatively consistent number of days of sick leave per FTE from 2015 to 2017.
                                            * The "Days Annual Leave" (blue bars) indicate a peak in January 2017, suggesting a possible seasonal pattern.
                                            * The breakdown of leave by department and employee suggests that Sales has the highest percentage of leave days.",
                "recommendations" :   " * Investigate the reasons behind the sharp decline in "Year Over Year Growth" in July/August 2017. This could involve analyzing employee data for that period, checking for any changes in company policies, or identifying any external factors that might have impacted leave patterns.
                                        * Further analyze the seasonal pattern in "Days Annual Leave" to understand potential drivers and plan accordingly.
                                        * Explore the reasons behind the higher percentage of leave days in the Sales department. This could be related to factors like workload, stress levels, or specific industry trends.",
                "conclusion" : "By investigating these anomalies and observations, the company can gain a deeper understanding of its leave patterns and identify areas for improvement."
            }]"""
    }
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
    ("system", "You are an expert at analyzing Power BI reports for anomalies. Analyze the uploaded image for any anomalies, including spikes and other irregular patterns. Provide a location and detailed summary of any detected anomalies and the response is in the form of json."),
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
            resp = chain.invoke({"image_data": image_data}).content
            # print(resp)
            start_index=resp.index("[")
            end_index=resp.rindex("]")
            json_array=resp[start_index:end_index+1]
            json_load=json.loads(json_array)
            response=pd.DataFrame(json_load).to_dict()
            print(response)


            return jsonify({"status": "success", "response": response, "image_url": f"/static/uploads/temp_file.jpg"})
        return render_template("index.html")
    except Exception as e:
        print("The Error is : ", e)

if __name__ == "__main__":
    app.run(debug=True)
