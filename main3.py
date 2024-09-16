from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
import base64
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

# Load environment variables
load_dotenv()

google_api_key=os.getenv("GEMINI_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)

examples = [
    {"system": """[
                    {'analysis': 'The Power BI report displays stacked bar charts showing quantity and sales by product category and region. The chart uses four colors to represent the regions: Central, East, South, and West.', 
                    'anomalies': [{'location': 'Office Supplies - West', 'description': 'The West region shows a significant spike in sales for Office Supplies compared to the other regions. This could be due to a large order or a promotion specific to the West region.'}, 
                                {'location': 'Technology - West', 'description': 'The West region also has the highest sales for Technology products, but the difference compared to other regions is not as pronounced as with Office Supplies.'}, 
                                {'location': 'Furniture - Central', 'description': 'The Central region has a surprisingly low quantity for Furniture compared to the other regions. This could indicate a potential supply chain issue or a lack of demand for Furniture in the Central region.'}], 
                    'additional-observation': 'The report shows a general trend of higher sales in the West region for all product categories. This could indicate a stronger market presence or more successful marketing efforts in the West.', 
                    'recommendations': 'Investigate the reasons behind the anomalies observed in the report. This could involve analyzing sales data, customer demographics, and marketing campaigns to identify potential factors contributing to the discrepancies. Consider adjusting strategies or addressing supply chain issues to optimize sales performance in each region.', 
                    'conclusion': 'By analyzing the anomalies and observing the trends in the report, the company can identify areas for improvement and make data-driven decisions to enhance its overall sales performance.'
                    }
                ]"""},
    {"system": """[
                    {'analysis': 'The Power BI report visualizes data related to employee sick leave patterns. It includes a line chart showing the trend of sick leave per FTE over time, a bar chart representing monthly annual leave, and a breakdown of sick leave days by department and employee.', 
                    'anomalies': [{'location': 'Year-over-Year Growth in Monthly Annual Leave', 'description': 'There is a significant spike in the Year-over-Year Growth of Monthly Annual Leave in July 2017. This suggests a sudden increase in sick leave days compared to the previous year. It could be due to an unexpected event or a change in company policy.'}, 
                                {'location': 'Sick Leave per FTE in 2017', 'description': 'The Sick Leave per FTE in 2017 is slightly lower than in 2016. This could indicate a potential improvement in employee health or a change in workplace environment.'}, 
                                {'location': 'Department Breakdown of Sick Leave Days', 'description': 'The Sales department has a significantly higher percentage of sick leave days compared to other departments (27.2%). This could be due to factors like higher workload, stress, or a higher proportion of employees susceptible to illness.'}, 
                                {'location': 'Employee Sick Leave Taken by FY', 'description': 'Glad is the employee with the highest number of sick leave days (25) in the given fiscal year. This could be a cause for concern and requires further investigation to understand the reasons behind this high number.'}], 
                    'additional-observation': 'The report shows a general trend of decreasing sick leave per FTE from 2015 to 2017, which could indicate positive changes in employee well-being. However, the spike in year-over-year growth in July 2017 requires further analysis to understand its cause.', 
                    'recommendations': ['Investigate the cause of the spike in year-over-year growth in July 2017. This could involve analyzing employee data, company events, and any policy changes that might have occurred during that period.', 
                                        'Examine the reasons behind the high sick leave days in the Sales department. This could involve analyzing workload, employee stress levels, and workplace environment factors.', 
                                        'Further investigate the high sick leave days taken by Glad. This could involve reviewing medical records, assessing work environment factors, and providing necessary support to improve their well-being.', 
                                        'Continue monitoring the sick leave trends over time to identify any recurring patterns or emerging issues that require attention.'], 
                    'conclusion': 'The Power BI report provides valuable insights into employee sick leave patterns. By analyzing the anomalies and observing the trends, the company can address potential issues, improve employee well-being, and optimize workplace productivity.'
                    }
            ]"""}
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
            output_parser=JsonOutputParser()
            #invoke the model with the prompt and image data
            chain=prompt | llm | output_parser
            resp = chain.invoke({"image_data": image_data})
            return jsonify({"status": "success", "response": resp[0], "image_url": f"/static/uploads/temp_file.jpg"})
        return render_template("index.html")
    except Exception as e:
        print("The Error is : ", e)

if __name__ == "__main__":
    app.run(debug=True)
