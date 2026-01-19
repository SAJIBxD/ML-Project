import gradio as gr
import pandas as pd
import numpy as np
import pickle

# load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# main logic
def predict_charges(age, sex, bmi, children, smoker, region):
    
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]],
    columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    )
    
    prediction = model.predict(input_df)[0]
    
    return f"Appoximate insurance cost is ${prediction:.2f}"

# interface
#['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
inputs = [
    gr.Number(label="Age", value=18),
    gr.Radio(['male', 'female'], label='Sex'),
    gr.Number(label="BMI", value=30),
    gr.Slider(0, 6, step=1, label="Children"),
    gr.Radio(['yes', 'no'], label='Smoker'),
    gr.Dropdown(['southwest', 'southeast', 'northwest', 'northeast'], label="Region")
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs='text',
    title='Insurence Cost Predictor'
)

app.launch()
