from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import re

llm_restro = ChatGroq(
    api_key = "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",
    model = "meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature = 0.0
)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template=(
        "Diet Recommendation System:\n"
        "Provide ONLY homemade/home-cooked meal ideas with simple prep notes. Do not mention restaurants, takeout, or packaged meals.\n\n"
        "Homemade Staples:\n"
        "- staple1 (prep note)\n- staple2\n- staple3\n- staple4\n- staple5\n- staple6\n\n"
        "Breakfast:\n"
        "- item1 (calories, macros, prep note)\n- item2\n- item3\n- item4\n- item5\n- item6\n\n"
        "Lunch:\n"
        "- item1 (calories, macros, prep note)\n- item2\n- item3\n- item4\n- item5\n- item6\n\n"
        "Dinner:\n"
        "- item1 (calories, macros, prep note)\n- item2\n- item3\n- item4\n- item5\n\n"
        "Workouts:\n"
        "- workout1\n- workout2\n- workout3\n- workout4\n- workout5\n- workout6\n\n"
        "Criteria:\n"
        "Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} ft, "
        "Vegetarian: {veg_or_nonveg}, Disease: {disease}, Region: {region}, "
        "Allergics: {allergics}, Food Preference: {foodtype}.\n"
    )
)


chain = prompt_template_resto | llm_restro

input_data = {
    'age': 21,
    'gender': 'male',
    'weight': 65,
    'height': 6,
    'veg_or_nonveg': 'veg',
    'disease':'none',
    'region': 'India (Nashik)',
    'allergics': 'none',
    'foodtype': 'italian'
}

results = chain.invoke(input_data)

results_text = results.content if hasattr(results, 'content') else str(results)

homemade_staples = re.findall(r'Homemade Staples:\s*(.*?)\n\n', results_text, re.DOTALL)
breakfast_names = re.findall(r'Breakfast:\s*(.*?)\n\n', results_text, re.DOTALL)
lunch_names = re.findall(r'Lunch:\s*(.*?)\n\n', results_text, re.DOTALL)
dinner_names = re.findall(r'Dinner:\s*(.*?)\n\n', results_text, re.DOTALL)
workout_names = re.findall(r'Workouts:\s*(.*?)\n\n', results_text, re.DOTALL)

def clean_list(block):
    return [line.strip("- ") for line in block.strip().split("\n") if line.strip()]

homemade_staples = clean_list(homemade_staples[0]) if homemade_staples else []
breakfast_names = clean_list(breakfast_names[0]) if breakfast_names else []
lunch_names = clean_list(lunch_names[0]) if lunch_names else []
dinner_names = clean_list(dinner_names[0]) if dinner_names else []
workout_names = clean_list(workout_names[0]) if workout_names else []


print("\n Homemade Staples : \n", "\n".join(homemade_staples))
print("\n Recommended Breakfast : \n", "\n".join(breakfast_names))
print("\n Recommended Dinner : \n", "\n".join(dinner_names))
print("\n Recommended Workouts : \n", "\n".join(workout_names))