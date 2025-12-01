from flask import Flask, render_template, request
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import re

app = Flask(__name__)


groq_api_key = os.getenv('GROQ_API_KEY', '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')

llm_resto = ChatGroq(
    api_key = groq_api_key,
    model = "llama-3.3-70b-versatile",
    temperature=0.0
)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype', 'goal', 'activity_level'],
    template=(
        "You are a professional nutritionist and fitness coach. Create a detailed diet plan AND workout routine based on the following criteria:\n\n"
        "Personal Info: Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} ft\n"
        "Diet: {veg_or_nonveg}, Goal: {goal}, Activity Level: {activity_level}\n"
        "Health: Disease/Conditions: {disease}, Allergies: {allergics}\n"
        "Preferences: Region: {region}, Cuisine: {foodtype}\n\n"
        "CRITICAL: You MUST provide both diet recommendations AND workout recommendations. The workout section is REQUIRED. All food suggestions must be homemade/home-cooked with simple preparation notes—do not mention restaurants, takeout, or packaged meals.\n"
        "SUSTAINABILITY NON-NEGOTIABLES: Favor seasonal, local, bulk-bought ingredients, minimize packaging, highlight ways to reuse leftovers, and offer eco-friendly prep or storage tips for each food section. Mention plant-forward swaps even for non-veg eaters.\n\n"
        "Provide output in EXACTLY this format:\n\n"
        "Daily Nutrition Targets:\n"
        "Calories: [total calories per day]\n"
        "Protein: [grams] ([percentage]%)\n"
        "Carbs: [grams] ([percentage]%)\n"
        "Fats: [grams] ([percentage]%)\n"
        "Fiber: [grams]\n"
        "Sodium: [mg]\n"
        "Calcium: [mg]\n"
        "Iron: [mg]\n"
        "Vitamin D: [IU]\n"
        "Water: [liters]\n\n"
        "Homemade Staples:\n"
        "- staple1 (how to prep or batch cook)\n- staple2\n- staple3\n- staple4\n- staple5\n- staple6\n\n"
        "Breakfast:\n"
        "- item1 (calories, protein, carbs, homemade prep note)\n- item2 (calories, protein, carbs, homemade prep note)\n- item3\n- item4\n- item5\n- item6\n\n"
        "Lunch:\n"
        "- item1 (calories, protein, carbs, homemade prep note)\n- item2 (calories, protein, carbs, homemade prep note)\n- item3\n- item4\n- item5\n- item6\n\n"
        "Dinner:\n"
        "- item1 (calories, protein, carbs, homemade prep note)\n- item2 (calories, protein, carbs, homemade prep note)\n- item3\n- item4\n- item5\n\n"
        "Workouts:\n"
        "- workout1\n- workout2\n- workout3\n- workout4\n- workout5\n- workout6\n\n"
        "REQUIRED: You MUST include the 'Workouts:' section with at least 5-6 specific workout recommendations. Tailor workouts to the {goal} goal:\n"
        "- For BULK: Focus on compound movements, progressive overload, 4-5 days/week strength training\n"
        "- For CUT: Combine strength training with cardio, HIIT workouts, 5-6 days/week\n"
        "- For MAINTAIN: Balanced mix of strength, cardio, and flexibility, 3-5 days/week\n"
        "Include specific exercises like: Bench Press, Squats, Deadlifts, Running, Cycling, Yoga, etc. Be specific with exercise names.\n"
        "Calculate macros based on {goal} goal. For bulk: surplus calories, high protein. For cut: deficit calories, high protein, lower carbs. For maintain: maintenance calories.\n"
    )
)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods = ['POST'])
def recommend():
    if request.method == "POST":
        age = request.form['age']
        gender = request.form['gender']
        weight = request.form['weight']
        height = request.form['height']
        veg_or_nonveg = request.form['veg_or_nonveg']
        disease = request.form['disease']
        region = request.form['region']
        allergics = request.form['allergics']
        foodtype = request.form['foodtype']
        goal = request.form['goal']
        activity_level = request.form['activity_level']


        chain = prompt_template_resto | llm_resto

        input_data = {
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'veg_or_nonveg': veg_or_nonveg,
        'disease':disease,
        'region': region,
        'allergics': allergics,
        'foodtype': foodtype,
        'goal': goal,
        'activity_level': activity_level
        }

        results = chain.invoke(input_data)
        
        
        results_text = results.content if hasattr(results, 'content') else str(results)

        def clean_list(block):
            return [line.strip("- ") for line in block.strip().split("\n") if line.strip()]

        def extract_nutrition_stats(text):
            """Extract nutrition statistics from the AI response"""
            stats = {}
            # Extract calories
            calories_match = re.search(r'Calories:\s*([0-9,]+)', text, re.IGNORECASE)
            if calories_match:
                stats['calories'] = calories_match.group(1).replace(',', '')
            
            # Extract macros
            protein_match = re.search(r'Protein:\s*([0-9,]+)\s*g\s*\(([0-9]+)%\)', text, re.IGNORECASE)
            if protein_match:
                stats['protein'] = protein_match.group(1).replace(',', '')
                stats['protein_percent'] = protein_match.group(2)
            
            carbs_match = re.search(r'Carbs:\s*([0-9,]+)\s*g\s*\(([0-9]+)%\)', text, re.IGNORECASE)
            if carbs_match:
                stats['carbs'] = carbs_match.group(1).replace(',', '')
                stats['carbs_percent'] = carbs_match.group(2)
            
            fats_match = re.search(r'Fats:\s*([0-9,]+)\s*g\s*\(([0-9]+)%\)', text, re.IGNORECASE)
            if fats_match:
                stats['fats'] = fats_match.group(1).replace(',', '')
                stats['fats_percent'] = fats_match.group(2)
            
            # Extract other nutrients
            fiber_match = re.search(r'Fiber:\s*([0-9,]+)\s*g', text, re.IGNORECASE)
            if fiber_match:
                stats['fiber'] = fiber_match.group(1).replace(',', '')
            
            sodium_match = re.search(r'Sodium:\s*([0-9,]+)\s*mg', text, re.IGNORECASE)
            if sodium_match:
                stats['sodium'] = sodium_match.group(1).replace(',', '')
            
            calcium_match = re.search(r'Calcium:\s*([0-9,]+)\s*mg', text, re.IGNORECASE)
            if calcium_match:
                stats['calcium'] = calcium_match.group(1).replace(',', '')
            
            iron_match = re.search(r'Iron:\s*([0-9,]+)\s*mg', text, re.IGNORECASE)
            if iron_match:
                stats['iron'] = iron_match.group(1).replace(',', '')
            
            vitd_match = re.search(r'Vitamin D:\s*([0-9,]+)\s*IU', text, re.IGNORECASE)
            if vitd_match:
                stats['vitamin_d'] = vitd_match.group(1).replace(',', '')
            
            water_match = re.search(r'Water:\s*([0-9.]+)\s*liters?', text, re.IGNORECASE)
            if water_match:
                stats['water'] = water_match.group(1)
            
            return stats

        # Extract nutrition stats
        nutrition_stats = extract_nutrition_stats(results_text)
        
        homemade_staples = re.findall(r'Homemade Staples:\s*(.*?)(?=\n\n|Breakfast:|$)', results_text, re.DOTALL)
        breakfast_names = re.findall(r'Breakfast:\s*(.*?)(?=\n\n|Lunch:|$)', results_text, re.DOTALL)
        lunch_names = re.findall(r'Lunch:\s*(.*?)(?=\n\n|Dinner:|$)', results_text, re.DOTALL)
        dinner_names = re.findall(r'Dinner:\s*(.*?)(?=\n\n|Workouts:|$)', results_text, re.DOTALL)
        # Try multiple patterns for workouts - more flexible matching
        workout_names = re.findall(r'Workouts?:\s*(.*?)(?=\n\n|$)', results_text, re.DOTALL | re.IGNORECASE)
        # If not found, try alternative patterns
        if not workout_names:
            workout_names = re.findall(r'Workout[s\s]*[Rr]ecommendations?:\s*(.*?)(?=\n\n|$)', results_text, re.DOTALL | re.IGNORECASE)
        if not workout_names:
            workout_names = re.findall(r'Exercise[s\s]*[Rr]ecommendations?:\s*(.*?)(?=\n\n|$)', results_text, re.DOTALL | re.IGNORECASE)
        if not workout_names:
            # Try to find any section that mentions workout or exercise
            workout_names = re.findall(r'(?:Workout|Exercise)[s\s]*:?\s*(.*?)(?=\n\n|$)', results_text, re.DOTALL | re.IGNORECASE)
        # Debug: print if workouts are found (can be removed later)
        if not workout_names:
            # Last resort: look for lines starting with "-" after "Workout" keyword anywhere
            workout_section = re.search(r'Workout.*?(?:\n|$)(.*?)(?=\n\n|$)', results_text, re.DOTALL | re.IGNORECASE)
            if workout_section:
                workout_text = workout_section.group(1)
                workout_names = [workout_text]

        homemade_staples = clean_list(homemade_staples[0]) if homemade_staples else []
        breakfast_names = clean_list(breakfast_names[0]) if breakfast_names else []
        lunch_names = clean_list(lunch_names[0]) if lunch_names else []
        dinner_names = clean_list(dinner_names[0]) if dinner_names else []
        workout_names = clean_list(workout_names[0]) if workout_names else []

        return render_template('result.html', 
                            homemade_staples=homemade_staples, 
                            breakfast_names=breakfast_names, 
                            lunch_names=lunch_names, 
                            dinner_names=dinner_names, 
                            workout_names=workout_names,
                            nutrition_stats=nutrition_stats)
    return  render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)