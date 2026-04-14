from flask import Flask, render_template, request, send_from_directory, Response
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import re

app = Flask(__name__)

# Set base URL for canonical tags and structured data
BASE_URL = os.getenv('BASE_URL', 'https://getfitdiet.vercel.app')

groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM only if API key is available
llm_resto = None
if groq_api_key:
    try:
        llm_resto = ChatGroq(
            api_key = groq_api_key,
            model = "llama-3.3-70b-versatile",
            temperature=0.0
        )
    except Exception as e:
        print(f"Error initializing ChatGroq: {str(e)}")
        llm_resto = None

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
        "- staple1 (quantity/portion, how to prep or batch cook)\n- staple2 (quantity/portion, prep note)\n- staple3 (quantity/portion, prep note)\n- staple4 (quantity/portion, prep note)\n- staple5 (quantity/portion, prep note)\n- staple6 (quantity/portion, prep note)\n\n"
        "Breakfast:\n"
        "- item1 (quantity/portion: e.g., 2 eggs, 1 cup oats, 100g chicken) - calories, protein, carbs, homemade prep note\n- item2 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item3 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item4 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item5 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item6 (quantity/portion) - calories, protein, carbs, homemade prep note\n\n"
        "Lunch:\n"
        "- item1 (quantity/portion: e.g., 150g rice, 200g vegetables, 120g protein) - calories, protein, carbs, homemade prep note\n- item2 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item3 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item4 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item5 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item6 (quantity/portion) - calories, protein, carbs, homemade prep note\n\n"
        "Dinner:\n"
        "- item1 (quantity/portion: e.g., 100g protein, 1 cup vegetables, 80g carbs) - calories, protein, carbs, homemade prep note\n- item2 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item3 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item4 (quantity/portion) - calories, protein, carbs, homemade prep note\n- item5 (quantity/portion) - calories, protein, carbs, homemade prep note\n\n"
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
    return render_template("index.html", base_url=BASE_URL)

@app.route('/recommend', methods = ['POST'])
def recommend():
    try:
        if request.method == "POST":
            # Get form data with defaults to prevent KeyError
            age = request.form.get('age', '')
            gender = request.form.get('gender', '')
            weight = request.form.get('weight', '')
            height = request.form.get('height', '')
            veg_or_nonveg = request.form.get('veg_or_nonveg', '')
            disease = request.form.get('disease', 'none')
            region = request.form.get('region', '')
            allergics = request.form.get('allergics', 'none')
            foodtype = request.form.get('foodtype', '')
            goal = request.form.get('goal', '')
            activity_level = request.form.get('activity_level', '')

            # Validate required fields
            if not all([age, gender, weight, height, veg_or_nonveg, region, foodtype, goal, activity_level]):
                return render_template("index.html", error="Please fill in all required fields.", base_url=BASE_URL), 400

            # Check if LLM is initialized
            if not llm_resto:
                return render_template("index.html", error="API configuration error. Please contact support.", base_url=BASE_URL), 500

            chain = prompt_template_resto | llm_resto

            input_data = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'veg_or_nonveg': veg_or_nonveg,
                'disease': disease,
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
                                nutrition_stats=nutrition_stats,
                                base_url=BASE_URL)
    except KeyError as e:
        return render_template("index.html", error=f"Missing required field: {str(e)}", base_url=BASE_URL), 400
    except Exception as e:
        # Log the error for debugging (in production, use proper logging)
        print(f"Error in recommend route: {str(e)}")
        return render_template("index.html", error="An error occurred while generating recommendations. Please try again.", base_url=BASE_URL), 500
    
    return render_template("index.html", base_url=BASE_URL)

@app.route('/robots.txt')
def robots():
    try:
        return send_from_directory('.', 'robots.txt', mimetype='text/plain')
    except:
        return Response("User-agent: *\nAllow: /\nSitemap: https://dietgpt.com/sitemap.xml", mimetype='text/plain')

@app.route('/sitemap.xml')
def sitemap():
    try:
        return send_from_directory('.', 'sitemap.xml', mimetype='application/xml')
    except:
        sitemap_content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://dietgpt.com/</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>'''
        return Response(sitemap_content, mimetype='application/xml')

if __name__ == "__main__":
    app.run(debug=True)
