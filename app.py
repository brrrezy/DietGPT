import os
import re
from datetime import datetime, time, timedelta, timezone
from functools import wraps
from zoneinfo import ZoneInfo

import firebase_admin
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.firestore_v1.base_query import FieldFilter
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
IST_TIMEZONE = ZoneInfo("Asia/Kolkata")

# Set base URL for canonical tags and structured data
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5000")

# ──────────────────────────────────────────────
# LLM Setup — Groq for diet plans
# ──────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
llm_resto = None
if groq_api_key:
    try:
        llm_resto = ChatGroq(
            api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.0
        )
    except Exception as e:
        print(f"Error initializing ChatGroq: {str(e)}")

# ──────────────────────────────────────────────
# LLM Setup — OpenRouter for chat (or Groq fallback)
# ──────────────────────────────────────────────
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
chat_model_name = os.getenv("CHAT_MODEL", "google/gemini-2.5-flash")
llm_chat = None

if openrouter_api_key:
    try:
        from langchain_openai import ChatOpenAI

        llm_chat = ChatOpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=chat_model_name,
            temperature=0.3,
        )
    except Exception as e:
        print(f"Error initializing OpenRouter chat LLM: {str(e)}")

if not llm_chat and llm_resto:
    llm_chat = llm_resto  # fallback to Groq

# ──────────────────────────────────────────────
# Firebase
# ──────────────────────────────────────────────
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS")
firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

if not firebase_admin._apps:
    import json
    try:
        if firebase_credentials_json:
            # Vercel Production Environment
            cred_dict = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(cred_dict)
        elif firebase_credentials_path and os.path.exists(firebase_credentials_path):
            # Local Development Environment (.env variable mapping)
            cred = credentials.Certificate(firebase_credentials_path)
        elif os.path.exists("diet-gpt-ac388-firebase-adminsdk-fbsvc-3b6f564a71.json"):
            # Hard fallback
            cred = credentials.Certificate("diet-gpt-ac388-firebase-adminsdk-fbsvc-3b6f564a71.json")
        else:
            raise FileNotFoundError("No Firebase credentials found.")
        
        firebase_admin.initialize_app(cred)
        firestore_db = firestore.client()
    except Exception as e:
        print(f"⚠️ Firebase initialization failed: {str(e)}")
        print("🚀 Running in DEMO MODE with Mock Firestore.")
        
        # Simple Mock Firestore for Demo
        class MockDoc:
            def __init__(self, id, data=None):
                self.id = id
                self.exists = data is not None
                self._data = data or {}
            def to_dict(self): return self._data
            def get(self): return self
            @property
            def reference(self): return self
            def delete(self): pass
            def set(self, data, merge=False): pass

        class MockCollection:
            def document(self, id=None): return MockDoc(id or "mock_id")
            def where(self, *args, **kwargs): return self
            def order_by(self, *args, **kwargs): return self
            def limit(self, *args, **kwargs): return self
            def stream(self): return []
            def add(self, data): return None, MockDoc("mock_id", data)

        class MockFirestore:
            def __init__(self):
                self.SERVER_TIMESTAMP = "mock_timestamp"
                class MockQuery:
                    DESCENDING = "descending"
                    ASCENDING = "ascending"
                self.Query = MockQuery()
            def collection(self, name): return MockCollection()
        
        firestore_db = MockFirestore()
else:
    firestore_db = firestore.client()

# ──────────────────────────────────────────────
# OAuth
# ──────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ──────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


def current_user():
    if not session.get("user_id"):
        return None
    user_id = str(session.get("user_id"))
    return {
        "id": user_id,
        "name": session.get("user_name", "User"),
        "email": session.get("user_email"),
        "picture": session.get("user_picture"),
    }


def get_current_user_id():
    uid = session.get("user_id")
    return str(uid) if uid is not None else None


# ──────────────────────────────────────────────
# Profile helpers
# ──────────────────────────────────────────────
def get_user_profile(user_id):
    """Get user profile from Firestore."""
    try:
        doc = firestore_db.collection("users").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        print(f"Error fetching profile: {str(e)}")
    return {}


def save_user_profile(user_id, profile_data):
    """Save/update user profile to Firestore."""
    try:
        firestore_db.collection("users").document(user_id).set(
            profile_data, merge=True
        )
        return True
    except Exception as e:
        print(f"Error saving profile: {str(e)}")
        return False


# ──────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────
def get_today_bounds_utc():
    now_ist = datetime.now(IST_TIMEZONE)
    start_of_today_ist = datetime.combine(
        now_ist.date(), time.min, tzinfo=IST_TIMEZONE
    )
    start_of_tomorrow_ist = start_of_today_ist + timedelta(days=1)
    start_of_today_utc = start_of_today_ist.astimezone(timezone.utc).replace(
        tzinfo=None
    )
    start_of_tomorrow_utc = start_of_tomorrow_ist.astimezone(timezone.utc).replace(
        tzinfo=None
    )
    return start_of_today_utc, start_of_tomorrow_utc


# ──────────────────────────────────────────────
# Meal log helpers
# ──────────────────────────────────────────────
def get_today_meal_logs(user_id):
    start_of_today, start_of_tomorrow = get_today_bounds_utc()
    try:
        docs = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("meal_logs")
            .where(filter=FieldFilter("timestamp", ">=", start_of_today))
            .where(filter=FieldFilter("timestamp", "<", start_of_tomorrow))
            .order_by("timestamp")
            .stream()
        )
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    except Exception as e:
        print(f"Error fetching meal logs: {str(e)}")
        return []


# ──────────────────────────────────────────────
# Chat helpers
# ──────────────────────────────────────────────
def get_chat_messages(user_id, limit=50):
    """Get recent chat messages (persistent, not daily reset)."""
    try:
        docs = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("chats")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        messages = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        messages.reverse()  # oldest first
        return messages
    except Exception as e:
        print(f"Error fetching chat messages: {str(e)}")
        return []


def build_chat_context(chat_messages, limit=20):
    """Build context string from recent chat messages for the AI."""
    if not chat_messages:
        return "No previous conversation."
    recent = chat_messages[-limit:]
    lines = []
    for chat in recent:
        lines.append(f"User: {chat.get('message', '')}")
        lines.append(f"Assistant: {chat.get('response', '')}")
    return "\n".join(lines)


def clear_old_chats_safe():
    """Cleanup old chats — non-blocking, runs silently."""
    user_id = session.get("user_id")
    if not user_id:
        return
    user_id = str(user_id)
    try:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
        old_docs = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("chats")
            .where(filter=FieldFilter("timestamp", "<", cutoff))
            .limit(20)
            .stream()
        )
        for doc in old_docs:
            doc.reference.delete()
    except Exception:
        pass


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────
chat_prompt_template = PromptTemplate(
    input_variables=[
        "user_profile_summary",
        "today_meals_summary",
        "chat_history",
        "current_message",
    ],
    template=(
        "You are the DietGPT AI Buddy.\n"
        "Your mission: Provide short, crisp, and modular nutritional advice.\n\n"
        "STRICT STYLE GUIDELINES:\n"
        "- BE CONCISE. No long introductions or conclusions.\n"
        "- BE MODULAR. Use bullet points and bold headers for clarity.\n"
        "- NO BULKY TEXT. Break information into small, digestible blocks.\n"
        "- CONTEXT: Use the User Profile and Today's Log to give fast, relevant answers.\n\n"
        "USER CONTEXT:\n"
        "- Profile: {user_profile_summary}\n"
        "- Today's Log: {today_meals_summary}\n"
        "- History: {chat_history}\n\n"
        "USER QUESTION: {current_message}\n\n"
        "Buddy Response (Short & Modular):"
    ),
)

prompt_template_resto = PromptTemplate(
    input_variables=[
        "age",
        "gender",
        "weight",
        "height",
        "veg_or_nonveg",
        "disease",
        "region",
        "allergics",
        "foodtype",
        "goal",
        "activity_level",
    ],
    template=(
        "You are a professional and helpful AI Nutritionist.\n"
        "Your mission is to craft a detailed, well-organized, and clear meal and workout plan.\n\n"

        "CORE PHILOSOPHY:\n"
        "- Clear and direct advice.\n"
        "- Tailor everything to the user's specific data.\n"
        "- Focus on homemade, whole-food, regional ingredients.\n\n"

        "USER BIOMETRICS & CONTEXT:\n"
        "- Profile: {age}yo {gender}, {weight}kg, {height}ft\n"
        "- Target: {goal} ({activity_level} activity)\n"
        "- Health: {disease} (Allergies: {allergics})\n"
        "- Context: {region} region, {veg_or_nonveg} {foodtype} preference\n\n"

        "YOUR PLAN (OUTPUT IN THIS EXACT FORMAT - BE DETAILED AND EXPLANATORY):\n\n"

        "--- DAILY TARGETS ---\n"
        "Daily Calories: <number> kcal\n"
        "Protein: <number>g\n"
        "Carbs: <number>g\n"
        "Fats: <number>g\n"
        "Fiber: <number>g\n"
        "Hydration: <number> liters\n"
        "Plan Rationale: [Explain the nutrition strategy in 2 simple sentences]\n\n"

        "--- MEAL PLAN ---\n\n"

        "Breakfast: [Name of Dish]\n"
        "Why this meal: [Detailed explanation of why this meal is chosen]\n"
        "Primary Option:\n"
        "- item (qty) - macros\n"
        "- item\n"
        "Alternative Option:\n"
        "- item (qty) - macros\n"
        "- item\n"
        "Quick Tip: [A simple tip for this meal]\n\n"

        "Lunch: [Name of Dish]\n"
        "Why this meal: [Detailed explanation of why this meal is chosen]\n"
        "Primary Option:\n"
        "- item (qty) - macros\n"
        "Alternative Option:\n"
        "- item (qty) - macros\n"
        "Quick Tip: [A simple tip for this meal]\n\n"

        "Dinner: [Name of Dish]\n"
        "Why this meal: [Detailed explanation of why this meal is chosen]\n"
        "Primary Option:\n"
        "- item (qty) - macros\n"
        "Alternative Option:\n"
        "- item (qty) - macros\n"
        "Quick Tip: [A simple tip for this meal]\n\n"

        "Snacks & Staples:\n"
        "- [Item 1 Name]: [Detail]\n"
        "- [Item 2 Name]: [Detail]\n\n"

        "--- WORKOUT PLAN ---\n"
        "Phase: [Workout Name]\n"
        "Goal Strategy: [Explain the workout plan in simple terms]\n\n"
        "Day 1: [Focus]\n"
        "- [Exercise 1]: [Sets] x [Reps] - [Tip]\n"
        "- [Exercise 2]: ...\n"
        "- [Exercise 3]: ...\n\n"
        "Day 2: [Focus]\n"
        "- [Exercise 1]: ...\n\n"
        "Day 3: [Focus / Recovery]\n"
        "- [Activity]: ...\n\n"
        "Day 4: [Focus]\n"
        "- [Exercise 1]: ...\n\n"
        "Day 5: [Focus]\n"
        "- [Exercise 1]: ...\n\n"

        "--- LIFESTYLE TIPS ---\n"
        "Meal Prep Tip: [Simple efficiency strategy]\n"
        "Recovery Tip: [Simple tip for sleep or recovery]\n"
    ),
)

# AI calorie estimator prompt
calorie_estimate_prompt = PromptTemplate(
    input_variables=["food_description"],
    template=(
        "Estimate nutrition for a typical Indian home portion.\n\n"

        "RULES:\n"
        "- Numbers only\n"
        "- No explanation\n"
        "- If unclear → make best reasonable estimate\n\n"

        "FOOD:\n{food_description}\n\n"

        "OUTPUT (STRICT):\n"
        "calories: <number>\n"
        "protein: <number>\n"
        "carbs: <number>\n"
        "fats: <number>"
    ),
)


# ──────────────────────────────────────────────
# Before request
# ──────────────────────────────────────────────
@app.before_request
def run_daily_cleanup():
    try:
        clear_old_chats_safe()
    except Exception:
        pass


# ──────────────────────────────────────────────
# Auth routes
# ──────────────────────────────────────────────
@app.route("/login", methods=["GET"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    token = google.authorize_access_token()
    user_info = token.get("userinfo")
    if not user_info:
        user_info = google.userinfo()
    if not user_info or not user_info.get("sub"):
        flash("Google login failed. Please try again.", "error")
        return redirect(url_for("login"))

    user_id = user_info["sub"]
    try:
        user_doc = firestore_db.collection("users").document(user_id)
        user_doc.set(
            {
                "name": user_info.get("name", "User"),
                "email": user_info.get("email"),
                "picture": user_info.get("picture"),
                "last_login_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "created_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
    except Exception as e:
        print(f"Firestore user save failed (non-blocking): {str(e)}")

    session["user_id"] = user_id
    session["user_name"] = user_info.get("name", "User")
    session["user_email"] = user_info.get("email")
    session["user_picture"] = user_info.get("picture")
    return redirect(url_for("dashboard"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────
@app.route("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    plans = []
    latest_plan = None
    profile = {}
    today_logs = []
    today_calories = 0

    try:
        profile = get_user_profile(user["id"])
        plan_docs = (
            firestore_db.collection("users")
            .document(user["id"])
            .collection("plans")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(10)
            .stream()
        )
        plans = [{"id": d.id, **d.to_dict()} for d in plan_docs]
        latest_plan = plans[0] if plans else None

        today_logs = get_today_meal_logs(user["id"])
        today_calories = sum(int(log.get("calories", 0)) for log in today_logs)
        today_protein = sum(int(log.get("protein", 0)) for log in today_logs)
        today_carbs = sum(int(log.get("carbs", 0)) for log in today_logs)
        today_fats = sum(int(log.get("fats", 0)) for log in today_logs)
    except Exception as e:
        print(f"Dashboard data load error: {str(e)}")

    target_calories = 0
    if profile.get("target_calories"):
        target_calories = int(profile["target_calories"])
    elif latest_plan and latest_plan.get("nutrition_stats", {}).get("calories"):
        target_calories = int(latest_plan["nutrition_stats"]["calories"])

    return render_template(
        "dashboard.html",
        user=user,
        profile=profile,
        plans=plans,
        latest_plan=latest_plan,
        today_logs=today_logs,
        today_calories=today_calories,
        today_protein=today_protein,
        today_carbs=today_carbs,
        today_fats=today_fats,
        target_calories=target_calories,
    )


# ──────────────────────────────────────────────
# Profile
# ──────────────────────────────────────────────
@app.route("/profile", methods=["GET"])
@login_required
def profile_page():
    user = current_user()
    profile = get_user_profile(user["id"])
    return render_template("profile.html", user=user, profile=profile)


@app.route("/profile", methods=["POST"])
@login_required
def profile_save():
    user = current_user()
    profile_data = {
        "weight": request.form.get("weight", ""),
        "height": request.form.get("height", ""),
        "age": request.form.get("age", ""),
        "gender": request.form.get("gender", ""),
        "goal": request.form.get("goal", "maintain"),
        "activity_level": request.form.get("activity_level", "moderate"),
        "diet_pref": request.form.get("diet_pref", ""),
        "region": request.form.get("region", ""),
        "allergies": request.form.get("allergies", ""),
        "target_calories": request.form.get("target_calories", ""),
        "profile_updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    if save_user_profile(user["id"], profile_data):
        flash("Profile saved!", "success")
    else:
        flash("Could not save profile. Try again.", "error")
    return redirect(url_for("profile_page"))


# ──────────────────────────────────────────────
# Logger (Meal Logging)
# ──────────────────────────────────────────────
@app.route("/logger")
@login_required
def logger_page():
    user = current_user()
    profile = get_user_profile(user["id"])
    today_logs = get_today_meal_logs(user["id"])
    today_calories = sum(int(log.get("calories", 0)) for log in today_logs)
    today_protein = sum(int(log.get("protein", 0)) for log in today_logs)
    today_carbs = sum(int(log.get("carbs", 0)) for log in today_logs)
    today_fats = sum(int(log.get("fats", 0)) for log in today_logs)

    target_calories = int(profile.get("target_calories", 0) or 0)

    return render_template(
        "logger.html",
        user=user,
        profile=profile,
        today_logs=today_logs,
        today_calories=today_calories,
        today_protein=today_protein,
        today_carbs=today_carbs,
        today_fats=today_fats,
        target_calories=target_calories,
    )


@app.route("/log_meal", methods=["POST"])
@login_required
def log_meal():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized."}), 401

        payload = request.get_json(silent=True) or {}
        description = (payload.get("description") or "").strip()
        meal_type = payload.get("meal_type", "snack")

        if not description:
            return jsonify({"error": "Food description is required."}), 400

        # AI calorie estimation
        calories = payload.get("calories")
        protein = payload.get("protein")
        carbs = payload.get("carbs")
        fats = payload.get("fats")

        if not calories and llm_chat:
            try:
                est_chain = calorie_estimate_prompt | llm_chat
                est_result = est_chain.invoke({"food_description": description})
                est_text = (
                    est_result.content
                    if hasattr(est_result, "content")
                    else str(est_result)
                )
                cal_match = re.search(r"calories:\s*(\d+)", est_text, re.IGNORECASE)
                prot_match = re.search(r"protein:\s*(\d+)", est_text, re.IGNORECASE)
                carb_match = re.search(r"carbs:\s*(\d+)", est_text, re.IGNORECASE)
                fat_match = re.search(r"fats:\s*(\d+)", est_text, re.IGNORECASE)
                calories = int(cal_match.group(1)) if cal_match else 0
                protein = int(prot_match.group(1)) if prot_match else 0
                carbs = int(carb_match.group(1)) if carb_match else 0
                fats = int(fat_match.group(1)) if fat_match else 0
            except Exception as e:
                print(f"Calorie estimation error: {str(e)}")
                calories = 0
                protein = 0
                carbs = 0
                fats = 0

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        log_ref = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("meal_logs")
            .document()
        )
        log_data = {
            "description": description,
            "meal_type": meal_type,
            "calories": int(calories or 0),
            "protein": int(protein or 0),
            "carbs": int(carbs or 0),
            "fats": int(fats or 0),
            "timestamp": timestamp,
        }
        log_ref.set(log_data)

        return (
            jsonify(
                {
                    "id": log_ref.id,
                    **log_data,
                    "timestamp": timestamp.isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        print(f"Error in log_meal: {str(e)}")
        return jsonify({"error": "Failed to log meal."}), 500


@app.route("/delete_meal/<meal_id>", methods=["DELETE"])
@login_required
def delete_meal(meal_id):
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized."}), 401
        firestore_db.collection("users").document(user_id).collection(
            "meal_logs"
        ).document(meal_id).delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error deleting meal: {str(e)}")
        return jsonify({"error": "Failed to delete meal."}), 500


@app.route("/api/today_logs", methods=["GET"])
@login_required
def api_today_logs():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized."}), 401
    logs = get_today_meal_logs(user_id)
    for log in logs:
        if log.get("timestamp"):
            log["timestamp"] = log["timestamp"].isoformat()
    return jsonify({"logs": logs}), 200


# ──────────────────────────────────────────────
# Chat (AI Friend — with memory)
# ──────────────────────────────────────────────
@app.route("/chat_page")
@login_required
def chat_page():
    user = current_user()
    profile = get_user_profile(user["id"])
    return render_template("chat.html", user=user, profile=profile)


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized."}), 401

        if not llm_chat:
            return (
                jsonify({"error": "Chat AI not configured. Add an API key in settings."}),
                500,
            )

        payload = request.get_json(silent=True) or {}
        current_message = (payload.get("message") or "").strip()
        if not current_message:
            return jsonify({"error": "Message is required."}), 400

        # Auto-load profile
        profile = get_user_profile(user_id)
        profile_parts = []
        if profile.get("weight"):
            profile_parts.append(f"Weight: {profile['weight']} kg")
        if profile.get("height"):
            profile_parts.append(f"Height: {profile['height']} ft")
        if profile.get("age"):
            profile_parts.append(f"Age: {profile['age']}")
        if profile.get("gender"):
            profile_parts.append(f"Gender: {profile['gender']}")
        if profile.get("goal"):
            profile_parts.append(f"Goal: {profile['goal']}")
        if profile.get("target_calories"):
            profile_parts.append(f"Target calories: {profile['target_calories']}")
        if profile.get("diet_pref"):
            profile_parts.append(f"Diet: {profile['diet_pref']}")
        user_profile_summary = (
            ", ".join(profile_parts) if profile_parts else "No profile set up yet."
        )

        # Today's meals
        today_logs = get_today_meal_logs(user_id)
        if today_logs:
            meal_lines = []
            total_cal = 0
            for log in today_logs:
                cal = log.get("calories", 0)
                total_cal += int(cal)
                meal_lines.append(
                    f"- {log.get('meal_type', 'meal').title()}: {log.get('description', 'N/A')} ({cal} cal)"
                )
            meal_lines.append(f"Total today: {total_cal} cal")
            today_meals_summary = "\n".join(meal_lines)
        else:
            today_meals_summary = "No meals logged today yet."

        # Chat history (persistent)
        chat_messages = get_chat_messages(user_id, limit=30)
        chat_history = build_chat_context(chat_messages, limit=15)

        chat_chain = chat_prompt_template | llm_chat
        ai_result = chat_chain.invoke(
            {
                "user_profile_summary": user_profile_summary,
                "today_meals_summary": today_meals_summary,
                "chat_history": chat_history,
                "current_message": current_message,
            }
        )

        response_text = (
            ai_result.content if hasattr(ai_result, "content") else str(ai_result)
        )

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        chat_ref = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("chats")
            .document()
        )
        chat_ref.set(
            {
                "message": current_message,
                "response": response_text,
                "timestamp": timestamp,
            }
        )

        return (
            jsonify(
                {
                    "id": chat_ref.id,
                    "message": current_message,
                    "response": response_text,
                    "timestamp": timestamp.isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({"error": "Failed to process message."}), 500


@app.route("/chat_history", methods=["GET"])
@login_required
def chat_history():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized."}), 401

        messages = get_chat_messages(user_id, limit=50)
        return (
            jsonify(
                {
                    "messages": [
                        {
                            "id": chat["id"],
                            "message": chat.get("message"),
                            "response": chat.get("response"),
                            "timestamp": chat.get("timestamp").isoformat()
                            if chat.get("timestamp")
                            else None,
                        }
                        for chat in messages
                    ],
                }
            ),
            200,
        )
    except Exception as e:
        print(f"Error in chat_history route: {str(e)}")
        return jsonify({"error": "Failed to fetch chat history."}), 500


@app.route("/chat_clear", methods=["POST"])
@login_required
def chat_clear():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized."}), 401
        docs = (
            firestore_db.collection("users")
            .document(user_id)
            .collection("chats")
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error clearing chats: {str(e)}")
        return jsonify({"error": "Failed to clear chats."}), 500


# ──────────────────────────────────────────────
# Diet Plan Generation
# ──────────────────────────────────────────────
@app.route("/plan/new")
@login_required
def index():
    user = current_user()
    profile = get_user_profile(user["id"])
    return render_template("index.html", base_url=BASE_URL, user=user, profile=profile)


@app.route("/recommend", methods=["POST"])
@login_required
def recommend():
    try:
        if request.method == "POST":
            age = request.form.get("age", "")
            gender = request.form.get("gender", "")
            weight = request.form.get("weight", "")
            height_ft = request.form.get("height_ft", "")
            height_in = request.form.get("height_in", "")
            veg_or_nonveg = request.form.get("veg_or_nonveg", "")
            disease = request.form.get("disease", "none")
            region = request.form.get("region", "")
            allergics = request.form.get("allergics", "none")
            foodtype = request.form.get("foodtype", "")
            goal = request.form.get("goal", "")
            activity_level = request.form.get("activity_level", "")

            # Combine height for prompt
            height = f"{height_ft}ft {height_in}in" if height_ft else ""

            if not all(
                [age, gender, weight, height_ft, height_in, veg_or_nonveg, region, foodtype, goal, activity_level]
            ):
                return (
                    render_template(
                        "index.html",
                        error="Please fill in all required fields.",
                        base_url=BASE_URL,
                        user=current_user(),
                    ),
                    400,
                )

            if not llm_resto:
                return (
                    render_template(
                        "index.html",
                        error="API configuration error. Please contact support.",
                        base_url=BASE_URL,
                        user=current_user(),
                    ),
                    500,
                )

            chain = prompt_template_resto | llm_resto

            input_data = {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "veg_or_nonveg": veg_or_nonveg,
                "disease": disease,
                "region": region,
                "allergics": allergics,
                "foodtype": foodtype,
                "goal": goal,
                "activity_level": activity_level,
            }

            results = chain.invoke(input_data)
            results_text = (
                results.content if hasattr(results, "content") else str(results)
            )

            def clean_list(block):
                return [
                    line.strip("- ")
                    for line in block.strip().split("\n")
                    if line.strip()
                ]

            def extract_nutrition_stats(text):
                stats = {}
                stats["calories"] = (re.findall(r"Daily Calories:\s*(\d+)", text, re.I) or ["0"])[0]
                stats["protein"] = (re.findall(r"Protein:\s*(\d+)", text, re.I) or ["0"])[0]
                stats["carbs"] = (re.findall(r"Carbs:\s*(\d+)", text, re.I) or ["0"])[0]
                stats["fats"] = (re.findall(r"Fats:\s*(\d+)", text, re.I) or ["0"])[0]
                stats["fiber"] = (re.findall(r"Fiber:\s*(\d+)", text, re.I) or ["0"])[0]
                stats["water"] = (re.findall(r"(?:Hydration|Water):\s*([\d.]+)", text, re.I) or ["0"])[0]
                stats["rationale"] = (re.findall(r"Elite Rationale:\s*(.*?)(?=\n\n|$)", text, re.DOTALL | re.I) or [""])[0].strip()
                return stats

            nutrition_stats = extract_nutrition_stats(results_text)

            # Updated regex with more flexible headers to match elite format
            homemade_staples = re.findall(r"(?:Homemade Staples|Snacks):\s*(.*?)(?=\n\n|---|$)", results_text, re.DOTALL | re.I)
            breakfast_names = re.findall(r"Breakfast:\s*(.*?)(?=\n\n|Lunch:|$)", results_text, re.DOTALL | re.I)
            lunch_names = re.findall(r"Lunch:\s*(.*?)(?=\n\n|Dinner:|$)", results_text, re.DOTALL | re.I)
            dinner_names = re.findall(r"Dinner:\s*(.*?)(?=\n\n|---|$)", results_text, re.DOTALL | re.I)
            workout_names = re.findall(r"(?:Workouts?|Performance Workout Protocol):\s*(.*?)(?=\n\n|---|$)", results_text, re.DOTALL | re.I)

            # Extract Elite Tips
            performance_secrets = re.findall(r"\[Chef's Performance Secret\]:\s*(.*?)(?=\n|$)", results_text, re.I)
            meal_prep_hack = (re.findall(r"Meal Prep Hack:\s*(.*?)(?=\n|$)", results_text, re.I) or [""])[0]
            recovery_secret = (re.findall(r"Recovery Secret:\s*(.*?)(?=\n|$)", results_text, re.I) or [""])[0]

            homemade_staples = clean_list(homemade_staples[0]) if homemade_staples else []
            breakfast_names = clean_list(breakfast_names[0]) if breakfast_names else []
            lunch_names = clean_list(lunch_names[0]) if lunch_names else []
            dinner_names = clean_list(dinner_names[0]) if dinner_names else []
            workout_names = clean_list(workout_names[0]) if workout_names else []

            user = current_user()
            if user:
                # Save plan
                plan_ref = (
                    firestore_db.collection("users")
                    .document(user["id"])
                    .collection("plans")
                    .document()
                )
                plan_ref.set(
                    {
                        "plan_text": results_text,
                        "nutrition_stats": nutrition_stats,
                        "breakfast": breakfast_data,
                        "lunch": lunch_data,
                        "dinner": dinner_data,
                        "workout_rationale": workout_rationale,
                        "workout_days": workout_days,
                        "staples_text": staples_block,
                        "meal_prep_hack": meal_prep_hack,
                        "recovery_secret": recovery_secret,
                        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    }
                )
                # Auto-update profile with plan data
                profile_update = {"goal": goal, "activity_level": activity_level}
                if weight:
                    profile_update["weight"] = weight
                if height:
                    profile_update["height"] = height
                if age:
                    profile_update["age"] = age
                if gender:
                    profile_update["gender"] = gender
                if veg_or_nonveg:
                    profile_update["diet_pref"] = veg_or_nonveg
                if region:
                    profile_update["region"] = region
                if allergics and allergics != "none":
                    profile_update["allergies"] = allergics
                if nutrition_stats.get("calories"):
                    profile_update["target_calories"] = nutrition_stats["calories"]
                save_user_profile(user["id"], profile_update)

            return render_template(
                "result.html",
                nutrition_stats=nutrition_stats,
                breakfast=breakfast_data,
                lunch=lunch_data,
                dinner=dinner_data,
                workout_rationale=workout_rationale,
                workout_days=workout_days,
                staples_text=staples_block,
                meal_prep_hack=meal_prep_hack,
                recovery_secret=recovery_secret,
                base_url=BASE_URL,
                user=user,
            )
    except KeyError as e:
        return (
            render_template(
                "index.html",
                error=f"Missing required field: {str(e)}",
                base_url=BASE_URL,
                user=current_user(),
            ),
            400,
        )
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return (
            render_template(
                "index.html",
                error="An error occurred while generating recommendations. Please try again.",
                base_url=BASE_URL,
                user=current_user(),
            ),
            500,
        )

    return render_template("index.html", base_url=BASE_URL, user=current_user())


@app.route("/plan/<plan_id>")
@login_required
def view_plan(plan_id):
    user = current_user()
    plan_ref = (
        firestore_db.collection("users")
        .document(user["id"])
        .collection("plans")
        .document(str(plan_id))
    )
    doc = plan_ref.get()
    if not doc.exists:
        abort(404)
    plan = {"id": doc.id, **doc.to_dict()}
    nutrition_stats = plan.get("nutrition_stats", {})
    return render_template(
        "saved_plan.html", user=user, plan=plan, nutrition_stats=nutrition_stats
    )


# ──────────────────────────────────────────────
# SEO routes
# ──────────────────────────────────────────────
@app.route("/robots.txt")
def robots():
    try:
        return send_from_directory(".", "robots.txt", mimetype="text/plain")
    except Exception:
        return Response(
            "User-agent: *\nAllow: /\nSitemap: https://fixyourdiet.vercel.app/sitemap.xml",
            mimetype="text/plain",
        )


@app.route("/sitemap.xml")
def sitemap():
    try:
        return send_from_directory(".", "sitemap.xml", mimetype="application/xml")
    except Exception:
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://fixyourdiet.vercel.app/</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>"""
        return Response(sitemap_content, mimetype="application/xml")


if __name__ == "__main__":
    app.run(debug=True)
