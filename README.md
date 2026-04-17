# DietGPT

DietGPT is a comprehensive, AI-powered nutritionist and fitness coach application inspired by the clean, high-fidelity aesthetics of Apple Health. It enables users to track their daily meals, generate highly personalized diet and workout plans, and chat with an intelligent AI nutritionist.

## Features

- **Google OAuth Login**: Secure authentication flow utilizing Firebase and Authlib.
- **AI Nutritionist Chat**: Persistent, context-aware chatbot that understands your diet goals and past log history to provide personalized advice.
- **Smart Meal Logger**: Simply describe your food and DietGPT will automatically estimate your calorie and macronutrient intake (Protein, Carbs, Fats) using AI.
- **Customized Diet & Workout Plans**: Generate tailored meal plans and workout routines based on specific physical metrics, dietary preferences (e.g., veg/non-veg), region, and goals (Bulk, Cut, Maintain).
- **Responsive Dashboard**: An intuitive Apple Health-inspired UI to track daily caloric progress and review generated plans.

## Tech Stack

- **Backend**: Python, Flask
- **AI & LLM Integration**: LangChain, Groq (Llama-3), OpenRouter (Google Gemini/OpenAI) 
- **Database & Auth**: Firebase Admin SDK (Firestore)
- **Frontend**: HTML Templates, CSS
- **Deployment**: Configured for serverless deployment on Vercel (`vercel.json`)

## Prerequisites

1. **Python 3.9+** installed locally.
2. A **Firebase** project with Firestore and Google Authentication enabled.
3. API keys for **Groq**, **OpenRouter**, or your LLM provider of choice.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dietgpt
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Create a `.env` file in the root directory and populate it with your credentials:
   ```env
   SECRET_KEY=your_flask_secret_key
   BASE_URL=http://127.0.0.1:5000

   # LLM Keys
   GROQ_API_KEY=your_groq_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   CHAT_MODEL=google/gemini-2.5-flash

   # Google OAuth Credentials
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret

   # Firebase Credentials
   FIREBASE_CREDENTIALS=diet-gpt-firebase-adminsdk.json
   ```
   *Make sure your Firebase service account JSON is placed in the project directory.*

5. **Run the application:**
   ```bash
   flask run
   # or
   python app.py
   ```
   The app will be accessible at `http://127.0.0.1:5000`.

## Project Structure

- `app.py`: Main Flask server containing routes for auth, AI chatting, profile management, and dashboard data.
- `requirements.txt`: Python package dependencies.
- `templates/`: HTML files for the front-end views (Dashboard, Profile, Chat, Logger).
- `static/`: Static assets such as CSS stylesheets, scripts, and images.
- `vercel.json`: Deployment configuration for Vercel.

## Deployment

The project is structured to be easily deployed to Vercel. Be sure to configure your environment variables in the Vercel dashboard, including mapping `FIREBASE_CREDENTIALS_JSON` for your Firebase service account instead of relying on the local JSON file.

## License

This project is licensed under the MIT License.
