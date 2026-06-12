import os
import time
import logging
import json
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# if not os.environ.get("GEMINI_API_KEY"):
#     logger.error("GEMINI_API_KEY not found in environment.")



# --- GOOGLE AI CONFIGURATION ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    # We are using Gemini 1.5 Flash (Google's fastest and most capable free-tier model)
    model_name = "gemma-4-26b-a4b-it"
    chat_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.environ.get("GEMINI_API_KEY"), # Uncommented and using GEMINI_API_KEY
        temperature=0.7,
    )
    logger.info(f"Google ChatLLM ({model_name}) initialized successfully.")
except ImportError:
    logger.error("langchain_google_genai not installed. Chat functionality will be limited.")
    chat_llm = None
except Exception as e:
    logger.error(f"Failed to initialize Chat LLM: {e}")
    chat_llm = None

# Import our RAG Retriever function
try:
    from src.rag.retriever import get_relevant_context
except ImportError:
    logger.warning("RAG retriever not found. Proceeding without medical knowledge base.")
    def get_relevant_context(q: str, k: int = 2) -> str:
        return ""

def _rate_limit_callback(output):
    """
    Forces a 15-second pause after an agent finishes a task.
    This guarantees we never hit Google's 5 Requests Per Minute (RPM) free-tier limit.
    """
    logger.info("Task completed. Pausing for 15 seconds to respect Google's 5 RPM API limit...")
    time.sleep(15)

def run_health_analysis(historical_metrics: list, user_profile: dict) -> str:
    """
    Uses the "Expert Council" prompt pattern to force a single AI Agent to evaluate data
    from multiple specialist perspectives, guaranteeing exactly 1 API request.
    """
    logger.info("Initializing E.V.E. Council for historical health analysis...")

    goal = user_profile.get("primary_goal", "General Health")
    activity_level = user_profile.get("activity_level", "Moderate")
    caffeine = user_profile.get("caffeine", "Moderate")
    alcohol = user_profile.get("alcohol", "Rarely")
    late_meals = user_profile.get("late_meals", "Rarely")

    profile_context = f"""
    USER LIFESTYLE CONTEXT:
    - Primary Goal: {goal}
    - Baseline Activity Level: {activity_level}
    - Caffeine Intake: {caffeine}
    - Alcohol Consumption: {alcohol}
    - Late/Heavy Meals: {late_meals}
    """

    gemini_model = "gemini/gemma-4-26b-a4b-it"

    # ==========================================
    # STEP 1: DEFINE THE SINGLE MASTER AGENT
    # ==========================================

    eve_master_agent = Agent(
        role="E.V.E. (Entity for Vital Evaluation)",
        goal="Perform a multi-disciplinary health analysis of a multi-day timeline and deliver a unified clinical briefing.",
        backstory=(
            "You are E.V.E., an advanced Cognitive Health Intelligence. You contain sub-routines for "
            "Cardiology, Sleep Science, and Physical Therapy. You analyze multi-day historical data sequentially through these "
            "three lenses before delivering your final verdict to the user. You NEVER use flowery language; "
            "you are highly clinical, direct, and cause-and-effect oriented."
        ),
        llm=gemini_model,
        verbose=True
    )

    # ==========================================
    # STEP 2: DEFINE THE "EXPERT COUNCIL" TASK
    # ==========================================

    council_task = Task(
        description=f"""
        Analyze the following historical timeline data of the user (from oldest to newest):
        {historical_metrics}
        
        {profile_context}
        
        INTERNAL PROCESSING STEPS (Do not output these steps, just use them to form your conclusion):
        1. (Cardiac Protocol): Assess the Resting HR trends (Target 60). Correlate with alcohol/caffeine/meals.
        2. (Sleep Protocol): Assess Sleep duration trends (Target 8.0). Correlate with lifestyle profile.
        3. (Kinetic Protocol): Assess Steps trends (Target 10,000) against the user's primary goal.
        
        FINAL OUTPUT REQUIREMENTS:
        Write a final clinical briefing addressed to the user based on your internal processing of their historical timeline.
        1. Start exactly with: '<strong>E.V.E. // SYSTEM DIAGNOSTIC</strong><br>'
        2. Write a single, cohesive paragraph that points out exactly HOW their reported lifestyle habits are impacting their historical metric trends over this period. Be clinical.
        3. End with a bolded '<strong>Directive:</strong>' followed by one specific, actionable adjustment based on the most pressing trend.
        4. Keep the entire response strictly under 5 sentences. Format for direct HTML injection.
        """,
        expected_output="A clinical, 4-5 sentence HTML paragraph summarizing the multi-disciplinary analysis of historical data.",
        agent=eve_master_agent
    )

    # ==========================================
    # STEP 3: RUN THE CREW
    # ==========================================
    # This guarantees exactly 1 task execution = 1 API request.
    medical_crew = Crew(
        agents=[eve_master_agent],
        tasks=[council_task],
        process=Process.sequential,
        verbose=True
    )

    logger.info("Crew assembled. Starting Expert Council workflow...")
    try:
        result = medical_crew.kickoff()
        logger.info("Analysis complete.")
        return str(result)
    except Exception as e:
        logger.error(f"CrewAI execution failed: {e}")
        return f"ERROR: E.V.E. encountered a critical failure. Details: {e}"

def generate_chat_response(prompt: str, chat_history: list, historical_metrics: list, user_profile: dict) -> str:
    """
    Handles interactive chat using the general LLM, providing context of the user's data
    AND retrieving context from our medical knowledge base (RAG).
    """
    if not chat_llm:
        return "I'm sorry, my cognitive processors (LLM) are currently offline. Please check your API keys."

    try:
        logger.info(f"Querying ChromaDB for: '{prompt}'")
        medical_context = get_relevant_context(prompt, k=2)

        context = f"""You are E.V.E. (Entity for Vital Evaluation), an advanced Cognitive Health Intelligence assistant.
        
User's Historical Metrics (Last 7-30 days):
{historical_metrics}
The available metrics are: Active Energy, HRV, Average Heart Rate, Resting Heart Rate, Steps, Sleep Duration, Walking Steadiness, Walking/Running Speed, Stair Flights Climbed, Stride Length, High/Low Heart Rate Notifications, ECG, Blood Oxygen (SpO2), Respiratory Rate, Deep Sleep, REM Sleep, Core Sleep, Awake Sessions, Workouts (Type, Duration, Distance, Calories, Route/GPS).

User's Profile:
{user_profile}

Medical Knowledge Base Context (Use this to support your answer if relevant):
---
{medical_context}
---

You are currently in an interactive chat session with the user. 
Answer their questions clearly and concisely. 
If the Medical Knowledge Base Context contains the answer, YOU MUST base your answer strictly on that information.
Do NOT use flowery language. Be professional, clinical, and helpful.

DO NOT output your internal thinking process. ONLY output the final text response.
NO JSON ARRAYS. Please output plain text.

User's Query: {prompt}
"""
        response = chat_llm.invoke(context)

        content = response.content
        # 1. Handle case where content is already parsed into a list of dictionaries
        if isinstance(content, list):
            text_parts = [part["text"] for part in content if
                          isinstance(part, dict) and part.get("type") == "text" and "text" in part]
            if text_parts:
                return "\n\n".join(text_parts)
            return str(content)

        # 2. Handle string responses (with or without markdown JSON code blocks)
        if isinstance(content, str):
            content_stripped = content.strip()

            # Clean markdown code block wrapping
            if content_stripped.startswith('```json'):
                content_stripped = content_stripped.removeprefix('```json').removesuffix('```').strip()
            elif content_stripped.startswith('```'):
                content_stripped = content_stripped.removeprefix('```').removesuffix('```').strip()

            # Parse the JSON array if present
            if content_stripped.startswith('['):
                try:
                    parsed_content = json.loads(content_stripped)
                    if isinstance(parsed_content, list):
                        text_parts = [part["text"] for part in parsed_content if
                                      isinstance(part, dict) and part.get("type") == "text" and "text" in part]
                        if text_parts:
                            return "\n\n".join(text_parts)
                except json.JSONDecodeError:
                    pass

            return content_stripped

        return str(content)

    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        return f"I apologize, I encountered an error processing your request: {str(e)}"