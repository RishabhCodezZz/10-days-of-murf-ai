import logging
import json
import os
from typing import Annotated, Optional, List, Dict

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr-agent")
load_dotenv(".env.local")

# --- PATHS ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPANY_FILE = os.path.join(ROOT_DIR, "company_data.json")
PRODUCT_FILE = os.path.join(ROOT_DIR, "products.json")
LEADS_FILE = os.path.join(ROOT_DIR, "leads.json")

# --- DATA LOADING HELPER ---
def load_json_file(filepath: str) -> any:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    return None

# Load initial data
COMPANY_DATA = load_json_file(COMPANY_FILE) or {}
PRODUCT_DATA = load_json_file(PRODUCT_FILE) or []

# Construct Knowledge Base String for the LLM
KNOWLEDGE_BASE = f"""
COMPANY PROFILE:
Name: {COMPANY_DATA.get('name', 'The Health Factory')}
Mission: {COMPANY_DATA.get('mission', 'Healthy bread for everyone.')}
About: {COMPANY_DATA.get('about', '')}

FREQUENTLY ASKED QUESTIONS:
{json.dumps(COMPANY_DATA.get('faqs', []), indent=2)}

PRODUCT CATALOG:
{json.dumps(PRODUCT_DATA, indent=2)}
"""

# --- AGENT CLASS ---

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""You are 'Alisha', a friendly and professional Sales Development Representative (SDR) for 'The Health Factory'.
            
            YOUR GOAL:
            1. Answer user questions about our healthy, zero-maida breads using the KNOWLEDGE BASE below.
            2. Qualify the user by collecting lead information naturally during the conversation.
            
            KNOWLEDGE BASE:
            {KNOWLEDGE_BASE}

            QUALIFICATION FIELDS TO COLLECT:
            - Name
            - Use Case (Personal health, For kids, Gym/Fitness, or Business/Cafe)
            - Product Interest (Which bread caught their eye?)
            - Timeline (Buying now, browsing, or monthly subscription)

            CONVERSATION FLOW:
            1. Greet cleanly: "Hi! Welcome to The Health Factory. I'm Alisha. I can help you find the perfect healthy bread. What brings you here today?"
            2. Answer questions briefly using the FAQ/Catalog.
            3. If they ask about price/products, use 'lookup_product' to give details.
            4. weave in qualification questions: "By the way, is this for your personal diet or for your family?"
            5. When the user indicates they are done or satisfied, call 'save_lead' to capture their details.
            
            TONE:
            - Energetic, helpful, and Indian context-aware (mentioning pav bhaji, chai time is okay).
            - Don't be pushy. Be a consultant.
            """,
        )
        # Internal session reference for sending data messages
        self._current_session: Optional[AgentSession] = None

    async def send_frontend_message(self, message: str):
        """Sends a text/json message to the frontend via LiveKit data channel."""
        if self._current_session and self._current_session.room and self._current_session.room.local_participant:
            try:
                await self._current_session.room.local_participant.publish_data(
                    payload=json.dumps({"message": message}).encode("utf-8"),
                    topic="chat"
                )
                logger.info(f"Sent chat message: {message}")
            except Exception as e:
                logger.error(f"Failed to send chat message: {e}")

    @function_tool
    async def lookup_product(self, ctx: RunContext, product_name: str):
        """Search for specific product details (price, weight, features) by name."""
        query = product_name.lower()
        # Simple fuzzy search
        found = [p for p in PRODUCT_DATA if query in p['name'].lower()]
        
        if found:
            p = found[0]
            # Send a nice visual summary to the frontend chat
            msg = f"🍞 **{p['name']}**\n💰 {p['price']} ({p['weight']})\n✨ {', '.join(p.get('features', []))}"
            await self.send_frontend_message(msg)
            return json.dumps(found)
        
        return "Product not found. We have Zero Maida Whole Wheat, Multi-Protein, Vegan Protein, and Multigrain breads."

    @function_tool
    async def save_lead(
        self, 
        ctx: RunContext,
        name: Annotated[str, "The user's name"],
        use_case: Annotated[str, "Why they want the bread (e.g., Health, Business, Kids)"],
        interest: Annotated[str, "Which product they liked most"],
        timeline: Annotated[str, "When they want to buy (Now, Later)"],
        summary: Annotated[str, "A brief 1-sentence summary of the call"]
    ):
        """
        Call this tool to save the lead information when the conversation wraps up.
        """
        logger.info(f"Saving lead: {name}")
        
        lead_data = {
            "name": name,
            "use_case": use_case,
            "interest": interest,
            "timeline": timeline,
            "summary": summary
        }

        # Persist to JSON
        leads = []
        if os.path.exists(LEADS_FILE):
            try:
                with open(LEADS_FILE, "r") as f:
                    leads = json.load(f)
            except json.JSONDecodeError:
                leads = []
        
        leads.append(lead_data)
        
        try:
            with open(LEADS_FILE, "w") as f:
                json.dump(leads, f, indent=2)
            
            # Confirm to user visually
            await self.send_frontend_message(f"✅ **Lead Captured**\nName: {name}\nInterest: {interest}\nSummary: {summary}")
            return "Lead saved successfully. Thank the user and wish them good health!"
        except Exception as e:
            logger.error(f"Failed to save lead: {e}")
            return "Error saving lead, but I have noted the details."


# --- ENTRYPOINT ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting SDR Agent for room {ctx.room.name}")

    # Setup the session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        # Use an energetic female voice for 'Alisha'
        tts=murf.TTS(
            voice="en-US-alicia", 
            style="Promo", 
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    agent = SDRAgent()
    # Pass session reference for chat tools
    agent._current_session = session

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))