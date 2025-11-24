import logging
import json
import os
from datetime import datetime
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

LOG_FILE = "wellness_log.json"

def get_last_checkin_context():
    """Reads the JSON log and returns a summary string of the last session."""
    if not os.path.exists(LOG_FILE):
        return "This is the user's first check-in. Welcome them warmly."
    
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
            if not data:
                return "This is the user's first check-in. Welcome them warmly."
            
            # Get the last entry
            last_entry = data[-1]
            date_str = last_entry.get("timestamp", "Unknown date")
            mood = last_entry.get("mood", "Unknown")
            goals = ", ".join(last_entry.get("objectives", []))
            
            return f"""
            CONTEXT FROM PREVIOUS SESSION:
            - Date: {date_str}
            - User's Last Mood: {mood}
            - User's Last Goals: {goals}
            
            Start the conversation by referencing this past info naturally (e.g., "Last time we spoke, you were feeling... how are things today?").
            """
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return "Could not retrieve past history due to an error."

class WellnessCompanion(Agent):
    def __init__(self, past_context: str) -> None:
        super().__init__(
            instructions=f"""You are a supportive, grounded Health & Wellness Voice Companion.
            Your goal is to conduct a short daily check-in with the user.
            
            {past_context}

            BEHAVIOR GUIDELINES:
            1. **Check-In:** Ask about their mood and energy levels (e.g., "How is your energy today?").
            2. **Objectives:** Ask for 1-3 simple, practical goals for the day (e.g., "What's one thing you want to get done?").
            3. **Support:** Offer short, grounded advice (e.g., "Remember to take a 5-minute break").
               - DO NOT give medical diagnoses or clinical advice. You are a companion, not a doctor.
            4. **Recap & Save:** Once you have the mood, energy, and goals, summarize them back to the user to confirm.
               - IF the user confirms, you MUST call the 'log_daily_checkin' tool to save the entry.
            
            Tone: Warm, encouraging, concise, and realistic. Avoid toxic positivity.
            """
        )

    @function_tool
    async def log_daily_checkin(
        self, 
        ctx: RunContext, 
        mood: Annotated[str, "The user's self-reported mood"],
        energy: Annotated[str, "The user's described energy level (e.g., High, Low, Tired)"],
        objectives: Annotated[List[str], "A list of 1-3 goals the user mentioned for the day"],
        summary: Annotated[str, "A brief one-sentence summary of the check-in"]
    ):
        """
        Call this tool ONLY after the user has confirmed their check-in details (mood, goals, etc.).
        """
        logger.info(f"Logging check-in: {mood}, {objectives}")

        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "energy": energy,
            "objectives": objectives,
            "summary": summary
        }

        # Read existing data to append
        history = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = [] # Start fresh if corrupt

        history.append(new_entry)

        # Write back to file
        try:
            with open(LOG_FILE, "w") as f:
                json.dump(history, f, indent=2)
            return "Check-in saved successfully! You can now wish the user a great day and say goodbye."
        except Exception as e:
            logger.error(f"Failed to save check-in: {e}")
            return "There was an error saving the check-in."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load history before the agent starts to inject it into the prompt
    past_context_str = get_last_checkin_context()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Initialize the agent with the history context
    agent = WellnessCompanion(past_context=past_context_str)

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