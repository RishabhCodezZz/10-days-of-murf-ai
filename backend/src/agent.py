import logging
import json
from typing import Annotated, List

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

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # DAY 2 TASK: Define the Persona
            instructions="""You are a friendly and energetic barista at 'Code Brew Café'. 
            Your goal is to take a coffee order from the customer.
            
            You MUST collect the following information to complete an order:
            1. Drink Type (e.g., Latte, Cappuccino, Americano)
            2. Size (Small, Medium, Large)
            3. Milk Type (Whole, Oat, Almond, Soy)
            4. Extras (e.g., Sugar, Syrup, or "None")
            5. Customer Name

            Ask clarifying questions one by one or in small groups to fill these fields. 
            Do not assume any values (except strictly implied ones). 
            
            Once you have ALL the information, you MUST call the 'save_order' tool to submit the order.
            After saving, thank the customer by name and tell them their order is coming right up.
            
            Keep your responses conversational, short, and polite. 
            Do not use emojis or special formatting.""",
        )

    # DAY 2 TASK: Create the Tool to capture state and save JSON
    @function_tool
    async def save_order(
        self, 
        ctx: RunContext, 
        drink_type: Annotated[str, "The type of coffee drink (e.g., Latte, Espresso)"],
        size: Annotated[str, "The size of the drink (Small, Medium, Large)"],
        milk: Annotated[str, "The type of milk preference"],
        customer_name: Annotated[str, "The customer's name"],
        extras: Annotated[List[str], "A list of any extras like sugar or flavors"]
    ):
        """
        Call this tool ONLY when you have collected all details for the customer's order.
        """
        logger.info(f"Saving order for {customer_name}")

        # Create the order object structure required by the task
        order_data = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": customer_name
        }

        # Save to JSON file
        try:
            with open("order.json", "w") as f:
                json.dump(order_data, f, indent=2)
            return "Order saved successfully! You can now confirm to the user."
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "There was an error saving the order."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))