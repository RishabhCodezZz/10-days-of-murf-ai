import logging

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

import json
from typing import Literal

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# MOCK DATABASE: Multiple Fraud Cases (Total 20)
# Key is normalized to lowercase first name for simple lookup.
FRAUD_CASES = {
    # Existing Cases (Shadow, Luna, Ravi)
    "shadow": { 
        "case_id": "FRD-9876",
        "customer_name": "Shadow",
        "security_identifier": "ID-421A",
        "masked_card": "**** 9012",
        "transaction_amount": "452.99",
        "merchant_name": "ElectroGadget Inc.",
        "location": "New Delhi, India",
        "timestamp": "Nov 26, 2025, 2:30 PM IST",
        "security_question": "What city were you born in?",
        "security_answer": "surat", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "luna": {
        "case_id": "FRD-1024",
        "customer_name": "Luna",
        "security_identifier": "ID-555B",
        "masked_card": "**** 4321",
        "transaction_amount": "1,200.00",
        "merchant_name": "SkyTravel Agency",
        "location": "New York, USA",
        "timestamp": "Nov 26, 2025, 8:00 AM EST",
        "security_question": "What is the name of your first pet?",
        "security_answer": "mittens", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "ravi": {
        "case_id": "FRD-7777",
        "customer_name": "Ravi Sharma",
        "security_identifier": "ID-300C",
        "masked_card": "**** 6789",
        "transaction_amount": "150.50",
        "merchant_name": "Local Grocery Store",
        "location": "Mumbai, India",
        "timestamp": "Nov 25, 2025, 7:15 PM IST",
        "security_question": "What is the last four digits of your registered phone number?",
        "security_answer": "5432", 
        "status": "pending_review",
        "outcome_note": ""
    },
    # Existing Cases (Gambit to Isagi)
    "gambit": { 
        "case_id": "FRD-3333",
        "customer_name": "Gambit LeBeau",
        "security_identifier": "ID-123G",
        "masked_card": "**** 2222",
        "transaction_amount": "250.00",
        "merchant_name": "Rare Card Emporium",
        "location": "New Orleans, USA",
        "timestamp": "Nov 26, 2025, 1:00 PM CST",
        "security_question": "What is your favorite color?",
        "security_answer": "black", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "dark": { 
        "case_id": "FRD-4444",
        "customer_name": "Dark Schneider",
        "security_identifier": "ID-456D",
        "masked_card": "**** 1111",
        "transaction_amount": "8000.00",
        "merchant_name": "Magical Artifacts Ltd.",
        "location": "Tokyo, Japan",
        "timestamp": "Nov 26, 2025, 10:00 AM JST",
        "security_question": "What is your birth month?",
        "security_answer": "august", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "naruto": { 
        "case_id": "FRD-5555",
        "customer_name": "Naruto Uzumaki",
        "security_identifier": "ID-789N",
        "masked_card": "**** 5555",
        "transaction_amount": "14.99",
        "merchant_name": "Ramen Shop Konoha",
        "location": "Los Angeles, USA",
        "timestamp": "Nov 26, 2025, 9:00 PM PST",
        "security_question": "What is your favorite food?",
        "security_answer": "ramen", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "jinwoo": { 
        "case_id": "FRD-6666",
        "customer_name": "Jinwoo Sung",
        "security_identifier": "ID-012J",
        "masked_card": "**** 6666",
        "transaction_amount": "5000.00",
        "merchant_name": "Hunter Association Gear",
        "location": "Seoul, South Korea",
        "timestamp": "Nov 26, 2025, 3:30 PM KST",
        "security_question": "What is your rank?",
        "security_answer": "s", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "rimaru": { 
        "case_id": "FRD-7778",
        "customer_name": "Rimaru Tempest",
        "security_identifier": "ID-345R",
        "masked_card": "**** 7777",
        "transaction_amount": "1500.00",
        "merchant_name": "Slime Labs Research",
        "location": "Singapore",
        "timestamp": "Nov 26, 2025, 5:00 PM SGT",
        "security_question": "What is your original name?",
        "security_answer": "satoru", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "noir": { 
        "case_id": "FRD-8888",
        "customer_name": "Noir",
        "security_identifier": "ID-678N",
        "masked_card": "**** 8888",
        "transaction_amount": "100.00",
        "merchant_name": "Assassin's Guild Supplies",
        "location": "London, UK",
        "timestamp": "Nov 26, 2025, 11:00 AM GMT",
        "security_question": "What is the last four digits of your social security number?",
        "security_answer": "9876", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "diablo": { 
        "case_id": "FRD-9999",
        "customer_name": "Diablo",
        "security_identifier": "ID-901D",
        "masked_card": "**** 9999",
        "transaction_amount": "666.00",
        "merchant_name": "Demonic Investments Corp",
        "location": "Frankfurt, Germany",
        "timestamp": "Nov 26, 2025, 2:00 PM CET",
        "security_question": "What is your true title?",
        "security_answer": "demon", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "luffy": { 
        "case_id": "FRD-1111",
        "customer_name": "Monkey D. Luffy",
        "security_identifier": "ID-234L",
        "masked_card": "**** 1010",
        "transaction_amount": "10.00",
        "merchant_name": "Meat Market Paradise",
        "location": "Paris, France",
        "timestamp": "Nov 26, 2025, 1:30 PM CET",
        "security_question": "What is your main goal?",
        "security_answer": "pirate king", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "goku": { 
        "case_id": "FRD-2222",
        "customer_name": "Son Goku",
        "security_identifier": "ID-567G",
        "masked_card": "**** 2020",
        "transaction_amount": "20.00",
        "merchant_name": "World Martial Arts",
        "location": "Toronto, Canada",
        "timestamp": "Nov 26, 2025, 4:00 PM EST",
        "security_question": "What is your first martial arts teacher's name?",
        "security_answer": "roshi", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "ichigo": { 
        "case_id": "FRD-3334",
        "customer_name": "Ichigo Kurosaki",
        "security_identifier": "ID-890I",
        "masked_card": "**** 3030",
        "transaction_amount": "300.00",
        "merchant_name": "Soul Society Gear",
        "location": "New York, USA",
        "timestamp": "Nov 26, 2025, 12:00 PM EST",
        "security_question": "What is your favorite drink?",
        "security_answer": "orange soda", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "asta": { 
        "case_id": "FRD-4445",
        "customer_name": "Asta",
        "security_identifier": "ID-112A",
        "masked_card": "**** 4040",
        "transaction_amount": "5.00",
        "merchant_name": "Clovers General Store",
        "location": "Milan, Italy",
        "timestamp": "Nov 26, 2025, 10:00 AM CET",
        "security_question": "What is the color of your cloak?",
        "security_answer": "black", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "isagi": { 
        "case_id": "FRD-5556",
        "customer_name": "Yoichi Isagi",
        "security_identifier": "ID-334Y",
        "masked_card": "**** 5050",
        "transaction_amount": "50.00",
        "merchant_name": "Blue Lock Football",
        "location": "Berlin, Germany",
        "timestamp": "Nov 26, 2025, 3:00 PM CET",
        "security_question": "What is your primary weapon?",
        "security_answer": "ego", 
        "status": "pending_review",
        "outcome_note": ""
    },
    # New Female Character Cases
    "hinata": { 
        "case_id": "FRD-6060",
        "customer_name": "Hinata Hyuga",
        "security_identifier": "ID-6060H",
        "masked_card": "**** 6060",
        "transaction_amount": "55.00",
        "merchant_name": "Ninja Tool Shop",
        "location": "Konoha, Japan",
        "timestamp": "Nov 26, 2025, 6:00 AM JST",
        "security_question": "What is your clan symbol?",
        "security_answer": "byakugan", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "shinobu": { 
        "case_id": "FRD-7070",
        "customer_name": "Shinobu Kocho",
        "security_identifier": "ID-7070S",
        "masked_card": "**** 7070",
        "transaction_amount": "150.00",
        "merchant_name": "Wisteria Pharmaceuticals",
        "location": "Kyoto, Japan",
        "timestamp": "Nov 26, 2025, 1:00 PM JST",
        "security_question": "What color is your hair?",
        "security_answer": "black", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "mitsuri": { 
        "case_id": "FRD-8080",
        "customer_name": "Mitsuri Kanroji",
        "security_identifier": "ID-8080M",
        "masked_card": "**** 8080",
        "transaction_amount": "25.00",
        "merchant_name": "Sweets and Tea House",
        "location": "Paris, France",
        "timestamp": "Nov 26, 2025, 4:00 PM CET",
        "security_question": "What is your favorite food?",
        "security_answer": "sakura mochi", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "makima": { 
        "case_id": "FRD-9090",
        "customer_name": "Makima",
        "security_identifier": "ID-9090K",
        "masked_card": "**** 9090",
        "transaction_amount": "900.00",
        "merchant_name": "Public Safety HQ",
        "location": "Berlin, Germany",
        "timestamp": "Nov 26, 2025, 11:00 AM CET",
        "security_question": "What is your true identity?",
        "security_answer": "control devil", 
        "status": "pending_review",
        "outcome_note": ""
    },
    "mikasa": { 
        "case_id": "FRD-1313",
        "customer_name": "Mikasa Ackerman",
        "security_identifier": "ID-1313A",
        "masked_card": "**** 1313",
        "transaction_amount": "300.00",
        "merchant_name": "ODM Gear Maintenance",
        "location": "London, UK",
        "timestamp": "Nov 26, 2025, 9:00 AM GMT",
        "security_question": "What is the color of your scarf?",
        "security_answer": "red", 
        "status": "pending_review",
        "outcome_note": ""
    },
}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an extremely precise and professional Fraud Detection Representative for OmniBank. Your single purpose is to resolve a single suspicious transaction with the customer.
            The user is interacting with you via voice.
            Your responses are concise, professional, reassuring, and completely free of any formatting including emojis or asterisks.
            
            Strict Call Flow:
            1. Introduce yourself as the OmniBank Fraud Department and state the reason for the call: "a suspicious transaction on your card".
            2. **IMMEDIATELY ask for the customer's full name to confirm their identity and look up their case.**
            3. Use the **`load_fraud_case`** tool with the provided name.
            4. If the case is loaded successfully, the tool's output will provide a **security question**. Ask this question immediately.
            5. Use the **`verify_security_answer`** tool with the customer's answer. **Do not attempt to check the answer yourself.**
            6. If the security verification passes (tool returns "Verification successful..."): Read the detailed transaction summary that the tool provided and clearly ask the customer: **"Did you make this transaction?" (A simple yes or no is required).**
            7. If the security verification fails (tool returns "Verification failed..."): Politely state that you cannot proceed due to failed verification and tell them to call the bank's main line, then disconnect/end the conversation.
            8. After receiving a 'yes' or 'no' answer in step 6, use the **`confirm_transaction`** tool with the user's final decision.
            9. The last response you provide to the user must be the **final action taken** returned by the `confirm_transaction` tool, and then you must say goodbye and end the conversation immediately. Do not deviate from this structured, professional call flow.
            """,
        )
        # FIX: Initializing a state dictionary directly on the Assistant instance (self)
        # to bypass the RunContext.run_state attribute error.
        self.user_session_data = {}

    @function_tool
    async def load_fraud_case(self, context: RunContext, username: str) -> str:
        """
        Loads the details of a single, pending fraud case for the given username. 
        You MUST call this tool immediately after getting the user's name.

        Args:
            username: The name of the customer to look up the fraud case for.
        
        Returns:
            A JSON string containing the case details and the security question, or an error message.
        """
        # Normalize and split all input words from the transcription for robust lookup
        input_words = [word.strip() for word in username.lower().split() if word.strip()]
        
        found_key = None
        # Check if any word in the user's spoken name matches a key in the database
        for word in input_words:
            if word in FRAUD_CASES:
                found_key = word
                break
        
        if found_key:
            user_key = found_key
            
            # We fetch the specific case for the user
            case = FRAUD_CASES[user_key].copy()
            
            # FIX: Storing state on the Assistant instance (self.user_session_data)
            self.user_session_data["current_user_key"] = user_key
            
            # Prepare data to be visible to the LLM for conversation framing
            details = {
                "customer_name": case["customer_name"],
                "transaction_amount": case["transaction_amount"],
                "merchant_name": case["merchant_name"],
                "masked_card": case["masked_card"],
                "location": case["location"],
                "timestamp": case["timestamp"],
                "security_question": case["security_question"]
            }
            
            return json.dumps({
                "status": "case_loaded",
                "message": f"Case loaded. Proceed to security question: '{details['security_question']}'",
                "case_details": details
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"I'm sorry, I could not find a pending fraud alert associated with the name '{username}'. To protect your security, I must end this call. Please call our main fraud line later."
            })
            
    @function_tool
    async def verify_security_answer(self, context: RunContext, user_response: str) -> str:
        """
        Tool for the agent to use to check the customer's response against the stored security answer.
        The agent MUST call this tool after receiving the user's security question response.
        
        Args:
            user_response: The answer provided by the user.
            
        Returns:
            A string indicating if verification passed or failed, and the transaction details if passed.
        """
        # FIX: Getting state from the Assistant instance
        user_key = self.user_session_data.get("current_user_key")
        
        if not user_key or user_key not in FRAUD_CASES:
            # FIX: Setting state on the Assistant instance
            self.user_session_data["verification_failed"] = True
            return "Internal Error: Unable to verify account details."
            
        case_details = FRAUD_CASES[user_key]
        # We store the security answer in the database in lowercase, so we normalize the user's response for comparison
        expected_answer = case_details["security_answer"].strip().lower()

        if user_response.strip().lower() == expected_answer:
            # Construct transaction details to be read out
            details = (
                f"a purchase of ${case_details.get('transaction_amount', 'an unknown amount')} "
                f"at {case_details.get('merchant_name', 'an unknown merchant')} "
                f"in {case_details.get('location', 'an unknown location')} "
                f"on {case_details.get('timestamp', 'an unknown date')} "
                f"using card number {case_details.get('masked_card', '**** ****')}"
            )
            return f"Verification successful. The suspicious transaction details are: {details}. **You must now ask the user if they made this transaction (yes/no).**"
        else:
            # Flag verification as failed in state to trigger hangup behavior
            # FIX: Setting state on the Assistant instance
            self.user_session_data["verification_failed"] = True
            return "Verification failed. We cannot proceed further with the verification process."

    @function_tool
    async def confirm_transaction(self, context: RunContext, is_legitimate: Literal["yes", "no"]) -> str:
        """
        Updates the fraud case status in the mock database and provides the final action to read back to the user.
        The agent MUST call this tool after the user confirms or denies the transaction.

        Args:
            is_legitimate: The customer's response to whether they made the transaction ('yes' or 'no').
            
        Returns:
            A string describing the final action taken.
        """
        # FIX: Getting state from the Assistant instance
        user_key = self.user_session_data.get("current_user_key")
        if not user_key or user_key not in FRAUD_CASES:
            return "I'm sorry, an issue occurred with your case details. Please call our main fraud line for assistance."
            
        case_id = FRAUD_CASES[user_key]["case_id"]
        status_to_update = "processing_error"
        outcome_note_to_update = "Processing failed."
        action_taken_message = "I'm sorry, an issue occurred while processing your request. Please call our main fraud line for assistance."
        
        if is_legitimate.lower() == "yes":
            status_to_update = "confirmed_safe"
            outcome_note_to_update = "Customer confirmed transaction as legitimate."
            action_taken_message = "Thank you. The transaction has been marked as legitimate and your card is safe to use."
        elif is_legitimate.lower() == "no":
            status_to_update = "confirmed_fraud"
            outcome_note_to_update = "Customer denied transaction. Card blocked and dispute raised (mock)."
            action_taken_message = "Thank you for confirming. The transaction has been marked as fraudulent. We have immediately blocked your card and initiated a dispute. A new card will be sent to you in 3-5 business days."
        
        # Mock database update
        FRAUD_CASES[user_key]["status"] = status_to_update
        FRAUD_CASES[user_key]["outcome_note"] = outcome_note_to_update
        logger.info(f"Updated Fraud Case {case_id} ({user_key}): Status: {status_to_update}, Note: {outcome_note_to_update}")
        
        # --- START: JSON Logging Block ---
        log_entry = {
            "case_id": case_id,
            "customer_name": FRAUD_CASES[user_key]["customer_name"],
            "security_identifier": FRAUD_CASES[user_key]["security_identifier"],
            "transaction_amount": FRAUD_CASES[user_key]["transaction_amount"],
            "merchant_name": FRAUD_CASES[user_key]["merchant_name"],
            "location": FRAUD_CASES[user_key]["location"],
            "timestamp": FRAUD_CASES[user_key]["timestamp"],
            "final_status": status_to_update,
            "outcome_note": outcome_note_to_update,
        }

        try:
            with open("logger.json", "a") as f:
                # Write the JSON log entry followed by a newline for JSON Lines format
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Successfully logged Case {case_id} outcome to logger.json")
        except Exception as e:
            logger.error(f"Error writing to logger.json: {e}")
        # --- END: JSON Logging Block ---

        # The LLM is instructed to read this message back and end the call.
        return action_taken_message

    # Override on_error to ensure the call ends gracefully
    async def on_error(self, context: RunContext, error: Exception):
        logger.error(f"Agent error occurred: {error}")
        await context.say("I am sorry, an internal error has occurred and I need to disconnect. Please call us back to resolve this issue.", allow_interruptions=True)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using the Gemini LLM
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice (Murf Falcon)
        tts=murf.TTS(
                voice="en-IN-Anisha", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection manage conversation flow
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection (omitted for brevity)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))