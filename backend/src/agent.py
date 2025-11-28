import logging
import json
import os
import datetime
from typing import List, Dict, Optional

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

class ShoppingAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and helpful voice assistant for a grocery store.
            Your goals are to help the user browse the catalog, add items to their shopping cart, and place an order.
            
            Capabilities:
            - You have access to a catalog of Groceries, Snacks, and Prepared Foods.
            - You can add single items or quantities to the cart.
            - **Intelligent Assistance:** If a user asks for "ingredients for a sandwich" or "pasta dinner", use the `add_recipe_bundle` tool to add all necessary items at once.
            - You can remove items or clear the cart.
            - You can list the cart contents and total price.
            
            Conversation Style:
            - Be polite, concise, and helpful.
            - When you add an item, confirm the item name and the new cart total.
            - If a user says "that's all" or "place order", verify the contents one last time and then call `place_order`.
            - Do not use markdown formatting (like asterisks or bolding) in your spoken responses.
            """,
        )
        self.cart: List[Dict] = []
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> List[Dict]:
        """Loads the product catalog from the JSON file."""
        # Calculates path relative to this script file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        catalog_path = os.path.join(base_dir, "catalog.json")
        
        try:
            with open(catalog_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Catalog file not found at {catalog_path}")
            return []

    def _find_item(self, name_query: str) -> Optional[Dict]:
        """Helper to find an item in the catalog by fuzzy name matching."""
        name_query = name_query.lower()
        for item in self.catalog:
            if name_query in item["name"].lower():
                return item
        return None

    def _calculate_total(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self.cart)

    @function_tool
    async def get_cart_details(self, context: RunContext) -> str:
        """
        Returns a summary of the current items in the cart and the total price.
        Use this when the user asks "what is in my cart?" or before placing an order.
        """
        if not self.cart:
            return "Your cart is currently empty."
        
        summary = "Cart Contents:\n"
        for item in self.cart:
            summary += f"- {item['quantity']}x {item['name']} (${item['price'] * item['quantity']:.2f})\n"
        
        total = self._calculate_total()
        summary += f"\nTotal: ${total:.2f}"
        return summary

    @function_tool
    async def add_to_cart(self, context: RunContext, item_name: str, quantity: int = 1) -> str:
        """
        Adds a specific item to the shopping cart.
        
        Args:
            item_name: The name of the product to add (e.g., "milk", "bread").
            quantity: The number of units to add (default is 1).
        """
        product = self._find_item(item_name)
        
        if not product:
            return f"I'm sorry, I couldn't find '{item_name}' in our catalog. We have items like Bread, Milk, Eggs, Pizza, and Snacks."

        # Check if item already in cart, if so, update quantity
        for cart_item in self.cart:
            if cart_item["id"] == product["id"]:
                cart_item["quantity"] += quantity
                new_total = self._calculate_total()
                return f"Updated. You now have {cart_item['quantity']} {product['name']}s. Cart total is ${new_total:.2f}."

        # Add new item
        new_item = product.copy()
        new_item["quantity"] = quantity
        self.cart.append(new_item)
        
        new_total = self._calculate_total()
        return f"Added {quantity} {product['name']} to your cart. The new total is ${new_total:.2f}."

    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str) -> str:
        """
        Removes an item from the cart.
        """
        product = self._find_item(item_name)
        if not product:
            return "I couldn't identify that item to remove it."

        for i, cart_item in enumerate(self.cart):
            if cart_item["id"] == product["id"]:
                removed = self.cart.pop(i)
                new_total = self._calculate_total()
                return f"Removed {removed['name']} from your cart. New total is ${new_total:.2f}."
        
        return f"{item_name} was not in your cart."

    @function_tool
    async def add_recipe_bundle(self, context: RunContext, recipe_type: str) -> str:
        """
        Intelligently adds multiple ingredients to the cart based on a recipe or meal request.
        Supports: "sandwich", "pasta", "snack pack".
        
        Args:
            recipe_type: The type of meal the user wants ingredients for (e.g. "sandwich", "pasta").
        """
        recipe_type = recipe_type.lower()
        added_items = []
        
        # Logic to map recipes to catalog items
        items_to_add = []
        
        if "sandwich" in recipe_type:
            # Add Bread, Peanut Butter, Jam
            items_to_add = ["Whole Wheat Bread", "Peanut Butter", "Strawberry Jam"]
        elif "pasta" in recipe_type:
            # Add Pasta, Sauce, Cheese
            items_to_add = ["Spaghetti Pasta", "Tomato Basil Sauce", "Cheddar Cheese"]
        elif "snack" in recipe_type:
            items_to_add = ["Sea Salt Potato Chips", "Dark Chocolate Bar"]
        else:
            return "I currently only know recipes for Sandwiches, Pasta, and Snack Packs. Would you like to try one of those?"

        # Process the addition
        for name in items_to_add:
            product = self._find_item(name)
            if product:
                # Add 1 of each
                found_in_cart = False
                for cart_item in self.cart:
                    if cart_item["id"] == product["id"]:
                        cart_item["quantity"] += 1
                        found_in_cart = True
                        break
                if not found_in_cart:
                    new_item = product.copy()
                    new_item["quantity"] = 1
                    self.cart.append(new_item)
                added_items.append(product["name"])

        total = self._calculate_total()
        return f"I've added the ingredients for {recipe_type} to your cart: {', '.join(added_items)}. Your total is now ${total:.2f}."

    @function_tool
    async def place_order(self, context: RunContext) -> str:
        """
        Finalizes the purchase, saves the order to a JSON file, and clears the cart.
        Call this when the user says "checkout", "place order", or "that's all".
        """
        if not self.cart:
            return "Your cart is empty, so I cannot place an order yet."

        total = self._calculate_total()
        timestamp = datetime.datetime.now().isoformat()
        
        # Create order object
        order_data = {
            "order_id": f"ORD-{int(datetime.datetime.now().timestamp())}",
            "timestamp": timestamp,
            "items": self.cart,
            "total_amount": total,
            "status": "placed"
        }

        # Ensure orders directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        orders_dir = os.path.join(base_dir, "orders")
        os.makedirs(orders_dir, exist_ok=True)

        # Save to file
        filename = f"order_{order_data['order_id']}.json"
        filepath = os.path.join(orders_dir, filename)
        
        try:
            with open(filepath, "w") as f:
                json.dump(order_data, f, indent=2)
            
            # Clear cart after successful save
            self.cart = []
            return f"Success! Your order has been placed. The total was ${total:.2f}. Your order ID is {order_data['order_id']}. Thank you for shopping with us!"
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "I'm sorry, there was a technical issue saving your order. Please try again."

    async def on_error(self, context: RunContext, error: Exception):
        logger.error(f"Agent error occurred: {error}")
        await context.say("I encountered a technical glitch. Could you please repeat that?", allow_interruptions=True)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-Anisha", 
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
        agent=ShoppingAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))