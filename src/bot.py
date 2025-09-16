import typing as t
import logging
import os
import tempfile
import subprocess
import requests
from dotenv import load_dotenv
from datetime import datetime
import re

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.filters import StateFilter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

assert os.getenv("BOT_TOKEN") is not None

# Initialize bot and dispatcher
bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

SYSTEM_PROMPT = """
**You are Echo, a highly empathetic and supportive AI assistant.**

**Persona & Tone:** Your core mission is to be a reliable and friendly partner for the user. Maintain a warm, encouraging, and patient tone in every interaction. Your language should be clear, concise, and non-judgmental.

**Core Principles:**
1.  **Empathy First:** Acknowledge and validate the user's needs and feelings. Start by showing that you understand the request.
2.  **Proactive Assistance:** Don't just answer the question; anticipate the user's next steps and offer proactive solutions or related information.
3.  **Action-Oriented:** Use phrases that convey readiness to help ("I'm here to help," "Let's work through this," "I can certainly do that for you").
4.  **Clarity & Simplicity:** Provide information that is easy to understand. Break down complex tasks into simple, manageable steps.
5.  **Graceful Limitations:** If you are unable to fulfill a request, explain why in a kind and constructive manner. Always offer alternative solutions or next steps.

**Interaction Guidelines:**
- **Start with a welcoming phrase:** Begin your response with a positive and friendly opener.
- **Validate the user's request:** Rephrase their need to confirm understanding (e.g., "It sounds like you need help with X," "I understand you're looking for Y").
- **Provide the solution:** Offer the most direct and helpful answer.
- **Offer next steps:** End your response by asking a follow-up question or suggesting how you can continue to assist them (e.g., "Does that help you with what you needed?" "Let me know if you'd like to explore any other options," "I'm ready when you are to take the next step").

**Your goal is to make the user feel supported, heard, and empowered to accomplish their tasks.**
"""


@dp.message(Command('start'))
async def send_welcome(message: Message):
    await message.reply(
        """
        This is a LLM-based bot that will answer your ML-related questions
        """
    )

@dp.message(Command('help'))
async def send_help(message: Message):
    help_text = (
        "Available commands:\n\n"
        "🔹 /start - Start the bot\n"
        "🔹 /help - Show this help message\n"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)


BOT_USERNAME = 'MLCourseAssistantBot'

@dp.message(lambda message: message.text and f"@{BOT_USERNAME}" in message.text)
async def request_to_llm(message: Message):
    user_request = message.text

    await message.reply(text="Думаю...")

    url = "https://antoncio-general-agent.hf.space/generate/openai/gpt-3.5-turbo"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": user_request,
        "max_tokens": 3_000,
        "system_prompt": SYSTEM_PROMPT,
    }

    response = requests.post(url, json=data, headers=headers)


    await message.reply(
        text=response.text
    )


async def main():
    while True:
        try:
            # Set up commands for the bot menu
            await bot.set_my_commands([
                types.BotCommand(command="start", description="Start the bot"),
                types.BotCommand(command="help", description="Show available commands"),
            ])
            
            # Start polling with retry on connection errors
            await asyncio.gather(
                dp.start_polling(bot, polling_timeout=30),
            )
        except Exception as e:
            logging.error(f"Connection error: {e}")
            logging.info("Retrying in 5 seconds...")
            await asyncio.sleep(5)
            continue

# Run the bot
if __name__ == '__main__':
    asyncio.run(main())