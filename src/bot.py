import typing as t
import logging
import os
import tempfile
import json
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

assert os.getenv("BOT_TOKEN") is not None
assert os.getenv('LLM_URL') is not None

# Initialize bot and dispatcher
bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

SYSTEM_PROMPT = """
**You are Echo, a highly empathetic and supportive AI assistant.**

**Persona & Tone:** Your core mission is to be a reliable and friendly partner for the user. Maintain a warm, encouraging, and patient tone in every interaction. Your language should be clear, concise, and non-judgmental.

**Core Principles:**
1.  **Empathy First:** Acknowledge and validate the user's needs and feelings. Start by showing that you understand the request. Always answer in the same language user interacts with you.
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
        This is a LLM-based personal assistant bot
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


@dp.message(lambda message: message.text)
async def request_to_llm(message: Message):
    user_request = message.text

    url = os.getenv("LLM_URL")

    # Prepare variables
    full_response = ""
    telegram_message = None
    last_update_time = asyncio.get_event_loop().time()
    message_chunks = []
    current_chunk = ""
    MAX_MESSAGE_LENGTH = 4000  # Leave buffer for safety
    MIN_UPDATE_INTERVAL = 1.0  # Increase to 2 seconds to avoid rate limits
    
    def split_message_if_needed(text: str) -> list:
        """Split message into chunks if it's too long"""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]
        
        chunks = []
        current = ""
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if len(current + sentence) > MAX_MESSAGE_LENGTH:
                if current:
                    chunks.append(current.strip())
                    current = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for word in words:
                        if len(current + " " + word) > MAX_MESSAGE_LENGTH:
                            if current:
                                chunks.append(current.strip())
                                current = word
                            else:
                                # Single word is too long, truncate
                                chunks.append(word[:MAX_MESSAGE_LENGTH])
                        else:
                            current += " " + word if current else word
            else:
                current += " " + sentence if current else sentence
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    try:
        response = requests.post(
            url, 
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={
                # "model": "tinyllama",
                # "model": "llama2:7b",
                "model": "gemma3:1b",
                "messages": [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_request}
                ],
                "max_tokens": 6_000,
                "stream": True
            },
            stream=True,
            timeout=(10, 120)  # (connection_timeout, read_timeout)
        )
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            await message.reply(f"API Error {response.status_code}: {response.text}")
            return
            
        for line in response.iter_lines(decode_unicode=True, chunk_size=None):
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                
                if data_str.strip() == '[DONE]':
                    break
                    
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        
                        if content:
                            full_response += content
                            current_time = asyncio.get_event_loop().time()
                            
                            # Create message on first content chunk
                            if telegram_message is None and len(full_response.strip()) > 10:
                                # Wait a bit before creating first message
                                chunks = split_message_if_needed(full_response)
                                telegram_message = await message.answer(chunks[0])
                                last_update_time = current_time
                                
                                # Send additional chunks if needed
                                for chunk in chunks[1:]:
                                    await message.answer(chunk)
                                    await asyncio.sleep(0.1)  # Small delay between messages
                                    
                            elif telegram_message is not None:
                                # Check if it's time to update the Telegram message
                                if current_time - last_update_time >= MIN_UPDATE_INTERVAL:
                                    chunks = split_message_if_needed(full_response)
                                    
                                    try:
                                        # Update first message
                                        await bot.edit_message_text(
                                            chat_id=telegram_message.chat.id,
                                            message_id=telegram_message.message_id,
                                            text=chunks[0]
                                        )
                                        
                                        # Send additional chunks if response grew
                                        if len(chunks) > 1:
                                            for chunk in chunks[1:]:
                                                await message.answer(chunk)
                                                await asyncio.sleep(0.1)
                                                
                                        last_update_time = current_time
                                        
                                    except Exception as e:
                                        if "Flood control exceeded" in str(e):
                                            # Back off on rate limit
                                            MIN_UPDATE_INTERVAL = min(MIN_UPDATE_INTERVAL * 1.5, 10.0)
                                            logging.warning(f"Rate limit hit, increasing interval to {MIN_UPDATE_INTERVAL}s")
                                        elif "MESSAGE_TOO_LONG" in str(e):
                                            # Message is too long even after splitting, skip this update
                                            logging.warning("Message still too long after splitting")
                                        else:
                                            logging.error(f"Error editing Telegram message: {e}")
                                    
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    continue
                    
    except requests.exceptions.RequestException as e:
        await message.reply(f"Connection error: {e}")
        return
        
    # Final update to ensure the message is complete
    if telegram_message is not None and full_response:
        try:
            chunks = split_message_if_needed(full_response)
            
            await bot.edit_message_text(
                chat_id=telegram_message.chat.id,
                message_id=telegram_message.message_id,
                text=chunks[0]
            )
            
            # Send final additional chunks if needed
            for chunk in chunks[1:]:
                await message.answer(chunk)
                await asyncio.sleep(0.1)
                
        except Exception as e:
            if "MESSAGE_TOO_LONG" not in str(e) and "message is not modified" not in str(e):
                logging.error(f"Final edit failed: {e}")
            # The message is likely complete enough, so we can continue
            pass
    elif not telegram_message:
        # If no message was created (no content received), send error message
        await message.reply("No response received from LLM")


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