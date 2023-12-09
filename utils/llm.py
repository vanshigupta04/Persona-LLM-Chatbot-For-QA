import re
import asyncio
from collections import Counter
import aiohttp
import json
import requests

async def llm_completion(
        system_instruction,
        user_query,
        api_url=None,
        port=8080,
        max_new_tokens=256,
        top_p=0.9,
        temperature=0.7
    ):
    """
    Returns the response from the API call

    Args:
        system_instruction (str): The system instruction.
        user_query (str): The user query.
        api_url (str): The API URL.
        port (int): The port.

    Returns:
        dict: The response from the API call.
    """

    if api_url is None:
        api_url = f"http://localhost:{port}/completion"
    
    json_body = {
        "prompt": f"[INST] <<SYS>>{system_instruction}<<SYS>> {user_query} [/INST] ",
        "max_new_tokens":max_new_tokens, 
        "top_p":top_p, 
        "temperature":temperature
        }
    
    data = json.dumps(json_body)

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=data) as response:
            try:
                return await response.json()
            except:
                return await response.text()
