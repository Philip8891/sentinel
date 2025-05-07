#!/usr/bin/env python3
import os
import time
import hashlib
import hmac
import base64
import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
API_URL = "https://api.kraken.com"

def generate_nonce():
    return str(int(time.time() * 1000))

def get_kraken_signature(url_path, data, secret):
    postdata = urllib.parse.urlencode(data)
    message = url_path.encode('utf-8') + hashlib.sha256((data['nonce'] + postdata).encode('utf-8')).digest()
    # Győződj meg róla, hogy a secret nem None, és kódoljuk UTF-8-ra
    if secret is None:
        raise ValueError("KRAKEN_API_SECRET is not set in the .env file.")
    secret_bytes = secret.encode('utf-8')
    signature = hmac.new(base64.b64decode(secret_bytes), message, hashlib.sha512).digest()
    return base64.b64encode(signature)

def execute_trade(symbol, volume, side):
    url_path = "/0/private/AddOrder"
    url = API_URL + url_path
    nonce = generate_nonce()
    data = {
        "nonce": nonce,
        "pair": symbol,
        "type": side,         # "buy" vagy "sell"
        "ordertype": "market",
        "volume": volume
    }
    headers = {
        "API-Key": API_KEY,
        "API-Sign": get_kraken_signature(url_path, data, API_SECRET).decode()
    }
    response = requests.post(url, data=data, headers=headers, timeout=10)
    return response.json()

if __name__ == "__main__":
    result = execute_trade("XBTUSD", "0.001", "buy")
    print("Kraken trade result:", result)
