import asyncio
from cgitb import handler
import websockets
import threading
import streamlit as st

class WebSockets():
    def __init__(self ,messages = []):
        self.MESSAGES = messages
        
    # Funzione handler per il server WebSocket
    async def handler(self,websocket, path):
        print("Client connected")
        try:
            async for message in websocket:
                print(f"Received message: {message}")
                print(f"Received message len: {len(message)}")
                self.MESSAGES.append(message)
                await websocket.send(1)
        except websockets.ConnectionClosed:
            print("Client disconnected")
    
    # Funzione principale per avviare il server WebSocket
    async def main(self):
        async with websockets.serve(self.handler, "localhost", 8765):
            await asyncio.Future()
    
    # Funzione per avviare il server WebSocket in un thread separato
    def start_websocket_server(self):
        asyncio.run(self.main())




