import asyncio
from socket import socket
from typing import Callable
import websockets
import threading
import streamlit as st
from Services.WebSoket import WebSockets as scket

MESSAGES = []

# # Funzione handler per il server WebSocket
# async def handler(websocket, path):
#     print("Client connected")
#     try:
#         async for message in websocket:
#             print(f"Received message: {message}")
#             print(f"Received message len: {len(message)}")
#             await websocket.send(f"Echo: {message}")
#     except websockets.ConnectionClosed:
#         print("Client disconnected")

# # Funzione principale per avviare il server WebSocket
# async def main():
#     async with websockets.serve(handler, "localhost", 8765):
#         await asyncio.Future()

# Funzione per avviare il server WebSocket in un thread separato
def start_websocket_server(function:Callable):
    asyncio.run(function)

# # Funzione per il client WebSocket che riceve messaggi
# async def websocket_receiver():
#     try:
#         async with websockets.connect('ws://localhost:8765') as websocket:
#             while True:
#                 message = await websocket.recv()
#                 MESSAGES.append(message)
#                 st.experimental_rerun()
#     except websockets.ConnectionClosed:
#         st.write("WebSocket connection closed")
#     except Exception as e:
#         st.write(f"An error occurred: {e}")

# Avvia il server WebSocket in un thread separato
if st.button('websocket_server_thread'):
    try:
         if 'websocket_server_thread' not in st.session_state:
            st.session_state.websocket_server_thread = None
            
         if 'soket' not in st.session_state:
             sk = scket()
             st.session_state.soket = sk
         # funt = start_websocket_server(st.session_state.soket.main()) 
         # st.session_state.websocket_server_thread = threading.Thread(target=funt, daemon=True)
         # st.session_state.websocket_server_thread.start()
    except Exception as e:
        print(e)
        
if st.button('Clear sket'):
    st.session_state.pop('soket')
   

# # Inizializza la lista dei messaggi se non esiste
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # Avvia il client WebSocket in un thread separato
# if 'receiver_thread' not in st.session_state:
#     async def start_receiver():
#         await websocket_receiver()
#     st.session_state.receiver_thread = threading.Thread(target=asyncio.run, args=(start_receiver(),), daemon=True)
#     st.session_state.receiver_thread.start()

# Interfaccia utente di Streamlit
st.title("WebSocket Server Status Checker")

if st.button('Receive'):
    if 'soket' in st.session_state:
        st.write(dir(st.session_state.soket))
    else:
        st.write('soket not in')

# st.write("Real-time data from WebSocket:")
# for message in st.session_state.messages:
#     st.write(message)

async def check_websocket_connection():
    try:
        async with websockets.connect('ws://localhost:8765'):
            return True
    except:
        return False

def run_check():
    result = asyncio.run(check_websocket_connection())
    st.session_state.websocket_status = result

if 'websocket_status' not in st.session_state:
    st.session_state.websocket_status = None

if st.button("Check WebSocket Server Status"):
    run_check()

if st.session_state.websocket_status is not None:
    if st.session_state.websocket_status:
        st.write("WebSocket Server is running: True")
    else:
        st.write("WebSocket Server is running: False")
