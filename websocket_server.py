import asyncio
import websockets
import json

# Define the WebSocket server address and port
HOST = 'localhost'
PORT = 8765

async def send_message(websocket):
    """
    Reads messages from the terminal and sends them to the connected client.
    """
    print(f"WebSocket connection established on ws://{HOST}:{PORT}")
    print("Type messages in the terminal and press Enter to send them.")

    try:
        while True:
            # Read input from the terminal
            message_content = await asyncio.get_event_loop().run_in_executor(
                None, input, ""
            )

            # Format the message into the required JSON payload
            payload = {
                "role": "system",
                "content": message_content
            }

            # Send the JSON payload to the connected client
            await websocket.send(json.dumps(payload))
            print(f"Sent message: {payload}")

    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client disconnected with error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    """
    Starts the WebSocket server.
    """
    # Start the WebSocket server
    async with websockets.serve(send_message, HOST, PORT):
        print(f"WebSocket server started on ws://{HOST}:{PORT}")
        # Keep the server running indefinitely
        await asyncio.Future()

if __name__ == "__main__":
    # Run the main function to start the server
    asyncio.run(main())
