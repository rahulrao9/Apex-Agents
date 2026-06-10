# NMMO Replay Streamer
import asyncio
import websockets
import lzma
import json
import sys

async def handler(websocket):
    print("Unity Client connected! Initializing stream...")
    
    try:
        print("Decompressing and parsing faction_war.lzma...")
        with lzma.open("faction_war_match_v1.lzma", "rt") as f:
            # Load the complete raw JSON data structure
            full_data = json.loads(f.read())
    except Exception as e:
        print(f"Error reading or parsing LZMA file: {e}")
        return

    # Handle both common NMMO replay formats (list of frames or dictionary of ticks)
    frames = []
    if isinstance(full_data, list):
        frames = full_data
    elif isinstance(full_data, dict) and 'packets' in full_data:
        frames = full_data['packets']
    elif isinstance(full_data, dict):
        # Fallback for sorted dictionary ticks
        try:
            frames = [full_data[str(k)] for k in sorted(map(int, full_data.keys()))]
        except ValueError:
            frames = [full_data[k] for k in sorted(full_data.keys())]

    if not frames:
        print("Error: Could not extract frame packet structure from replay data.")
        return

    print(f"Successfully loaded {len(frames)} game ticks. Starting packet stream...")

    try:
        for idx, frame in enumerate(frames):
            # Send the individual tick packet
            await websocket.send(json.dumps(frame))
            
            if idx % 10 == 0:  # Print every 100 ticks instead to avoid terminal lag
                print(f"Streamed tick {idx}/{len(frames)}")
            
            # Reduce sleep to a tiny fraction so it blasts into the Unity buffer instantly
            await asyncio.sleep(0.3) 
            
        print("Finished streaming entire match replay.")
        
    except websockets.ConnectionClosed:
        print("Unity Client disconnected.")
    except Exception as e:
        print(f"Streaming error occurred: {e}")

async def main():
    print("Starting NMMO Replay Streamer on port 8080...")
    async with websockets.serve(handler, "localhost", 8080, ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())