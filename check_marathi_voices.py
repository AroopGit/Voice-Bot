import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    mr_voices = [v for v in voices if v['Locale'].startswith('mr-IN')]
    for v in mr_voices:
        print(f"Name: {v['ShortName']}, Gender: {v['Gender']}")

if __name__ == "__main__":
    asyncio.run(main())
