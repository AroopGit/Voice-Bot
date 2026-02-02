class TransportInput(FrameProcessor):
    def __init__(self, transport_name):
        super().__init__()
        self._transport_name = transport_name
        
    async def process_frame(self, frame: Frame, direction):
        await self.push_frame(frame, direction)

class TransportOutput(FrameProcessor):
    def __init__(self, transport):
        super().__init__()
        self._transport = transport
        
    async def process_frame(self, frame: Frame, direction):
        await self._transport.send_frame(frame, direction)
        await self.push_frame(frame, direction)

class SimpleWebsocketTransport(BaseTransport):
    def __init__(self, websocket, params: TransportParams = None):
        super().__init__(params)
        self._websocket = websocket
        self._input = TransportInput("simple_input")
        self._output = TransportOutput(self)

    async def start(self, frame_handler):
        self._handler = frame_handler
        try:
            while True:
                data = await self._websocket.receive_bytes()
                # Assuming PCM 16kHz 16-bit mono from client
                frame = InputAudioRawFrame(
                    audio=data,
                    sample_rate=16000,
                    num_channels=1
                )
                if self._handler:
                    await self._handler(frame)
                    
        except Exception as e:
            print(f"WebSocket transport input loop ended: {e}")
            if self._handler:
                await self._handler(EndFrame())

    async def send_frame(self, frame: Frame, direction):
        if isinstance(frame, OutputAudioRawFrame):
            try:
                # Client expects raw bytes
                await self._websocket.send_bytes(frame.audio)
            except Exception as e:
                print(f"Error sending audio: {e}")
                
        elif isinstance(frame, TextFrame):
            try:
                msg = json.dumps({"text": frame.text})
                await self._websocket.send_text(msg)
            except Exception as e:
                print(f"Error sending text: {e}")
        
        elif isinstance(frame, TranscriptionFrame):
             # Also send transcription to client for captions
            try:
                msg = json.dumps({"text": frame.text, "type": "transcription"})
                await self._websocket.send_text(msg)
            except Exception as e:
                print(f"Error sending transcription: {e}")
                
        elif isinstance(frame, EndFrame):
            try:
                await self._websocket.close()
            except:
                pass
                
    def input(self):
        return self._input

    def output(self):
        return self._output
