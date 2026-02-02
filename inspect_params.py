import inspect
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams

print("FastAPIWebsocketParams init signature:")
print(inspect.signature(FastAPIWebsocketParams.__init__))
