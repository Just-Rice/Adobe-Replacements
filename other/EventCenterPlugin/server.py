#!/usr/bin/env python3

import aioredis
import asyncio
import redis
import sys
import toml
import websockets

from aioredis.pubsub import Receiver
from aioredis.abc import AbcChannel

CHANNELS = [
  # Text to speech
  'tts',
  'say',
  'vocode',

  # Object generation
  'spawn',

  # Teleport
  'warp',

  # Misc
  'testing',
]

print('reading secrets', flush=True)
SECRETS = toml.load("secrets.toml")

print('defining server', flush=True)

async def server(websocket, path):
    print('subscribing to redis pubsub', flush=True)

    queue = asyncio.Queue()

    mpsc = Receiver(loop=loop)
    async def reader(mpsc):
        print('foo', flush=True)
        async for channel, msg in mpsc.iter():
            print('bar', flush=True)
            assert isinstance(channel, AbcChannel)
            print("Got {!r} in channel {!r}".format(msg, channel))

            #print('sending message over websocket we got from redis pubsub', flush=True)
            #await websocket.send("Got message: {}".format(msg))
            
            channel_name = channel.name.decode('utf-8')
            message = msg.decode('utf-8')

            print('enqueuing websocket message', flush=True)
            payload = "{}|{}".format(channel_name, message)
            print('payload: {}'.format(payload, flush=True))
            queue.put_nowait(payload)

    asyncio.ensure_future(reader(mpsc))

    redis = await aioredis.create_redis_pool(SECRETS['redis_url'])

    channels = [mpsc.channel(ch) for ch in CHANNELS]
    await redis.subscribe(*channels)

    print('starting websocket server', flush=True)
    try:
        while True:
            # TODO: Do we need to ping to keepalive?
            if not queue.empty():
                msg = queue.get_nowait() # TODO THROWS EXCEPTION IF EMPTY (data race?)

                print('sending pubsub msg over websocket: {}'.format(msg), flush=True)
                await websocket.send(msg)
            else:
                await asyncio.sleep(1)

    except websockets.exceptions.ConnectionClosed:
        print('Connection Closed!')

loop = asyncio.get_event_loop()
loop.set_debug(True)

ws_server = websockets.serve(server, 'localhost', 8765)
loop.run_until_complete(ws_server)
loop.run_forever()

