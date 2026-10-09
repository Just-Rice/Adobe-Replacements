#!/usr/bin/env python3

import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

for i in range(1000):
    message = 'message: {}'.format(i)
    redis_client.publish('testing', message)

