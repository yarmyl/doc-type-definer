#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import sys

sock = socket.socket()
sock.connect(('localhost', 8090))
print(sys.argv[1])
with open(sys.argv[1], 'rb') as f:
    sock.sendfile(f)
    sock.send(b'DONE!')
    data = sock.recv(1024)
    print(data.decode("utf-8"))
sock.close()
