from __future__ import annotations

import asyncio
import os

from app.classifier import classify_body, init_classifier

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8080"))
MAX_BODY_BYTES = 16 * 1024
HEADER_LIMIT = 32 * 1024

READY_BODY = b"OK"
READY_RESPONSE = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/plain\r\n"
    b"Content-Length: 2\r\n"
    b"Connection: keep-alive\r\n"
    b"\r\n"
    + READY_BODY
)
NOT_FOUND_RESPONSE = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
BAD_REQUEST_RESPONSE = (
    b"HTTP/1.1 400 Bad Request\r\n"
    b"Content-Type: application/json\r\n"
    b"Content-Length: 27\r\n"
    b"Connection: close\r\n"
    b"\r\n"
    b'{"error":"invalid_request"}'
)
TOO_LARGE_RESPONSE = b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

RESPONSE_PREFIX = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
RESPONSE_MID = b"\r\nConnection: keep-alive\r\n\r\n"


def _json_response(body: bytes) -> bytes:
    return RESPONSE_PREFIX + str(len(body)).encode("ascii") + RESPONSE_MID + body


JSON_RESPONSES = tuple(_json_response(body) for body in (
    b'{"approved":true,"fraud_score":0}',
    b'{"approved":true,"fraud_score":0.2}',
    b'{"approved":true,"fraud_score":0.4}',
    b'{"approved":false,"fraud_score":0.6}',
    b'{"approved":false,"fraud_score":0.8}',
    b'{"approved":false,"fraud_score":1}',
))


def _install_event_loop() -> None:
    if os.name == "nt":
        return
    try:
        import uvloop
    except Exception:
        return
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def _content_length(headers: list[bytes]) -> int:
    for line in headers:
        if line[:15].lower() == b"content-length:":
            return int(line[15:].strip())
    return 0


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            try:
                header_block = await reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                break
            except asyncio.LimitOverrunError:
                writer.write(BAD_REQUEST_RESPONSE)
                await writer.drain()
                break

            if len(header_block) > HEADER_LIMIT:
                writer.write(BAD_REQUEST_RESPONSE)
                await writer.drain()
                break

            lines = header_block.split(b"\r\n")
            if not lines or not lines[0]:
                break
            parts = lines[0].split(b" ")
            if len(parts) < 2:
                writer.write(BAD_REQUEST_RESPONSE)
                await writer.drain()
                break

            method, path = parts[0], parts[1]

            if method == b"GET" and path == b"/ready":
                writer.write(READY_RESPONSE)
                await writer.drain()
                continue

            if method == b"POST" and path == b"/fraud-score":
                length = _content_length(lines[1:])
                if length <= 0 or length > MAX_BODY_BYTES:
                    writer.write(TOO_LARGE_RESPONSE)
                    await writer.drain()
                    break

                body = await reader.readexactly(length)
                try:
                    writer.write(JSON_RESPONSES[classify_body(body)])
                except Exception:
                    writer.write(BAD_REQUEST_RESPONSE)
                    await writer.drain()
                    break
                await writer.drain()
                continue

            writer.write(NOT_FOUND_RESPONSE)
            await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    init_classifier()
    server = await asyncio.start_server(handle_client, HOST, PORT, limit=HEADER_LIMIT, backlog=4096)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    _install_event_loop()
    asyncio.run(main())
