"""
WebSocket 连接管理器
管理所有前端连接，支持广播主动消息
"""
import asyncio
import json
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"[WS] 新连接，当前 {len(self._connections)} 个")

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(f"[WS] 断开，剩余 {len(self._connections)} 个")

    async def broadcast(self, message: dict):
        """广播消息到所有前端连接"""
        if not self._connections:
            return
        data = json.dumps(message, ensure_ascii=False)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def has_clients(self) -> bool:
        return len(self._connections) > 0


# 全局单例
ws_manager = ConnectionManager()
