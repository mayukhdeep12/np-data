"""
mcp_manager.py
==============
Manages a persistent background asyncio event loop for MCP connections.
Compatible with langchain-mcp-adapters >= 0.1.0
"""

import asyncio
import threading
from typing import Any, Coroutine

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPManager:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._client = None
        self._tools: list = []
        self._connected: bool = False

    def run_sync(self, coro: Coroutine, timeout: int = 60) -> Any:
        """Submit a coroutine to the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def connect(self, server_path: str) -> list:
        """Launch the MCP stdio server and load all tool definitions."""
        if self._connected:
            self.disconnect()

        self._tools = self.run_sync(self._connect_async(server_path), timeout=30)
        self._connected = True

        # Tag every MCP tool so the UI can distinguish it from built-in tools
        for t in self._tools:
            if not hasattr(t, "metadata") or t.metadata is None:
                t.metadata = {}
            t.metadata["source"] = "mcp"

        return self._tools

    async def _connect_async(self, server_path: str) -> list:
        # langchain-mcp-adapters >= 0.1.0: instantiate then call get_tools() directly
        self._client = MultiServerMCPClient({
            "banking": {
                "command": "python",
                "args": [server_path],
                "transport": "stdio",
            }
        })
        tools = await self._client.get_tools()
        return tools

    def disconnect(self):
        """Shut down the MCP client."""
        self._client    = None
        self._tools     = []
        self._connected = False

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tool_names(self) -> set:
        return {t.name for t in self._tools}

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop