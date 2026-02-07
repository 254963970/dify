import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType
from typing import Any, Optional, cast
from urllib.parse import urlparse

from core.mcp.client.sse_client import sse_client
from core.mcp.client.streamable_client import streamablehttp_client
from core.mcp.error import MCPAuthError, MCPConnectionError
from core.mcp.session.client_session import ClientSession
from core.mcp.types import Tool

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(
        self,
        server_url: str,
        provider_id: str,
        tenant_id: str,
        authed: bool = True,
        authorization_code: Optional[str] = None,
        for_list: bool = False,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        sse_read_timeout: Optional[float] = None,
    ):
        # Initialize info
        self.provider_id = provider_id
        self.tenant_id = tenant_id
        self.client_type = "streamable"
        self.server_url = server_url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

        # Authentication info
        self.authed = authed
        self.authorization_code = authorization_code
        if authed:
            from core.mcp.auth.auth_provider import OAuthClientProvider

            self.provider = OAuthClientProvider(self.provider_id, self.tenant_id, for_list=for_list)
            self.token = self.provider.tokens()

        # Initialize session and client objects
        self._session: Optional[ClientSession] = None
        self._streams_context: Optional[AbstractContextManager[Any]] = None
        self._session_context: Optional[ClientSession] = None
        self._exit_stack = ExitStack()

        # Whether the client has been initialized
        self._initialized = False

    def __enter__(self):
        self._initialize()
        self._initialized = True
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[TracebackType]
    ):
        """
        Clean up resources when exiting the context manager.
        
        Ensures proper cleanup even if errors occur during the process.
        Logs all cleanup errors for debugging but doesn't suppress the original exception.
        """
        try:
            self.cleanup()
        except Exception as cleanup_error:
            # Log cleanup error but don't suppress the original exception
            logger.error("Error during context manager cleanup: %s", cleanup_error)
            # If there was no original exception, re-raise the cleanup error
            if exc_type is None:
                raise
            # Otherwise, log the cleanup error and let the original exception propagate

    def _initialize(
        self,
    ):
        """Initialize the client with fallback to SSE if streamable connection fails"""
        connection_methods: dict[str, Callable[..., AbstractContextManager[Any]]] = {
            "mcp": streamablehttp_client,
            "sse": sse_client,
        }

        parsed_url = urlparse(self.server_url)
        path = parsed_url.path or ""
        method_name = path.rstrip("/").split("/")[-1] if path else ""
        if method_name in connection_methods:
            client_factory = connection_methods[method_name]
            self.connect_server(client_factory, method_name)
        else:
            try:
                logger.debug("Not supported method %s found in URL path, trying default 'mcp' method.", method_name)
                self.connect_server(sse_client, "sse")
            except MCPConnectionError:
                logger.debug("MCP connection failed with 'sse', falling back to 'mcp' method.")
                self.connect_server(streamablehttp_client, "mcp")

    def connect_server(
        self, client_factory: Callable[..., AbstractContextManager[Any]], method_name: str, first_try: bool = True
    ):
        """
        Connect to MCP server using the specified transport method.
        
        Args:
            client_factory: Factory function to create the transport client
            method_name: Name of the transport method ('mcp' or 'sse')
            first_try: Whether this is the first connection attempt
            
        This method:
        1. Establishes the transport connection (SSE or StreamableHTTP)
        2. Creates and initializes the session
        3. Handles authentication errors with retry logic
        4. Properly manages context managers via ExitStack
        
        Raises:
            MCPConnectionError: If connection fails
            MCPAuthError: If authentication fails after retry
            ValueError: If authentication flow fails
        """
        from core.mcp.auth.auth_flow import auth

        try:
            headers = (
                {"Authorization": f"{self.token.token_type.capitalize()} {self.token.access_token}"}
                if self.authed and self.token
                else self.headers
            )
            self._streams_context = client_factory(
                url=self.server_url,
                headers=headers,
                timeout=self.timeout,
                sse_read_timeout=self.sse_read_timeout,
            )
            if not self._streams_context:
                raise MCPConnectionError("Failed to create connection context")

            # Use exit_stack to manage context managers properly
            # This ensures cleanup happens in reverse order (LIFO)
            if method_name == "mcp":
                read_stream, write_stream, _ = self._exit_stack.enter_context(self._streams_context)
                streams = (read_stream, write_stream)
            else:  # sse_client
                streams = self._exit_stack.enter_context(self._streams_context)

            self._session_context = ClientSession(*streams)
            self._session = self._exit_stack.enter_context(self._session_context)
            session = cast(ClientSession, self._session)
            session.initialize()
            return

        except MCPAuthError:
            if not self.authed:
                raise
            try:
                auth(self.provider, self.server_url, self.authorization_code)
            except Exception as e:
                raise ValueError(f"Failed to authenticate: {e}")
            self.token = self.provider.tokens()
            if first_try:
                return self.connect_server(client_factory, method_name, first_try=False)
        except Exception as e:
            # If connection fails, ensure partial cleanup before re-raising
            logger.error("Error during server connection: %s", e)
            raise

    def list_tools(self) -> list[Tool]:
        """Connect to an MCP server running with SSE transport"""
        # List available tools to verify connection
        if not self._initialized or not self._session:
            raise ValueError("Session not initialized.")
        response = self._session.list_tools()
        tools = response.tools
        return tools

    def invoke_tool(self, tool_name: str, tool_args: dict):
        """Call a tool"""
        if not self._initialized or not self._session:
            raise ValueError("Session not initialized.")
        return self._session.call_tool(tool_name, tool_args)

    def cleanup(self):
        """
        Clean up resources and ensure proper shutdown sequence.
        
        This method ensures:
        1. ExitStack properly closes all managed context managers in reverse order
        2. Session is cleaned up before streams
        3. All references are cleared to prevent resource leaks
        4. Errors during cleanup are logged but don't prevent full cleanup
        
        Note: This method is safe to call multiple times (idempotent).
        """
        cleanup_errors = []
        
        try:
            # ExitStack will handle proper cleanup of all managed context managers
            # in reverse order of their entry (LIFO)
            self._exit_stack.close()
        except Exception as e:
            cleanup_error = f"Error during exit stack cleanup: {e}"
            logging.exception(cleanup_error)
            cleanup_errors.append(cleanup_error)
        finally:
            # Clear all references to ensure garbage collection
            # even if errors occurred during cleanup
            self._session = None
            self._session_context = None
            self._streams_context = None
            self._initialized = False
        
        # If there were any errors during cleanup, raise a consolidated error
        if cleanup_errors:
            error_msg = "; ".join(cleanup_errors)
            raise ValueError(f"Errors during cleanup: {error_msg}")
