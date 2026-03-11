# Run this standalone with `python mcp_server.py` to expose research tools to any
# MCP-compatible client (Claude Desktop, Cursor, etc.). The server communicates over
# stdio, making it compatible with the MCP standard without any additional transport setup.

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types
from tools import search_arxiv, search_wikipedia, search_web

server = Server("multi-agent-research-tools")


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="search_arxiv",
            description=(
                "Search ArXiv for academic papers. "
                "CRITICAL INSTRUCTION: You MUST extract ONLY the core scientific or technical "
                "entities and omit subjective words like 'Market Growth'. "
                "Returns up to max_results papers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Strict scientific keywords..."},
                    "max_results": {"type": "integer", "description": "Number of papers to return (1-20)"}
                },
                "required": ["query"]
            }
        ),
        mcp_types.Tool(
            name="search_wikipedia",
            description=(
                "Retrieve a Wikipedia summary for a given topic. "
                "Returns approximately 7 sentences of encyclopedic background information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The subject or concept to look up on Wikipedia"
                    }
                },
                "required": ["topic"]
            }
        ),
        mcp_types.Tool(
            name="search_web",
            description=(
                "Search the web using DuckDuckGo for current information about a query. "
                "Returns up to 5 web results with title, snippet, and URL. "
                "Useful for finding recent news, blog posts, and practical information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or question to look up on the web"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    try:        
        if name == "search_arxiv":
            query = arguments.get("query", "")
            max_res = arguments.get("max_results", 5)
            result = search_arxiv(query, max_res)
            return [mcp_types.TextContent(type="text", text=result)]

        elif name == "search_wikipedia":
            topic = arguments.get("topic", "")
            result = search_wikipedia(topic)
            return [mcp_types.TextContent(type="text", text=result)]

        elif name == "search_web":
            query = arguments.get("query", "")
            max_res = arguments.get("max_results", 5)
            result = search_web(query, max_res)
            return [mcp_types.TextContent(type="text", text=result)]

        else:
            error_msg = json.dumps({"error": f"Unknown tool: {name}"})
            return [mcp_types.TextContent(type="text", text=error_msg)]

    except Exception as e:
        error_msg = json.dumps(
            {"error": f"Tool execution failed: {str(e)}", "tool": name})
        return [mcp_types.TextContent(type="text", text=error_msg)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
