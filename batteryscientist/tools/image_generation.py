"""Agent-visible image tool using the shared external-service adapter."""
from __future__ import annotations

import asyncio
import json

from batteryscientist.integrations.images import generate
from .core import ToolContext, ToolResult, tool


@tool()
async def generate_image(ctx: ToolContext, prompt: str,
                         output_path: str = 'figures/ai_generated/figure.png',
                         size: str = '1536x1024', quality: str = 'high',
                         image_path: str | None = None) -> ToolResult:
    """Generate a PNG, or edit a project image supplied with image_path, under figures/.
    A failed or interrupted API request is not retried automatically.
    """
    try:
        ctx.resolve(output_path, write=True)
        options = {'size': size, 'quality': quality}
        if image_path:
            ctx.resolve(image_path)
            options['image_path'] = image_path
        receipt = await asyncio.to_thread(generate, ctx.workspace, prompt, output_path, **options)
    except Exception as exc:
        return ToolResult(str(exc), True)
    return ToolResult(json.dumps(receipt, ensure_ascii=False), metadata=receipt)
