"""Image editing uploads project assets and preserves text-generation behavior."""
import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from batteryscientist.integrations import images
from batteryscientist.tools.core import ToolContext
from batteryscientist.tools.image_generation import generate_image


@pytest.fixture
def image_api(tmp_path, monkeypatch):
    stream = io.BytesIO()
    Image.new('RGB', (16, 16), 'white').save(stream, format='PNG')
    content = stream.getvalue()
    (tmp_path / 'figures').mkdir()
    (tmp_path / 'figures/input.png').write_bytes(content)
    client = MagicMock()
    client.__enter__.return_value = client
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(content).decode())])
    client.images.generate.return_value = response
    client.images.edit.return_value = response
    monkeypatch.setattr(images, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setattr(images, 'record_usage', lambda *args, **kwargs: None)
    monkeypatch.setenv('BATTERYSCIENTIST_IMAGE_API_KEY', 'test-key')
    return tmp_path, client, content


def test_edit_uploads_image_through_tool(image_api):
    import asyncio
    project, client, content = image_api
    uploaded = []
    def edit(**kwargs):
        uploaded.append(kwargs['image'].read())
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(content).decode())])
    client.images.edit.side_effect = edit
    result = asyncio.run(generate_image(ToolContext(workspace=project), 'polish',
                                       output_path='figures/edited.png', size='auto',
                                       image_path='figures/input.png'))
    assert not result.is_error, result.output
    assert uploaded == [content]
    assert (project / 'figures/edited.png').read_bytes() == content
    assert result.metadata['image_path'] == 'figures/input.png'
    client.images.generate.assert_not_called()


def test_text_generation_remains_supported(image_api):
    project, client, _ = image_api
    receipt = images.generate(project, 'draw', 'figures/generated.png', size='auto', quality='high')
    assert receipt['status'] == 'completed'
    client.images.generate.assert_called_once()
    client.images.edit.assert_not_called()


@pytest.mark.parametrize('source', ['../outside.png', 'figures/missing.png'])
def test_invalid_input_does_not_call_api(image_api, source):
    project, client, _ = image_api
    with pytest.raises(ValueError):
        images.generate(project, 'edit', 'figures/output.png', size='auto', quality='high', image_path=source)
    client.images.edit.assert_not_called()


def test_input_is_not_overwritten(image_api):
    project, client, content = image_api
    with pytest.raises(ValueError):
        images.generate(project, 'edit', 'figures/input.png', size='auto', quality='high',
                        image_path='figures/input.png')
    assert (project / 'figures/input.png').read_bytes() == content
    client.images.edit.assert_not_called()
