#  Copyright (c) ZenML GmbH 2026. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Contract tests for blob repositories."""

import hashlib
import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import event

from conftest import (
    FakeBlobRepository,
    pg_session,
    pg_session_with_engine,
    postgres_available,
)
from kitaru.server.adapters.db.repositories.account_repository import (
    SQLAccountRepository,
)
from kitaru.server.adapters.db.repositories.blob_repository import SQLBlobRepository
from kitaru.server.adapters.db.repositories.plugin_repository import (
    SQLPluginRepository,
)
from kitaru.server.application.interfaces.blob_repository import BlobRepository
from kitaru.server.domain.account import Account
from kitaru.server.domain.blob import Blob, BlobInUse, BlobNotFound
from kitaru.server.domain.plugin import (
    Plugin,
    PluginKind,
    ScriptPluginSource,
)

Setup = tuple[BlobRepository, uuid.UUID]


def _blob(owner_id: uuid.UUID | None, content: bytes = b"content") -> Blob:
    return Blob(
        owner_id=owner_id,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        media_type="text/plain",
        data=content,
    )


@pytest.fixture(params=["fake", "postgres"])
async def setup(request: pytest.FixtureRequest) -> AsyncGenerator[Setup, None]:
    """Provide each blob repository implementation plus an owner id."""
    if request.param == "fake":
        yield FakeBlobRepository(), uuid.uuid4()
        return
    if not await postgres_available():
        pytest.skip("PostgreSQL is not reachable")
    async with pg_session() as session:
        owner = await SQLAccountRepository(session).create(Account(name="owner"))
        yield SQLBlobRepository(session), owner.id


async def test_create_sets_timestamp(setup: Setup) -> None:
    """Store a new blob with its created timestamp set."""
    repository, owner_id = setup
    blob, created = await repository.create(_blob(owner_id))
    assert created is True
    assert blob.owner_id == owner_id
    assert blob.size == len(b"content")
    assert blob.media_type == "text/plain"
    assert blob.data == b"content"
    assert blob.created is not None


async def test_create_and_round_trip_without_owner(setup: Setup) -> None:
    """Store and reload a default plugin blob with no owner."""
    repository, _ = setup
    created, _ = await repository.create(_blob(None))
    assert created.owner_id is None
    loaded = await repository.get(created.id)
    assert loaded.owner_id is None


async def test_create_dedup(setup: Setup) -> None:
    """Return the existing row unmarked as created on a duplicate sha256."""
    repository, owner_id = setup
    first, created_first = await repository.create(_blob(owner_id, b"same"))
    assert created_first is True
    second, created_second = await repository.create(_blob(owner_id, b"same"))
    assert created_second is False
    assert second.id == first.id
    assert second.sha256 == first.sha256


async def test_get(setup: Setup) -> None:
    """Load a stored blob by id with its content."""
    repository, owner_id = setup
    created, _ = await repository.create(_blob(owner_id))
    loaded = await repository.get(created.id)
    assert loaded == created


async def test_get_not_found(setup: Setup) -> None:
    """Raise for an unknown blob id."""
    repository, _ = setup
    missing_id = uuid.uuid4()
    with pytest.raises(BlobNotFound, match=f"Blob {missing_id} was not found"):
        await repository.get(missing_id)


async def test_get_metadata(setup: Setup) -> None:
    """Load a stored blob's metadata without its content."""
    repository, owner_id = setup
    created, _ = await repository.create(_blob(owner_id))
    loaded = await repository.get_metadata(created.id)
    assert loaded.id == created.id
    assert loaded.owner_id == owner_id
    assert loaded.sha256 == created.sha256
    assert loaded.size == created.size
    assert loaded.media_type == created.media_type
    assert loaded.created == created.created
    assert loaded.data == b""


async def test_get_metadata_not_found(setup: Setup) -> None:
    """Raise for an unknown blob id on the metadata path."""
    repository, _ = setup
    missing_id = uuid.uuid4()
    with pytest.raises(BlobNotFound, match=f"Blob {missing_id} was not found"):
        await repository.get_metadata(missing_id)


async def test_delete(setup: Setup) -> None:
    """Delete a stored blob."""
    repository, owner_id = setup
    created, _ = await repository.create(_blob(owner_id))
    await repository.delete(created.id)
    with pytest.raises(BlobNotFound):
        await repository.get(created.id)


async def test_delete_not_found(setup: Setup) -> None:
    """Raise for an unknown blob id."""
    repository, _ = setup
    missing_id = uuid.uuid4()
    with pytest.raises(BlobNotFound, match=f"Blob {missing_id} was not found"):
        await repository.delete(missing_id)


async def test_delete_in_use(setup: Setup) -> None:
    """Reject deleting a blob referenced by a plugin version."""
    repository, owner_id = setup
    blob, _ = await repository.create(_blob(owner_id))

    if isinstance(repository, FakeBlobRepository):
        repository.mark_referenced(blob.id)
    else:
        assert isinstance(repository, SQLBlobRepository)
        plugin_repository = SQLPluginRepository(repository._session)
        plugin = await plugin_repository.create(
            Plugin(owner_id=owner_id, kind=PluginKind.EVALUATOR, name="scorer")
        )
        await plugin_repository.create_version(
            plugin.id,
            ScriptPluginSource(blob_id=blob.id, entrypoint="score"),
            display_version=None,
        )

    with pytest.raises(BlobInUse, match=f"Blob {blob.id} is in use"):
        await repository.delete(blob.id)


async def test_metadata_and_delete_skip_content_column() -> None:
    """Emit no SQL touching the content column outside the content load."""
    if not await postgres_available():
        pytest.skip("PostgreSQL is not reachable")
    async with pg_session_with_engine() as (session, engine):
        statements: list[str] = []
        event.listen(
            engine.sync_engine,
            "before_cursor_execute",
            lambda conn, cursor, statement, *args: statements.append(statement),
        )
        repository = SQLBlobRepository(session)
        owner = await SQLAccountRepository(session).create(Account(name="owner"))
        created, _ = await repository.create(_blob(owner.id))

        session.expire_all()
        statements.clear()
        loaded = await repository.get(created.id)
        assert loaded.data == b"content"
        assert any("blob.data" in s for s in statements)

        session.expire_all()
        statements.clear()
        await repository.get_metadata(created.id)
        await repository.delete(created.id)
        assert not any("blob.data" in s for s in statements)
