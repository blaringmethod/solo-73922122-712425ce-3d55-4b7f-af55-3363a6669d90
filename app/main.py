"""Web application for the institutional-repository harvester.

Pages
-----
``/``                          catalogue browser, watermark, run controls
``/people``                    署名归并：把已发布署名人工归到人物编号并列出人物书目
``/runs``                      round history (windows, attempts, restarts, errors)
``/status``                    statically rendered status JSON
``GET  /api/status``           polling endpoint used by the page
``GET  /api/catalog``          catalogue JSON (search + pagination)
``GET  /api/people``           merge-page state JSON (attributions, persons, version)
``POST /api/harvest``          start a round
``POST /api/harvest/cancel``   cancel the active round without waiting for it
``POST /api/harvest/restart``  cancel (if any) and immediately start a new round
``POST /people/assign``        attach one attribution to a person (version-checked)
``POST /people/unassign``      release one attribution (version-checked)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .harvest import HarvestConfig, Harvester
from .store import Store

BASE_DIR = Path(__file__).resolve().parent


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data"))


def _oai_url() -> str:
    return os.environ.get("OAI_URL", "http://demo-oai:8000/oai")


def create_app(
    *,
    db_path: str | Path | None = None,
    oai_url: str | None = None,
    read_timeout: float | None = None,
    max_attempts: int | None = None,
    client_factory: Any = None,
) -> FastAPI:
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(db_path or (data_dir / "harvester.sqlite3"))
    config = HarvestConfig(
        oai_url=oai_url or _oai_url(),
        read_timeout=float(
            read_timeout
            if read_timeout is not None
            else os.environ.get("OAI_READ_TIMEOUT", "20")
        ),
        connect_timeout=float(os.environ.get("OAI_CONNECT_TIMEOUT", "10")),
        max_attempts=int(
            max_attempts
            if max_attempts is not None
            else os.environ.get("HARVEST_MAX_ATTEMPTS", "3")
        ),
        retry_backoff_ms=int(os.environ.get("HARVEST_RETRY_BACKOFF_MS", "300")),
    )
    harvester = Harvester(store, config, client_factory=client_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # A fresh process owns no in-flight rounds; Store already marked any
        # 'running' rows from a previous process as failed without publishing.
        yield
        await harvester.shutdown()
        store.close()

    app = FastAPI(title="机构库采集控制台", lifespan=lifespan)
    app.state.store = store
    app.state.harvester = harvester
    app.state.config = config
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    static_dir = BASE_DIR / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def status_payload() -> dict[str, Any]:
        active = store.get_active_run()
        published = store.published_state()
        last = store.list_runs(limit=1)
        return {
            "active": active,
            "published": published,
            "last_run": last[0] if last else None,
            "config": {
                "oai_url": config.oai_url,
                "max_attempts": config.max_attempts,
                "read_timeout": config.read_timeout,
            },
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        q: str = "",
        page: int = 1,
        page_size: int = 20,
        message: str | None = None,
    ) -> HTMLResponse:
        catalogue = store.catalog_page(q, page, page_size)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "catalogue": catalogue,
                "published": store.published_state(),
                "status": status_payload(),
                "runs": store.list_runs(limit=10),
                "message": message,
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": store.list_runs(limit=50),
                "published": store.published_state(),
                "status": status_payload(),
            },
        )

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(status_payload())

    @app.get("/status")
    async def status_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "status.html", {"status": status_payload()}
        )

    @app.post("/api/harvest")
    async def start_harvest() -> JSONResponse:
        try:
            control = await harvester.start_run()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {"run_id": control.run_id, "status": "running"}, status_code=202
        )

    @app.post("/api/harvest/cancel")
    async def cancel_harvest() -> JSONResponse:
        active = store.get_active_run()
        if active is None:
            raise HTTPException(status_code=409, detail="当前没有正在运行的采集轮次")
        await harvester.cancel_run(active["id"], "用户在网页取消", "user")
        return JSONResponse({"run_id": active["id"], "status": "cancelled"})

    @app.post("/api/harvest/restart")
    async def restart_harvest() -> JSONResponse:
        """Cancel the active round (if any) and immediately start a new one.

        ``cancel_run`` returns as soon as SQLite has recorded the cancellation
        and the old asyncio task has been asked to stop; it never waits for a
        slow HTTP page.  The new run gets a new ``run_id`` and its own window,
        and the commit ownership check makes the old round unable to publish.
        """
        active = store.get_active_run()
        if active is not None:
            await harvester.cancel_run(
                active["id"], "用户取消了本轮以开始新一轮", "superseded"
            )
        try:
            control = await harvester.start_run()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {"run_id": control.run_id, "status": "running"}, status_code=202
        )

    @app.get("/api/catalog")
    async def api_catalog(q: str = "", page: int = 1, page_size: int = 20) -> JSONResponse:
        return JSONResponse(store.catalog_page(q, page, page_size))

    # ------------------------------------------------------------- 署名归并
    def people_view(
        request: Request,
        *,
        message: str | None = None,
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "people.html",
            {
                "state": store.merge_state(),
                "published": store.published_state(),
                "message": message,
                "error": error,
            },
            status_code=status_code,
        )

    def people_reject(request: Request, result: dict[str, Any]) -> HTMLResponse:
        """Refuse a stale or conflicting merge submit and show current data."""
        reason = result.get("reason")
        if reason == "stale":
            error = (
                "提交已过期：署名或归属在你打开页面后已变化。"
                "以下展示当前数据，请核对后重新提交。"
            )
        elif reason == "missing":
            error = "该署名在当前已发布目录中不存在。以下展示当前数据。"
        elif reason == "conflict":
            error = "该署名已有归属（一处署名只归一人）。以下展示当前数据。"
        elif reason == "no_person":
            error = "所选人物不存在。以下展示当前数据。"
        else:
            error = "提交被拒绝。以下展示当前数据。"
        return people_view(request, error=error, status_code=409)

    @app.get("/people", response_class=HTMLResponse)
    async def people_page(
        request: Request, message: str | None = None
    ) -> HTMLResponse:
        return people_view(request, message=message)

    @app.get("/api/people")
    async def api_people() -> JSONResponse:
        return JSONResponse(store.merge_state())

    @app.post("/people/assign")
    async def people_assign(
        request: Request,
        identifier: str = Form(...),
        position: int = Form(...),
        person_id: str = Form("__new__"),
        new_person_name: str = Form(""),
        version: str = Form(""),
    ) -> Response:
        create_new = person_id == "__new__"
        result = store.assign_attribution(
            identifier=identifier,
            position=position,
            expected_version=version,
            person_id=None if create_new else person_id,
            new_person_name=new_person_name or None,
        )
        if not result["ok"]:
            return people_reject(request, result)
        if result["created_person"]:
            message = (
                f"已建立人物 {result['person_id']}（{result['name_text']}）"
                f"并归并 {identifier} 第 {position + 1} 位署名"
            )
        else:
            message = (
                f"已把 {identifier} 第 {position + 1} 位署名"
                f"「{result['name_text']}」归入人物 {result['person_id']}"
            )
        return RedirectResponse(
            f"/people?message={quote(message)}", status_code=303
        )

    @app.post("/people/unassign")
    async def people_unassign(
        request: Request,
        identifier: str = Form(...),
        position: int = Form(...),
        version: str = Form(""),
    ) -> Response:
        result = store.unassign_attribution(
            identifier=identifier, position=position, expected_version=version
        )
        if not result["ok"]:
            return people_reject(request, result)
        message = f"已解除 {identifier} 第 {position + 1} 位署名的归属"
        return RedirectResponse(
            f"/people?message={quote(message)}", status_code=303
        )

    return app


app = create_app()
