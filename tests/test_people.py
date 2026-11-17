"""署名归并：人工把已发布书目的署名归到人物，以及与采集发布交错时的行为。

覆盖：
* 按分号拆分署名、建立/加入人物、一处署名只归一人、不按姓名自动合并；
* 人物书目与已发布目录一致；
* 过期提交（版本不符）被拒绝并展示当前数据；
* 成功采集改名/改序只保留序位与文字未变的归属，撤回解除归属，
  同名恢复不复活旧归属也不接受旧页面提交，重复采集未变化署名保持归属；
* 失败、取消、提交中途失败的采集都不改动归属。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.oai import OaiClient
from conftest import wait_for_outcome


def attribution(state, identifier, position):
    for item in state["attributions"]:
        if item["identifier"] == identifier and item["position"] == position:
            return item
    return None


def person(state, person_id):
    for p in state["persons"]:
        if p["id"] == person_id:
            return p
    return None


async def harvest_ok(harvester):
    control = await harvester.start_run()
    assert await wait_for_outcome(control) == "succeeded"
    return control


# --------------------------------------------------------------------------
# 归并本身
# --------------------------------------------------------------------------


async def test_split_assign_create_person_and_bibliography(harvester, store, repo):
    await harvest_ok(harvester)

    state = store.merge_state()
    # 演示数据 oai:demo:0001 的作者字段是 "王小明; 李静"，按分号拆成两处署名。
    first = attribution(state, "oai:demo:0001", 0)
    second = attribution(state, "oai:demo:0001", 1)
    assert first["name_text"] == "王小明"
    assert second["name_text"] == "李静"
    assert first["person_id"] is None and second["person_id"] is None

    result = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    assert result["ok"] and result["created_person"]
    pid = result["person_id"]
    assert pid.startswith("P")

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == pid
    # 同一条书目的另一处署名不会被顺带归并。
    assert attribution(state, "oai:demo:0001", 1)["person_id"] is None
    p = person(state, pid)
    assert p["display_name"] == "王小明"  # 未指定显示名时默认用署名文字
    assert p["attribution_count"] == 1
    assert [w["identifier"] for w in p["bibliography"]] == ["oai:demo:0001"]
    assert p["bibliography"][0]["title"] == "数字人文中的文本挖掘方法"


async def test_join_existing_person_and_one_attribution_one_person(harvester, store):
    await harvest_ok(harvester)

    state = store.merge_state()
    r1 = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    assert r1["ok"]
    pid = r1["person_id"]

    # 加入现有人物：另一条书目的署名归入同一人。
    state = store.merge_state()
    r2 = store.assign_attribution(
        identifier="oai:demo:0002",
        position=0,
        expected_version=state["version"],
        person_id=pid,
    )
    assert r2["ok"] and not r2["created_person"] and r2["person_id"] == pid

    state = store.merge_state()
    p = person(state, pid)
    assert p["attribution_count"] == 2
    assert {w["identifier"] for w in p["bibliography"]} == {
        "oai:demo:0001",
        "oai:demo:0002",
    }

    # 一处署名只归一人：已归属的署名再次归并被拒绝。
    state = store.merge_state()
    again = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    assert again == {"ok": False, "reason": "conflict", "person_id": pid}
    # 指定不存在的人物也被拒绝。
    ghost = store.assign_attribution(
        identifier="oai:demo:0003",
        position=0,
        expected_version=state["version"],
        person_id="P999999",
    )
    assert ghost["ok"] is False and ghost["reason"] == "no_person"


async def test_no_auto_merge_by_name(harvester, store, repo):
    await harvest_ok(harvester)
    repo.add_record(title="另一篇同名作者的论文", creator="王小明")
    await harvest_ok(harvester)

    state = store.merge_state()
    new_id = next(
        item["identifier"]
        for item in state["attributions"]
        if item["title"] == "另一篇同名作者的论文"
    )
    r = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    assert r["ok"]

    state = store.merge_state()
    # 姓名相同不会自动合并：另一条的“王小明”仍未归属。
    assert attribution(state, new_id, 0)["person_id"] is None
    assert person(state, r["person_id"])["attribution_count"] == 1


async def test_unassign_releases_attribution(harvester, store):
    await harvest_ok(harvester)
    state = store.merge_state()
    r = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    pid = r["person_id"]

    state = store.merge_state()
    gone = store.unassign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    assert gone == {"ok": True, "person_id": pid}

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] is None
    # 人物编号保留，书目为空。
    p = person(state, pid)
    assert p is not None and p["bibliography"] == [] and p["attribution_count"] == 0

    # 重复解除同一署名：冲突拒绝。
    again = store.unassign_attribution(
        identifier="oai:demo:0001",
        position=0,
        expected_version=store.merge_state()["version"],
    )
    assert again["ok"] is False and again["reason"] == "conflict"


async def test_stale_submit_is_rejected(harvester, store):
    await harvest_ok(harvester)
    stale_version = store.merge_state()["version"]

    r = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=stale_version
    )
    assert r["ok"]

    # 旧版本再提交（归并或解除）一律拒绝。
    stale_assign = store.assign_attribution(
        identifier="oai:demo:0001", position=1, expected_version=stale_version
    )
    assert stale_assign == {"ok": False, "reason": "stale"}
    stale_unassign = store.unassign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=stale_version
    )
    assert stale_unassign == {"ok": False, "reason": "stale"}

    # 不存在的署名序位：missing。
    missing = store.assign_attribution(
        identifier="oai:demo:0001",
        position=9,
        expected_version=store.merge_state()["version"],
    )
    assert missing["ok"] is False and missing["reason"] == "missing"


# --------------------------------------------------------------------------
# 采集发布 × 归属
# --------------------------------------------------------------------------


async def test_harvest_rename_keeps_only_unchanged_positions(harvester, store, repo):
    await harvest_ok(harvester)
    state = store.merge_state()
    p1 = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]
    state = store.merge_state()
    p2 = store.assign_attribution(
        identifier="oai:demo:0001", position=1, expected_version=state["version"]
    )["person_id"]

    # 源端把第 2 位作者改名（李静 → 李靖），第 1 位文字不变。
    repo.add_record(
        identifier="oai:demo:0001",
        title="数字人文中的文本挖掘方法",
        creator="王小明; 李靖",
    )
    await harvest_ok(harvester)

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == p1  # 保留
    assert attribution(state, "oai:demo:0001", 1)["person_id"] is None  # 解除
    assert attribution(state, "oai:demo:0001", 1)["name_text"] == "李靖"
    assert [w["identifier"] for w in person(state, p1)["bibliography"]] == [
        "oai:demo:0001"
    ]
    assert person(state, p2)["bibliography"] == []


async def test_harvest_reorder_drops_both_positions(harvester, store, repo):
    await harvest_ok(harvester)
    state = store.merge_state()
    store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )
    state = store.merge_state()
    store.assign_attribution(
        identifier="oai:demo:0001", position=1, expected_version=state["version"]
    )

    # 同样的两个名字，顺序对调：两个序位的文字都变了，归属全部解除。
    repo.add_record(
        identifier="oai:demo:0001",
        title="数字人文中的文本挖掘方法",
        creator="李静; 王小明",
    )
    await harvest_ok(harvester)

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] is None
    assert attribution(state, "oai:demo:0001", 1)["person_id"] is None


async def test_withdraw_releases_and_same_name_restore_does_not_revive(
    harvester, store, repo
):
    await harvest_ok(harvester)
    state = store.merge_state()
    pid = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]
    old_version = store.merge_state()["version"]

    # 撤回：归属解除，书目从目录与人物书目中消失。
    repo.withdraw("oai:demo:0001")
    await harvest_ok(harvester)
    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0) is None
    assert person(state, pid)["bibliography"] == []

    # 同名恢复：书目回来了，但旧归属不能复活。
    repo.add_record(
        identifier="oai:demo:0001",
        title="数字人文中的文本挖掘方法",
        creator="王小明; 李静",
    )
    await harvest_ok(harvester)
    state = store.merge_state()
    restored = attribution(state, "oai:demo:0001", 0)
    assert restored is not None and restored["name_text"] == "王小明"
    assert restored["person_id"] is None
    assert person(state, pid)["bibliography"] == []

    # 撤回前的旧页面提交也不能被接受。
    stale = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=old_version
    )
    assert stale == {"ok": False, "reason": "stale"}


async def test_reharvest_unchanged_attributions_keeps_assignments(
    harvester, store, repo
):
    await harvest_ok(harvester)
    state = store.merge_state()
    pid = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]
    version_before = store.merge_state()["version"]

    # 空窗口重复采集（边界秒会重读未变化的记录）：归属保持，页面版本不变。
    await harvest_ok(harvester)
    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == pid
    assert state["version"] == version_before

    # 只改标题、署名文字不变的更正：归属同样保持。
    repo.touch_record("oai:demo:0001", title="只改了标题")
    await harvest_ok(harvester)
    state = store.merge_state()
    kept = attribution(state, "oai:demo:0001", 0)
    assert kept["person_id"] == pid
    assert kept["title"] == "只改了标题"


async def test_failed_harvest_keeps_assignments(harvester, store, repo):
    await harvest_ok(harvester)
    state = store.merge_state()
    pid = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]
    version_before = store.merge_state()["version"]

    repo.set_meta("page_size", "1")
    repo.set_meta("token_failures", "10")
    repo.add_record(title="这条不会被发布")
    repo.add_record(title="这条也不会被发布")
    repo.add_record(title="这条同样不会")

    control = await harvester.start_run()
    assert await wait_for_outcome(control) == "failed"

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == pid
    assert state["version"] == version_before  # 目录与归属都没动
    assert person(state, pid)["bibliography"][0]["identifier"] == "oai:demo:0001"


async def test_cancelled_harvest_keeps_assignments(harvester, store, repo):
    await harvest_ok(harvester)
    state = store.merge_state()
    pid = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]
    version_before = store.merge_state()["version"]

    repo.set_meta("slow_seconds", "5")
    repo.add_record(title="本应被取消的改动")
    control = await harvester.start_run()
    await asyncio.sleep(0.3)
    await harvester.cancel_run(control.run_id, "测试取消", "user")
    assert await wait_for_outcome(control) == "cancelled"

    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == pid
    assert state["version"] == version_before


async def test_assign_during_running_harvest_then_commit_reconciles(
    harvester, store, repo
):
    """采集发布与人工归并交错：归并打在旧目录上，提交按新目录调和。"""
    await harvest_ok(harvester)
    state = store.merge_state()
    p1 = store.assign_attribution(
        identifier="oai:demo:0001", position=0, expected_version=state["version"]
    )["person_id"]

    # 源端改名第 2 位作者，并让源端变慢，使采集还在进行时仍能归并。
    repo.add_record(
        identifier="oai:demo:0001",
        title="数字人文中的文本挖掘方法",
        creator="王小明; 李靖",
    )
    repo.set_meta("slow_seconds", "2")
    control = await harvester.start_run()
    await asyncio.sleep(0.3)
    assert store.get_run(control.run_id)["status"] == "running"

    # 采集尚未提交，目录仍是旧数据：此时归并旧署名“李静”成功。
    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 1)["name_text"] == "李静"
    p2 = store.assign_attribution(
        identifier="oai:demo:0001", position=1, expected_version=state["version"]
    )["person_id"]

    repo.set_meta("slow_seconds", "0")
    assert await wait_for_outcome(control) == "succeeded"

    # 提交后：序位 0 文字未变 → 保留；序位 1 被改名 → 归并的那条归属被解除。
    state = store.merge_state()
    assert attribution(state, "oai:demo:0001", 0)["person_id"] == p1
    assert attribution(state, "oai:demo:0001", 1)["person_id"] is None
    assert attribution(state, "oai:demo:0001", 1)["name_text"] == "李靖"
    # 人物书目对应同一已发布目录：p1 仍有此书，p2 书目为空。
    assert [w["identifier"] for w in person(state, p1)["bibliography"]] == [
        "oai:demo:0001"
    ]
    assert person(state, p2)["bibliography"] == []


async def test_failed_commit_rolls_back_attribution_changes(store, monkeypatch):
    """提交事务中途失败：目录、归属一起回滚，保持提交前状态。"""
    run = store.create_run()
    store.set_window(run["id"], "2026-01-01T00:00:00Z")
    store.stage_records(
        run["id"],
        [
            {
                "identifier": "oai:m:1",
                "title": "旧标题",
                "creator": "张三; 李四",
                "datestamp": "2025-12-31T00:00:00Z",
                "deleted": False,
            }
        ],
        0,
    )
    assert store.commit_run(run["id"])["committed"]
    state = store.merge_state()
    pid = store.assign_attribution(
        identifier="oai:m:1", position=0, expected_version=state["version"]
    )["person_id"]

    run2 = store.create_run()
    store.set_window(run2["id"], "2026-01-02T00:00:00Z")
    store.stage_records(
        run2["id"],
        [
            {
                "identifier": "oai:m:1",
                "title": "旧标题",
                "creator": "张三; 王五",  # 第 2 位改名，本应在提交时解除其归属
                "datestamp": "2026-01-01T00:00:00Z",
                "deleted": False,
            }
        ],
        0,
    )

    class ExplodingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if sql.strip().startswith("INSERT INTO catalog"):
                raise sqlite3.OperationalError("simulated disk error")
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real_conn = store._conn
    monkeypatch.setattr(store, "_conn", ExplodingConnection(real_conn))
    with pytest.raises(sqlite3.OperationalError):
        store.commit_run(run2["id"])
    monkeypatch.setattr(store, "_conn", real_conn)

    # 回滚后：目录仍是旧文字，两处归属原样保留。
    state = store.merge_state()
    assert attribution(state, "oai:m:1", 0)["person_id"] == pid
    assert attribution(state, "oai:m:1", 1)["name_text"] == "李四"
    store.finish_failed(run2["id"], "simulated")


# --------------------------------------------------------------------------
# 网页层
# --------------------------------------------------------------------------


@pytest.fixture
def web_app(tmp_path, demo_app, repo):
    def factory():
        return OaiClient(
            "http://demo.local/oai",
            transport=httpx.ASGITransport(app=demo_app),
        )

    return create_app(
        db_path=tmp_path / "web.sqlite3",
        oai_url="http://demo.local/oai",
        client_factory=factory,
    )


def poll_until(client: TestClient, predicate, attempts: int = 100) -> dict:
    for _ in range(attempts):
        payload = client.get("/api/status").json()
        if predicate(payload):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"condition not met; last={payload}")


def harvest_via_web(client: TestClient, success_count: int = 1) -> None:
    assert client.post("/api/harvest").status_code == 202
    poll_until(
        client,
        lambda p: p["active"] is None
        and p["published"]["success_count"] == success_count,
    )


def test_people_page_lists_attributions_and_nav(web_app):
    with TestClient(web_app) as client:
        harvest_via_web(client)
        page = client.get("/people")
        assert page.status_code == 200
        assert "署名归并" in page.text
        assert "王小明" in page.text and "李静" in page.text
        assert "未归属" in page.text
        # 各页面导航都有署名归并入口。
        for path in ("/", "/runs", "/status"):
            assert 'href="/people"' in client.get(path).text

        state = client.get("/api/people").json()
        assert state["version"]
        assert state["unassigned"] == len(state["attributions"])
        assert state["persons"] == []


def test_assign_and_unassign_via_web(web_app):
    with TestClient(web_app) as client:
        harvest_via_web(client)
        version = client.get("/api/people").json()["version"]

        response = client.post(
            "/people/assign",
            data={
                "identifier": "oai:demo:0001",
                "position": "0",
                "person_id": "__new__",
                "new_person_name": "",
                "version": version,
            },
        )
        assert response.status_code == 200  # 303 后跟随到 /people
        assert "已建立人物" in response.text

        state = client.get("/api/people").json()
        person_id = state["attributions"]
        person_id = next(
            a["person_id"]
            for a in state["attributions"]
            if a["identifier"] == "oai:demo:0001" and a["position"] == 0
        )
        assert person_id
        person = state["persons"][0]
        assert person["id"] == person_id
        assert [w["identifier"] for w in person["bibliography"]] == [
            "oai:demo:0001"
        ]
        # 人物书目在页面上列出。
        page = client.get("/people")
        assert person_id in page.text
        assert "数字人文中的文本挖掘方法" in page.text

        # 重复归并同一署名：409 并展示当前数据。
        conflict = client.post(
            "/people/assign",
            data={
                "identifier": "oai:demo:0001",
                "position": "0",
                "person_id": "__new__",
                "new_person_name": "",
                "version": state["version"],
            },
        )
        assert conflict.status_code == 409
        assert "一处署名只归一人" in conflict.text

        # 解除归属。
        state = client.get("/api/people").json()
        response = client.post(
            "/people/unassign",
            data={
                "identifier": "oai:demo:0001",
                "position": "0",
                "version": state["version"],
            },
        )
        assert response.status_code == 200
        state = client.get("/api/people").json()
        assert all(
            a["person_id"] is None for a in state["attributions"]
        )
        assert state["persons"][0]["bibliography"] == []  # 编号保留


def test_stale_web_submit_rejected_and_current_data_shown(web_app, repo):
    """采集发布与页面打开交错：旧版本提交被拒，响应展示当前数据。"""
    with TestClient(web_app) as client:
        harvest_via_web(client)
        stale_version = client.get("/api/people").json()["version"]

        # 用户拿着旧页面时，一轮采集改名了第 2 位作者并成功发布。
        repo.add_record(
            identifier="oai:demo:0001",
            title="数字人文中的文本挖掘方法",
            creator="王小明; 李靖",
        )
        harvest_via_web(client, success_count=2)

        response = client.post(
            "/people/assign",
            data={
                "identifier": "oai:demo:0001",
                "position": "1",
                "person_id": "__new__",
                "new_person_name": "",
                "version": stale_version,
            },
        )
        assert response.status_code == 409
        assert "已过期" in response.text
        # 展示的是当前数据：新署名“李靖”、未归属，且表单携带新版本。
        assert "李靖" in response.text
        assert "李静</td>" not in response.text
        current_version = client.get("/api/people").json()["version"]
        assert current_version != stale_version
        assert current_version in response.text
        # 当前数据里该署名仍未归属（旧提交没有生效）。
        state = client.get("/api/people").json()
        target = next(
            a
            for a in state["attributions"]
            if a["identifier"] == "oai:demo:0001" and a["position"] == 1
        )
        assert target["person_id"] is None
