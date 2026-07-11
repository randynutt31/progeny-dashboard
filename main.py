"""
Progyny Infinite Dashboard
Hosted on Railway. Open from any device, any browser.
All AI calls route through server — no browser CORS issues.
"""

from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
import hashlib
import anthropic
import httpx

app = FastAPI(title="Progyny Infinite Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_URL = os.environ.get("AGENT_URL", "https://picp-agent-production.up.railway.app")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ── ReptiTerra feeding dashboard — Supabase config + helpers ──────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FEEDING_TOKEN = os.environ.get("FEEDING_TOKEN", "")
# tier3_databank is RLS-locked (deny-all) and needs the service-role key. New-format
# sb_secret_ keys are NOT JWTs: they go in the apikey header ONLY, never Authorization.
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _sb_headers(prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


async def _sb(method, table, params=None, json=None, prefer=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase env vars (SUPABASE_URL / SUPABASE_KEY) not set")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as hc:
        r = await hc.request(method, url, headers=_sb_headers(prefer), params=params, json=json)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=f"Supabase error: {r.text}")
    if r.status_code == 204 or not r.text:
        return []
    try:
        return r.json()
    except Exception:
        return []


async def _sb_service(method, table, params=None, json=None, prefer=None):
    """Supabase REST for the RLS-locked tier3_databank, using the service-role key.
    CRITICAL: sb_secret_ keys are NOT JWTs - send the value in the apikey header ONLY.
    Putting it in an Authorization: Bearer header gets rejected (that was the 401).
    Feeding endpoints stay on _sb()/the anon SUPABASE_KEY and are untouched."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500,
                            detail="Supabase service env vars (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY) not set")
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as hc:
        r = await hc.request(method, url, headers=headers, params=params, json=json)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=f"Supabase error: {r.text}")
    if r.status_code == 204 or not r.text:
        return []
    try:
        return r.json()
    except Exception:
        return []


def rt_auth(x_api_token: str = Header(default="")):
    if FEEDING_TOKEN and x_api_token != FEEDING_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return True


class AIRequest(BaseModel):
    prompt: str
    web_search: bool = False


class IngestRequest(BaseModel):
    content: str
    source: str = "session"


class QueryRequest(BaseModel):
    question: str


# ── Server-side AI proxy ──────────────────────────────────────────────────────

@app.post("/api/ai")
async def ai_proxy(req: AIRequest):
    if not client:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=500)
    try:
        tools = [{"type": "web_search_20250305", "name": "web_search"}] if req.web_search else []
        kwargs = dict(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": req.prompt}],
        )
        if tools:
            kwargs["tools"] = tools
        response = client.messages.create(**kwargs)
        text = " ".join(b.text for b in response.content if hasattr(b, "text"))
        return {"result": text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Agent proxy ───────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest_proxy(req: IngestRequest):
    async with httpx.AsyncClient(timeout=30) as hc:
        r = await hc.post(f"{AGENT_URL}/ingest", json={"content": req.content, "source": req.source})
        return r.json()


@app.post("/api/query")
async def query_proxy(req: QueryRequest):
    async with httpx.AsyncClient(timeout=30) as hc:
        r = await hc.post(f"{AGENT_URL}/query", json={"question": req.question})
        return r.json()


@app.get("/api/log")
async def log_proxy():
    async with httpx.AsyncClient(timeout=10) as hc:
        r = await hc.get(f"{AGENT_URL}/log")
        return r.json()


@app.get("/api/agent-status")
async def agent_status():
    try:
        async with httpx.AsyncClient(timeout=5) as hc:
            r = await hc.get(f"{AGENT_URL}/")
            return {"online": True, **r.json()}
    except Exception:
        return {"online": False}


# ── ReptiTerra feeding routes ─────────────────────────────────────────────────

# Animals
@app.get("/api/feeding/animals")
async def rt_get_animals(_: bool = Depends(rt_auth)):
    return await _sb("GET", "rt_animals", params={"select": "*", "order": "created_at.asc"})


@app.post("/api/feeding/animals")
async def rt_add_animal(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    row = {
        "name": b.get("name"),
        "species": b.get("species"),
        "emoji": b.get("emoji") or "🦎",
        "feed_every_days": b.get("feedEveryDays") or 7,
        "prey_type": b.get("preyType"),
        "prey_size": b.get("preySize"),
    }
    return await _sb("POST", "rt_animals", json=row, prefer="return=representation")


@app.patch("/api/feeding/animals/{animal_id}")
async def rt_update_animal(animal_id: str, req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    mapping = {"name": "name", "species": "species", "emoji": "emoji",
               "feedEveryDays": "feed_every_days", "preyType": "prey_type", "preySize": "prey_size"}
    row = {v: b[k] for k, v in mapping.items() if k in b}
    return await _sb("PATCH", "rt_animals", params={"id": f"eq.{animal_id}"}, json=row, prefer="return=representation")


@app.delete("/api/feeding/animals/{animal_id}")
async def rt_delete_animal(animal_id: str, _: bool = Depends(rt_auth)):
    await _sb("DELETE", "rt_animals", params={"id": f"eq.{animal_id}"})
    return {"ok": True}


# Feeding logs
@app.get("/api/feeding/logs/{animal_id}")
async def rt_get_logs(animal_id: str, _: bool = Depends(rt_auth)):
    return await _sb("GET", "rt_feeding_logs",
                     params={"animal_id": f"eq.{animal_id}", "select": "*", "order": "ts.desc", "limit": "50"})


@app.post("/api/feeding/logs/{animal_id}")
async def rt_add_log(animal_id: str, req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    refused = bool(b.get("refused"))
    row = {"animal_id": animal_id, "date": b.get("date"), "prey": b.get("prey"),
           "size": b.get("size"), "notes": b.get("notes"), "refused": refused}
    res = await _sb("POST", "rt_feeding_logs", json=row, prefer="return=representation")
    deduct = b.get("deduct")
    if deduct and not refused:
        cat, size = deduct.get("category"), deduct.get("size")
        cur = await _sb("GET", "rt_feed_inventory",
                        params={"category": f"eq.{cat}", "size": f"eq.{size}", "select": "count"})
        if cur:
            newc = max(0, (cur[0].get("count") or 0) - 1)
            await _sb("PATCH", "rt_feed_inventory",
                      params={"category": f"eq.{cat}", "size": f"eq.{size}"}, json={"count": newc})
    return res


# Feed inventory
@app.get("/api/feeding/inventory")
async def rt_get_inventory(_: bool = Depends(rt_auth)):
    return await _sb("GET", "rt_feed_inventory", params={"select": "*"})


@app.post("/api/feeding/inventory/delivery")
async def rt_delivery(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    cat, items = b.get("category"), (b.get("items") or {})
    for size, qty in items.items():
        try:
            qty = int(qty)
        except Exception:
            continue
        if qty <= 0:
            continue
        cur = await _sb("GET", "rt_feed_inventory",
                        params={"category": f"eq.{cat}", "size": f"eq.{size}", "select": "count"})
        if cur:
            newc = (cur[0].get("count") or 0) + qty
            await _sb("PATCH", "rt_feed_inventory",
                      params={"category": f"eq.{cat}", "size": f"eq.{size}"}, json={"count": newc})
        else:
            await _sb("POST", "rt_feed_inventory", json={"category": cat, "size": size, "count": qty})
    return {"ok": True}


@app.post("/api/feeding/inventory/adjust")
async def rt_inventory_adjust(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    cat, size, delta = b.get("category"), b.get("size"), int(b.get("delta") or 0)
    cur = await _sb("GET", "rt_feed_inventory",
                    params={"category": f"eq.{cat}", "size": f"eq.{size}", "select": "count"})
    if cur:
        newc = max(0, (cur[0].get("count") or 0) + delta)
        await _sb("PATCH", "rt_feed_inventory",
                  params={"category": f"eq.{cat}", "size": f"eq.{size}"}, json={"count": newc})
    elif delta > 0:
        await _sb("POST", "rt_feed_inventory", json={"category": cat, "size": size, "count": delta})
    return {"ok": True}


# Water bowls & hide boxes (fixed sizes)
async def _fixed_get(table):
    return await _sb("GET", table, params={"select": "*"})


async def _fixed_adjust(table, size, delta):
    cur = await _sb("GET", table, params={"size": f"eq.{size}", "select": "count"})
    if cur:
        newc = max(0, (cur[0].get("count") or 0) + delta)
        await _sb("PATCH", table, params={"size": f"eq.{size}"}, json={"count": newc})
    else:
        await _sb("POST", table, json={"size": size, "count": max(0, delta)})
    return {"ok": True}


@app.get("/api/feeding/bowls")
async def rt_get_bowls(_: bool = Depends(rt_auth)):
    return await _fixed_get("rt_bowls")


@app.post("/api/feeding/bowls/adjust")
async def rt_bowls_adjust(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    return await _fixed_adjust("rt_bowls", b.get("size"), int(b.get("delta") or 0))


@app.get("/api/feeding/hides")
async def rt_get_hides(_: bool = Depends(rt_auth)):
    return await _fixed_get("rt_hides")


@app.post("/api/feeding/hides/adjust")
async def rt_hides_adjust(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    return await _fixed_adjust("rt_hides", b.get("size"), int(b.get("delta") or 0))


# Supplies
@app.get("/api/feeding/supplies")
async def rt_get_supplies(_: bool = Depends(rt_auth)):
    return await _sb("GET", "rt_supplies", params={"select": "*", "order": "created_at.asc"})


@app.post("/api/feeding/supplies")
async def rt_add_supply(req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    row = {
        "category": b.get("category"), "name": b.get("name"), "spec": b.get("spec"),
        "qty": b.get("qty") or 0, "unit": b.get("unit"),
        "reorder_point": b.get("reorderPoint"), "reorder_enabled": bool(b.get("reorderEnabled")),
    }
    return await _sb("POST", "rt_supplies", json=row, prefer="return=representation")


@app.delete("/api/feeding/supplies/{supply_id}")
async def rt_delete_supply(supply_id: str, _: bool = Depends(rt_auth)):
    await _sb("DELETE", "rt_supplies", params={"id": f"eq.{supply_id}"})
    return {"ok": True}


@app.get("/api/feeding/supplies/{supply_id}/logs")
async def rt_get_supply_logs(supply_id: str, _: bool = Depends(rt_auth)):
    return await _sb("GET", "rt_supply_logs",
                     params={"supply_id": f"eq.{supply_id}", "select": "*", "order": "ts.desc", "limit": "50"})


@app.post("/api/feeding/supplies/{supply_id}/logs")
async def rt_add_supply_log(supply_id: str, req: Request, _: bool = Depends(rt_auth)):
    b = await req.json()
    typ, qty = b.get("type"), int(b.get("qty") or 0)
    row = {"supply_id": supply_id, "date": b.get("date"), "type": typ, "qty": qty, "reason": b.get("reason")}
    res = await _sb("POST", "rt_supply_logs", json=row, prefer="return=representation")
    cur = await _sb("GET", "rt_supplies", params={"id": f"eq.{supply_id}", "select": "qty"})
    if cur:
        q = cur[0].get("qty") or 0
        if typ == "addition":
            q = q + qty
        elif typ == "replacement":
            q = max(0, q - qty)
        await _sb("PATCH", "rt_supplies", params={"id": f"eq.{supply_id}"}, json={"qty": q})
    return res


# ── TIER 4: THE MAIN BRAIN ──────────────────────────────────────────────────────
# tier4_brain is RLS-locked to the backend. Reuses the existing _sb() helper
# (SUPABASE_KEY) exactly like every other endpoint here — no supabase package.
# Table already has UNIQUE (section, source_id) and an english FTS index on content.

# IN — every Tier 3 source posts here. Upsert on (section, source_id):
# re-running an export updates, never duplicates.
@app.post("/tier3/push")
async def tier4_push(payload: dict, _: bool = Depends(rt_auth)):
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else [payload]
    rows = [{
        "section":     r.get("section", "randy"),
        "source_type": r.get("source_type"),
        "source_id":   r.get("source_id"),
        "title":       r.get("title"),
        "content":     r.get("content"),
        "record":      r.get("record", r),
    } for r in records]
    # on_conflict=section,source_id + Prefer: resolution=merge-duplicates -> upsert.
    # return=representation so we can count what was written.
    res = await _sb_service(
        "POST", "tier3_databank",
        params={"on_conflict": "section,source_id"},
        json=rows,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return {"ok": True, "written": len(res or [])}


# OUT — Tier 2 asks the brain questions. Optional full-text search over content
# hits the english FTS index via content=fts(english).<q>.
@app.get("/tier3/records")
async def tier4_records(
    section: str = "randy",
    q: str | None = None,
    limit: int = 50,
    _: bool = Depends(rt_auth),
):
    params = {
        "select": "id,section,source_type,source_id,title,content,record,ingested_at",
        "section": f"eq.{section}",
        "order": "ingested_at.desc",
        "limit": str(limit),
    }
    if q:
        params["content"] = f"fts(english).{q}"
    res = await _sb_service("GET", "tier3_databank", params=params)
    return {"section": section, "count": len(res or []), "records": res}


# Same auth + service-key read as /tier3/records, but returns only size metrics
# for the 'randy' section instead of the records themselves.
@app.get("/tier3/measure")
async def tier3_measure(_: bool = Depends(rt_auth)):
    params = {
        "select": "content",
        "section": "eq.randy",
    }
    res = await _sb_service("GET", "tier3_databank", params=params)
    rows = res or []
    total_chars = sum(len(r.get("content") or "") for r in rows)
    est_tokens = round(total_chars / 4)
    return {"count": len(rows), "total_chars": total_chars, "est_tokens": est_tokens}


# ── TIER 3 -> TIER 4 BELT ────────────────────────────────────────────────────────
# Move the 'randy' section off tier3_databank onto tier4_main. Same auth + service
# key as /tier3/records. Upsert on (source_section, source_id) so re-running the
# belt never duplicates. tier4_main already has UNIQUE (source_section, source_id).
@app.post("/tier4/load")
async def tier4_load(_: bool = Depends(rt_auth)):
    section = "randy"
    batch_size = 25
    last_id = 0
    moved = 0
    # Keyset (range) pagination on id so we never hold all 294 huge rows in one
    # read: pull 25 at a time via id>last_id, write them, then advance the cursor.
    # id is selected only to drive the cursor; the rest are the write columns.
    while True:
        batch = await _sb_service(
            "GET", "tier3_databank",
            params={
                "select": "id,section,source_type,source_id,title,content,record,ingested_at",
                "section": f"eq.{section}",
                "id": f"gt.{last_id}",
                "order": "id.asc",
                "limit": str(batch_size),
            },
        )
        if not batch:
            break
        rows = [{
            "source_section":     r.get("section"),
            "source_type":        r.get("source_type"),
            "source_id":          r.get("source_id"),
            "title":              r.get("title"),
            "content":            r.get("content"),
            "record":             r.get("record"),
            "source_ingested_at": r.get("ingested_at"),
        } for r in batch]
        written = await _sb_service(
            "POST", "tier4_main",
            params={"on_conflict": "source_section,source_id"},
            json=rows,
            prefer="resolution=merge-duplicates,return=representation",
        )
        moved += len(written or [])
        last_id = batch[-1]["id"]
        if len(batch) < batch_size:
            break
    return {"ok": True, "moved": moved, "section": section}


# Proof the belt ran clean: source count vs landed count, plus a per-shelf catalog
# of everything now sitting on tier4_main.
@app.get("/tier4/status")
async def tier4_status(_: bool = Depends(rt_auth)):
    section = "randy"
    t3 = await _sb_service(
        "GET", "tier3_databank",
        params={"select": "source_id", "section": f"eq.{section}", "limit": "1000000"},
    )
    tier3_count = len(t3 or [])
    t4 = await _sb_service(
        "GET", "tier4_main",
        params={"select": "source_section", "limit": "1000000"},
    )
    sections: dict = {}
    for r in (t4 or []):
        s = r.get("source_section")
        sections[s] = sections.get(s, 0) + 1
    tier4_count = sections.get(section, 0)
    return {"tier3_count": tier3_count, "tier4_count": tier4_count, "sections": sections}


# ── PASTE CHAT -> TIER 3 ─────────────────────────────────────────────────────────
# Clean a pasted Claude conversation with the SAME Anthropic client/key /api/ai uses,
# then file one row to tier3_databank via the same _sb() path /tier3/push uses.
@app.post("/tier3/ingest")
async def tier3_ingest(payload: dict, _: bool = Depends(rt_auth)):
    try:
        raw = (payload or {}).get("text") or ""
        if not raw.strip():
            return {"ok": False, "error": "No text provided"}
        if not client:
            return {"ok": False, "error": "ANTHROPIC_API_KEY not set"}

        instruction = (
            "Clean this exported Claude conversation into one readable record. "
            "Strip UI artifacts, timestamps, and repeated boilerplate. "
            "Return JSON only: {\"title\": a short descriptive title, "
            "\"content\": the cleaned full conversation}."
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                messages=[{"role": "user", "content": f"{instruction}\n\n{raw}"}],
            )
            text = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
        except Exception as e:
            return {"ok": False, "error": f"Claude call failed: {e}"}

        # Strip a ```json fence if Claude wrapped the JSON.
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except Exception as e:
            return {"ok": False, "error": f"Claude did not return valid JSON: {e}"}

        title = parsed.get("title")
        content = parsed.get("content")
        if not title or content is None:
            return {"ok": False, "error": "Claude JSON missing title or content"}

        source_id = "paste-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        row = {
            "section": "randy",
            "source_type": "claude_conversation",
            "source_id": source_id,
            "title": title,
            "content": content,
            "record": {"title": title, "content": content, "raw": raw},
        }
        try:
            await _sb_service(
                "POST", "tier3_databank",
                params={"on_conflict": "section,source_id"},
                json=[row],
                prefer="resolution=merge-duplicates,return=representation",
            )
        except HTTPException as e:
            return {"ok": False, "error": f"Supabase error: {e.detail}"}

        return {"ok": True, "title": title, "source_id": source_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Progyny Infinite — Command Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: 'Outfit', 'Segoe UI', sans-serif;
    min-height: 100vh;
  }
  .topbar {
    background: #111;
    border-bottom: 1px solid #1e1e1e;
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .logo { font-size: 16px; font-weight: 700; color: #c9a84c; letter-spacing: 2px; text-transform: uppercase; }
  .logo span { color: #555; font-weight: 300; }
  .agent-status { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #333; transition: background 0.3s; }
  .status-dot.online { background: #4caf50; box-shadow: 0 0 6px #4caf5066; }
  .nav {
    background: #0f0f0f;
    border-bottom: 1px solid #1a1a1a;
    display: flex;
    overflow-x: auto;
    padding: 0 24px;
    gap: 2px;
  }
  .nav::-webkit-scrollbar { height: 0; }
  .nav-tab {
    padding: 14px 20px;
    font-size: 13px;
    font-weight: 500;
    color: #444;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
    background: none;
    border-top: none; border-left: none; border-right: none;
    font-family: inherit;
  }
  .nav-tab:hover { color: #888; }
  .nav-tab.active { color: #c9a84c; border-bottom-color: #c9a84c; }
  .panel { display: none; padding: 28px 24px; max-width: 1200px; margin: 0 auto; }
  .panel.active { display: block; }
  .card { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }
  .card-title { font-size: 11px; font-weight: 600; color: #c9a84c; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 16px; }
  .projects-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; margin-bottom: 20px; }
  .project-card { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 8px; padding: 18px; transition: border-color 0.2s; }
  .project-card:hover { border-color: #333; }
  .project-name { font-size: 14px; font-weight: 600; color: #e0e0e0; margin-bottom: 6px; }
  .project-status { font-size: 12px; margin-bottom: 10px; }
  .project-status.active { color: #c9a84c; }
  .project-status.backburner { color: #555; }
  .project-status.live { color: #4caf50; }
  .project-desc { font-size: 12px; color: #555; line-height: 1.5; }
  .progress-bar { background: #1a1a1a; border-radius: 3px; height: 4px; margin-top: 12px; }
  .progress-fill { background: #c9a84c; border-radius: 3px; height: 4px; }
  .flags-list { list-style: none; }
  .flags-list li { padding: 10px 0; border-bottom: 1px solid #1a1a1a; font-size: 13px; color: #888; display: flex; align-items: flex-start; gap: 10px; }
  .flags-list li:last-child { border-bottom: none; }
  .flag-num { color: #c9a84c; font-weight: 600; font-size: 11px; min-width: 20px; margin-top: 1px; }
  .tracker-row { display: grid; grid-template-columns: 2fr 1fr 1fr 2fr; gap: 12px; padding: 12px 0; border-bottom: 1px solid #1a1a1a; font-size: 13px; align-items: center; }
  .tracker-header { font-size: 11px; color: #444; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge.priority { background: #1a1208; color: #c9a84c; border: 1px solid #c9a84c44; }
  .badge.active { background: #0d1a0d; color: #4caf50; border: 1px solid #4caf5044; }
  .badge.backburner { background: #1a1a1a; color: #555; border: 1px solid #33333344; }
  .badge.blocked { background: #1a0d0d; color: #e05555; border: 1px solid #e0555544; }
  textarea { width: 100%; background: #0a0a0a; border: 1px solid #222; border-radius: 6px; padding: 14px; color: #e0e0e0; font-size: 13px; font-family: monospace; resize: vertical; min-height: 160px; outline: none; line-height: 1.6; }
  textarea:focus { border-color: #c9a84c; }
  input[type="text"] { width: 100%; background: #0a0a0a; border: 1px solid #222; border-radius: 6px; padding: 11px 14px; color: #e0e0e0; font-size: 14px; font-family: inherit; outline: none; }
  input[type="text"]:focus { border-color: #c9a84c; }
  .btn { background: #c9a84c; border: none; border-radius: 6px; padding: 11px 24px; color: #0a0a0a; font-size: 13px; font-weight: 700; font-family: inherit; cursor: pointer; margin-top: 12px; transition: opacity 0.2s; text-transform: uppercase; letter-spacing: 1px; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost { background: transparent; border: 1px solid #c9a84c; color: #c9a84c; }
  .btn-ghost:hover { background: #c9a84c; color: #0a0a0a; opacity: 1; }
  .result { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 6px; padding: 14px; margin-top: 14px; font-size: 13px; color: #aaa; white-space: pre-wrap; font-family: monospace; min-height: 60px; display: none; line-height: 1.6; }
  .result.visible { display: block; }
  .result.success { border-color: #2d5a2d; color: #7ec87e; }
  .result.error { border-color: #5a2d2d; color: #e07e7e; }
  .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .chip { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; padding: 5px 12px; font-size: 11px; color: #666; cursor: pointer; transition: all 0.2s; font-family: inherit; }
  .chip:hover { border-color: #c9a84c; color: #c9a84c; }
  .score-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .niche-card { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 8px; padding: 16px; }
  .niche-name { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .niche-score { font-size: 28px; font-weight: 700; color: #c9a84c; }
  .niche-verdict { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .niche-verdict.go { color: #4caf50; }
  .niche-verdict.maybe { color: #ff9800; }
  .niche-verdict.tbd { color: #555; }
  .tools-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
  .tool-card { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 8px; padding: 20px; cursor: pointer; transition: border-color 0.2s; }
  .tool-card:hover { border-color: #c9a84c; }
  .tool-icon { font-size: 28px; margin-bottom: 10px; }
  .tool-name { font-size: 14px; font-weight: 600; color: #e0e0e0; margin-bottom: 4px; }
  .tool-desc { font-size: 12px; color: #555; line-height: 1.4; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
  .log-box { background: #060606; border: 1px solid #1a1a1a; border-radius: 6px; padding: 14px; font-size: 12px; color: #555; font-family: monospace; height: 200px; overflow-y: auto; line-height: 1.6; }
  /* ── ReptiTerra feeding tab (scoped, own tokens) ── */
  #panel-feeding { --rt-bg:#0d0d0f; --rt-card:#16181c; --rt-border:#2a2d35; --rt-accent:#e85d2f; --rt-text:#e8e4dc; --rt-muted:#666; --rt-ok:#4ade80; --rt-warn:#facc15; --rt-danger:#f87171; font-family:Georgia,'Times New Roman',serif; }
  #panel-feeding .rt-wrap { max-width:480px; margin:0 auto; color:var(--rt-text); }
  #panel-feeding .rt-title { font-size:18px; color:var(--rt-text); }
  #panel-feeding .rt-subnav { display:flex; gap:6px; margin-bottom:16px; }
  #panel-feeding .rt-subtab { flex:1; text-align:center; padding:10px; background:var(--rt-card); border:1px solid var(--rt-border); border-radius:8px; color:var(--rt-muted); cursor:pointer; font-size:14px; }
  #panel-feeding .rt-subtab.active { color:var(--rt-accent); border-color:var(--rt-accent); }
  #panel-feeding .rt-sub { display:none; }
  #panel-feeding .rt-sub.active { display:block; }
  #panel-feeding .rt-card { background:var(--rt-card); border:1px solid var(--rt-border); border-radius:10px; padding:16px; margin-bottom:12px; }
  #panel-feeding .rt-h { font-size:12px; text-transform:uppercase; letter-spacing:1px; color:var(--rt-muted); margin-bottom:10px; }
  #panel-feeding .rt-row { display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--rt-border); gap:8px; }
  #panel-feeding .rt-row:last-child { border-bottom:none; }
  #panel-feeding .rt-btn { background:var(--rt-accent); color:#0d0d0f; border:none; border-radius:6px; padding:9px 14px; font-family:inherit; font-weight:bold; cursor:pointer; font-size:13px; }
  #panel-feeding .rt-btn.ghost { background:transparent; color:var(--rt-accent); border:1px solid var(--rt-accent); }
  #panel-feeding .rt-btn.sm { padding:3px 9px; font-size:15px; line-height:1; }
  #panel-feeding .rt-input { width:100%; background:var(--rt-bg); border:1px solid var(--rt-border); border-radius:6px; padding:10px; color:var(--rt-text); font-family:inherit; font-size:14px; margin-bottom:10px; }
  #panel-feeding .rt-count { font-size:16px; min-width:32px; text-align:center; }
  #panel-feeding .rt-badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:10px; font-weight:bold; }
  #panel-feeding .rt-badge.low { background:#2a2410; color:var(--rt-warn); border:1px solid var(--rt-warn); }
  #panel-feeding .rt-badge.danger { background:#2a1010; color:var(--rt-danger); border:1px solid var(--rt-danger); }
  #panel-feeding .rt-statusbar { height:6px; border-radius:3px; background:var(--rt-border); overflow:hidden; margin:10px 0; }
  #panel-feeding .rt-chip { display:inline-block; width:12px; height:12px; border-radius:3px; margin-left:3px; vertical-align:middle; }
  #panel-feeding .rt-muted { color:var(--rt-muted); font-size:12px; }
  #panel-feeding .rt-lab { display:block; font-size:12px; color:var(--rt-muted); margin-bottom:4px; }
  #panel-feeding .rt-modal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,.72); z-index:500; align-items:center; justify-content:center; padding:16px; }
  #panel-feeding .rt-modal-bg.open { display:flex; }
  #panel-feeding .rt-modal { background:var(--rt-card); border:1px solid var(--rt-border); border-radius:12px; padding:20px; max-width:420px; width:100%; max-height:88vh; overflow-y:auto; }
  /* ── Sales Tracker (renders on Command Center) — scoped tokens, square corners ── */
  #panel-command .st { --st-bg:#0e0f12; --st-card:#15161b; --st-raised:#1b1d24; --st-border:#26282f; --st-gold:#c9a84c; --st-text:#e8e6df; --st-muted:#7a7e89; --st-dim:#50545f; --st-pos:#4ade80; --st-neg:#f87171;
    background:var(--st-bg); border:1px solid var(--st-border); color:var(--st-text); margin-top:22px; font-variant-numeric:tabular-nums; }
  #panel-command .st * { box-sizing:border-box; }
  #panel-command .st-head { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; padding:16px 20px; border-bottom:1px solid var(--st-border); }
  #panel-command .st-title { font-size:15px; font-weight:700; letter-spacing:4px; text-transform:uppercase; color:var(--st-text); }
  #panel-command .st-asof { font-size:11px; color:var(--st-muted); text-transform:uppercase; letter-spacing:1px; margin-left:12px; }
  #panel-command .st-stale { display:inline-block; margin-left:8px; padding:2px 8px; font-size:10px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--st-neg); border:1px solid var(--st-neg); }
  #panel-command .st-toggle { display:flex; border:1px solid var(--st-border); }
  #panel-command .st-toggle button { background:transparent; border:none; border-left:1px solid var(--st-border); color:var(--st-muted); font-family:inherit; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; padding:8px 14px; cursor:pointer; }
  #panel-command .st-toggle button:first-child { border-left:none; }
  #panel-command .st-toggle button.active { background:var(--st-raised); color:var(--st-gold); }
  #panel-command .st-body { padding:18px 20px; }
  #panel-command .st-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }
  @media (max-width:820px) { #panel-command .st-kpis { grid-template-columns:repeat(2,1fr); } }
  #panel-command .st-kpi { background:var(--st-card); border:1px solid var(--st-border); padding:14px 16px; }
  #panel-command .st-kpi-lab { font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:var(--st-muted); margin-bottom:8px; }
  #panel-command .st-kpi-val { font-size:24px; font-weight:700; color:var(--st-text); line-height:1.1; }
  #panel-command .st-kpi-sub { font-size:11px; margin-top:6px; color:var(--st-dim); }
  #panel-command .st-switch { display:flex; flex-wrap:wrap; gap:2px; margin-bottom:16px; border:1px solid var(--st-border); width:fit-content; }
  #panel-command .st-switch button { background:transparent; border:none; color:var(--st-muted); font-family:inherit; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; padding:7px 13px; cursor:pointer; }
  #panel-command .st-switch button.active { background:var(--st-gold); color:#0e0f12; }
  #panel-command .st-scroll { overflow-x:auto; }
  #panel-command table.st-matrix { border-collapse:collapse; width:100%; font-size:12px; }
  #panel-command .st-matrix th, #panel-command .st-matrix td { border:1px solid var(--st-border); padding:7px 10px; text-align:right; white-space:nowrap; }
  #panel-command .st-matrix thead th { font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--st-muted); font-weight:600; background:var(--st-raised); }
  #panel-command .st-matrix th.st-co, #panel-command .st-matrix td.st-co { text-align:left; color:var(--st-text); }
  #panel-command .st-matrix .st-cur { color:var(--st-gold) !important; }
  #panel-command .st-matrix tr.st-clickable { cursor:pointer; }
  #panel-command .st-matrix tr.st-clickable:hover td { background:var(--st-raised); }
  #panel-command .st-matrix tr.st-total td { border-top:2px solid var(--st-gold); font-weight:700; color:var(--st-text); background:var(--st-card); }
  #panel-command .st-dot { display:inline-block; width:8px; height:8px; margin-right:7px; vertical-align:middle; }
  #panel-command .st-mom-pos { color:var(--st-pos); }
  #panel-command .st-mom-neg { color:var(--st-neg); }
  #panel-command .st-mom-flat { color:var(--st-dim); }
  #panel-command .st-book { background:var(--st-bg); }
  #panel-command .st-book td { padding:0 !important; border:1px solid var(--st-border) !important; }
  #panel-command table.st-sub { border-collapse:collapse; width:100%; font-size:11px; }
  #panel-command .st-sub th, #panel-command .st-sub td { border:1px solid var(--st-border); padding:6px 10px; text-align:right; white-space:nowrap; }
  #panel-command .st-sub th { font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--st-muted); font-weight:600; }
  #panel-command .st-sub th.st-co, #panel-command .st-sub td.st-co { text-align:left; color:var(--st-muted); }
  #panel-command .st-cons-chart { background:var(--st-card); border:1px solid var(--st-border); padding:16px; margin-bottom:16px; }
  #panel-command .st-legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; font-size:11px; color:var(--st-muted); }
  #panel-command .st-legend span { display:inline-flex; align-items:center; gap:6px; }
  #panel-command .st-msg { padding:24px; text-align:center; color:var(--st-muted); font-size:13px; }
  #panel-command .st-msg.err { color:var(--st-neg); }
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">Progyny <span>/</span> Infinite</div>
  <div class="agent-status">
    <div class="status-dot" id="agentDot"></div>
    <span id="agentStatus">Connecting...</span>
  </div>
</div>

<nav class="nav">
  <button class="nav-tab active" onclick="switchTab('command')">Command Center</button>
  <button class="nav-tab" onclick="switchTab('finance')">Finance AI</button>
  <button class="nav-tab" id="rtsub-feed-nav" onclick="switchTab('feeding')">Indigo</button>
  <button class="nav-tab" onclick="switchTab('youtube')">YouTube Extractor</button>
  <button class="nav-tab" onclick="switchTab('tools')">Tools</button>
  <!-- Tier 3 Paste tab hidden from UI. Backend /tier3/ingest route and #panel-tier3 stay intact, just unreachable from the nav.
  <button class="nav-tab" onclick="switchTab('tier3')">Tier 3 Paste</button>
  -->
</nav>

<!-- COMMAND CENTER -->
<div class="panel active" id="panel-command">
  <div class="projects-grid">
    <div class="project-card">
      <div class="project-name">⭐ ProgenyVault</div>
      <div class="project-status active">Active Build — Highest Priority</div>
      <div class="project-desc">Exotic animal breeding records SaaS. Next.js / Supabase / Stripe. $29.99/mo single Pro tier.</div>
      <div class="progress-bar"><div class="progress-fill" style="width:45%"></div></div>
      <div style="font-size:11px;color:#444;margin-top:6px;">~45% complete — 18-25 hrs remaining</div>
    </div>
    <div class="project-card">
      <div class="project-name">🐍 Den of Indigos</div>
      <div class="project-status active">Active — Top of Funnel</div>
      <div class="project-desc">Content hub centered on Brigs. Kit email list. YouTube / TikTok. Lead magnets live.</div>
      <div class="progress-bar"><div class="progress-fill" style="width:60%"></div></div>
    </div>
    <div class="project-card">
      <div class="project-name">🧪 ReptiTerra Labs</div>
      <div class="project-status backburner">Back Burner</div>
      <div class="project-desc">Digital substrate PDFs on Gumroad. $9.99/blend. Listing copy incomplete.</div>
      <div class="progress-bar"><div class="progress-fill" style="width:70%"></div></div>
    </div>
    <div class="project-card">
      <div class="project-name">📈 Vault Trader</div>
      <div class="project-status backburner">Built — Undeployed</div>
      <div class="project-desc">Personal trading agent. RSI + Bollinger Bands. Blocked on Alpaca 2FA (Authy).</div>
      <div class="progress-bar"><div class="progress-fill" style="width:90%"></div></div>
    </div>
    <div class="project-card">
      <div class="project-name">🏭 Progyny Infinite Trust</div>
      <div class="project-status backburner">Pending DOR Audit</div>
      <div class="project-desc">Oregon Business Trust formation. Letter expected June 30. Bring to Claude first.</div>
      <div class="progress-bar"><div class="progress-fill" style="width:20%"></div></div>
    </div>
    <div class="project-card">
      <div class="project-name">🧠 PICP + Agent</div>
      <div class="project-status live">Live ✓</div>
      <div class="project-desc">Brain is running on Railway. Dashboard connected. Self-improving.</div>
      <div class="progress-bar"><div class="progress-fill" style="width:85%"></div></div>
    </div>
  </div>
  <!-- Command Center selector: gates Sales / Agent / Marketing / Tracker / Niche below the project cards. Empty by default. -->
  <div style="margin-top:22px;">
    <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">View</label>
    <select id="ccMenu" onchange="ccShow(this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:10px 12px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;min-width:220px;">
      <option value="">-- Select --</option>
      <option value="sales">Sales</option>
      <option value="agent">Agent Control</option>
      <option value="marketing">Marketing</option>
      <option value="tracker">Project Tracker</option>
      <option value="niche">Niche Scorer</option>
      <option value="employees">Employees</option>
    </select>
  </div>

  <!-- CC: SALES -->
  <div id="cc-sales" style="display:none;margin-top:18px;">
    <!-- SALES TRACKER — renders on Command Center. Reads Tier 3 section sales_status. -->
    <div class="st" id="salesTracker">
      <div class="st-head">
        <div>
          <span class="st-title">Sales</span>
          <span class="st-asof" id="stAsOf"></span>
        </div>
        <div class="st-toggle" id="stToggle">
          <button class="active" onclick="salesSetView('books')">Tier 3 · Books</button>
          <button onclick="salesSetView('consolidated')">Tier 4 · Consolidated</button>
        </div>
      </div>
      <div class="st-body" id="stBody">
        <div class="st-msg" id="stMsg">Loading sales…</div>
      </div>
    </div>
  </div>

  <!-- CC: AGENT CONTROL — Tier 3 -> Tier 4 belt -->
  <div id="cc-agent" style="display:none;margin-top:18px;">
    <div class="two-col">
      <div>
        <div class="card">
          <div class="card-title">Run Load</div>
          <p style="font-size:13px;color:#555;margin-bottom:14px;">Move every Tier 3 record in section <strong>randy</strong> onto the Tier 4 belt. Safe to re-run — it upserts, never duplicates.</p>
          <button class="btn" onclick="runLoad()">Run Load</button>
          <div class="result" id="loadResult"></div>
        </div>
        <div class="card">
          <div class="card-title">Gauge</div>
          <p style="font-size:13px;color:#555;margin-bottom:14px;">Tier 3 source vs Tier 4 landed. Green when the belt emptied clean.</p>
          <div id="gaugeBox" style="padding:16px;border-radius:8px;background:#f6f6f4;text-align:center;border-left:4px solid #ccc;">Refresh to read the gauge</div>
          <button class="btn btn-ghost" onclick="loadStatus()" style="margin-top:10px;">Refresh Gauge</button>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-title">Catalog</div>
          <p style="font-size:13px;color:#555;margin-bottom:14px;">What landed on Tier 4, by shelf (source section).</p>
          <div class="log-box" id="catalogBox">Click refresh to load the catalog</div>
          <button class="btn btn-ghost" onclick="loadStatus()" style="margin-top:10px;">Refresh Catalog</button>
        </div>
      </div>
    </div>
  </div>

  <!-- CC: MARKETING -->
  <div id="cc-marketing" style="display:none;margin-top:18px;">
    <div class="card">
      <div class="card-title">Marketing — Agent Board</div>
      <p style="font-size:13px;color:#555;margin-bottom:14px;">Live from the Tier 3 <strong>marketing_status</strong> card. Pick a company and an as-of date to view its board.</p>
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;">
        <div>
          <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Company</label>
          <select id="mktCompany" onchange="mktOnCompanyChange()" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:10px 12px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;min-width:190px;"></select>
        </div>
        <div>
          <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">As of</label>
          <select id="mktDate" onchange="mktRenderBoard()" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:10px 12px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;min-width:170px;"></select>
        </div>
        <button class="btn btn-ghost" onclick="loadMarketing()" style="margin-top:0;">Refresh Board</button>
      </div>
      <div class="result" id="marketingMeta" style="margin-top:14px;"></div>
    </div>
    <div id="marketingBoard"></div>
  </div>

  <!-- CC: PROJECT TRACKER -->
  <div id="cc-tracker" style="display:none;margin-top:18px;">
    <div class="card">
      <div class="card-title">Active Builds</div>
      <div class="tracker-row tracker-header"><div>Project</div><div>Status</div><div>Progress</div><div>Next Action</div></div>
      <div class="tracker-row"><div style="font-weight:600">ProgenyVault</div><div><span class="badge priority">Priority</span></div><div style="color:#c9a84c">45%</div><div style="color:#888;font-size:12px">Stack orientation → resume build</div></div>
      <div class="tracker-row"><div style="font-weight:600">Den of Indigos</div><div><span class="badge active">Active</span></div><div style="color:#4caf50">60%</div><div style="color:#888;font-size:12px">Resend domain verify → email sequences live</div></div>
      <div class="tracker-row"><div style="font-weight:600">ReptiTerra Labs</div><div><span class="badge backburner">Backburner</span></div><div style="color:#555">70%</div><div style="color:#888;font-size:12px">Write Tropical + Tunnel Blend copy</div></div>
      <div class="tracker-row"><div style="font-weight:600">Vault Trader</div><div><span class="badge blocked">Blocked</span></div><div style="color:#e05555">90%</div><div style="color:#888;font-size:12px">Authy 2FA → Alpaca keys → Railway deploy</div></div>
      <div class="tracker-row"><div style="font-weight:600">Progyny Infinite Trust</div><div><span class="badge blocked">Pending</span></div><div style="color:#e05555">20%</div><div style="color:#888;font-size:12px">Wait for DOR audit letter June 30</div></div>
      <div class="tracker-row"><div style="font-weight:600">PICP + Agent + Dashboard</div><div><span class="badge active">Live</span></div><div style="color:#4caf50">90%</div><div style="color:#888;font-size:12px">Package for Shawn Dark + user manual</div></div>
    </div>
    <div class="card">
      <div class="card-title">Factory Pipeline</div>
      <div class="tracker-row tracker-header"><div>Niche</div><div>Score</div><div>Status</div><div>Notes</div></div>
      <div class="tracker-row"><div>Koi / Fancy Goldfish</div><div style="color:#c9a84c;font-weight:700">42</div><div><span class="badge active">GO</span></div><div style="font-size:12px;color:#666">Highest score</div></div>
      <div class="tracker-row"><div>Tarantula / Invert</div><div style="color:#c9a84c;font-weight:700">41</div><div><span class="badge active">GO</span></div><div style="font-size:12px;color:#666">Huge community</div></div>
      <div class="tracker-row"><div>Dart Frog</div><div style="color:#c9a84c;font-weight:700">40</div><div><span class="badge active">GO</span></div><div style="font-size:12px;color:#666">Complex genetics</div></div>
      <div class="tracker-row"><div>Small Livestock</div><div style="color:#c9a84c;font-weight:700">38</div><div><span class="badge active">GO</span></div><div style="font-size:12px;color:#666">Fastest — PV reuse</div></div>
      <div class="tracker-row"><div>Entertainer Analytics</div><div style="color:#555">TBD</div><div><span class="badge backburner">Queued</span></div><div style="font-size:12px;color:#666">$14.99/mo, stigma moat</div></div>
      <div class="tracker-row"><div>Philately</div><div style="color:#555">TBD</div><div><span class="badge backburner">Queued</span></div><div style="font-size:12px;color:#666">$19.99/mo, $3.4B market</div></div>
    </div>
    <div class="card">
      <div class="card-title">Open Flags</div>
      <ul class="flags-list">
        <li><span class="flag-num">01</span>Stack orientation session required before ProgenyVault build resumes</li>
        <li><span class="flag-num">02</span>Alpaca 2FA — Authy from App Store, unblocks Vault Trader</li>
        <li><span class="flag-num">03</span>Resend domain verification — all three domains pending</li>
        <li><span class="flag-num">04</span>Oregon DOR audit letter by June 30 — bring to Claude first, no action before that</li>
        <li><span class="flag-num">05</span>Back garage door — 16" overfit unresolved</li>
        <li><span class="flag-num">06</span>Greywater permit — Oregon DEQ, not started</li>
        <li><span class="flag-num">07</span>Unknown AWT advisor — "master of micro business" identity unconfirmed</li>
        <li><span class="flag-num">08</span>Third Factory niche — pending</li>
        <li><span class="flag-num">09</span>ReptiTerra — Tropical and Tunnel Blend listing copy not written</li>
        <li><span class="flag-num">10</span>Self-hosting review — July 7</li>
      </ul>
    </div>
  </div>

  <!-- CC: NICHE SCORER -->
  <div id="cc-niche" style="display:none;margin-top:18px;">
    <div class="card">
      <div class="card-title">Score a New Niche</div>
      <input type="text" id="nicheInput" placeholder="Describe the niche (e.g. beekeeping operations tracker)" />
      <button class="btn" onclick="scoreNiche()">Score It</button>
      <div class="result" id="nicheResult"></div>
    </div>
    <div class="card">
      <div class="card-title">Current Pipeline</div>
      <div class="score-grid">
        <div class="niche-card"><div class="niche-name">Koi / Fancy Goldfish</div><div class="niche-score">42</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Tarantula / Invert</div><div class="niche-score">41</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Dart Frog</div><div class="niche-score">40</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Mushroom Cultivation</div><div class="niche-score">39</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Hotshot Trucking</div><div class="niche-score">39</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Small Livestock</div><div class="niche-score">38</div><div class="niche-verdict go">GO</div></div>
        <div class="niche-card"><div class="niche-name">Microgreens</div><div class="niche-score">35</div><div class="niche-verdict maybe">MAYBE</div></div>
        <div class="niche-card"><div class="niche-name">Entertainer Analytics</div><div class="niche-score">—</div><div class="niche-verdict tbd">Queued</div></div>
        <div class="niche-card"><div class="niche-name">Philately</div><div class="niche-score">—</div><div class="niche-verdict tbd">Queued</div></div>
      </div>
    </div>
  </div>

  <!-- CC: EMPLOYEES -->
  <div id="cc-employees" style="display:none;margin-top:18px;">
    <div class="card">
      <div class="card-title">Employees</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="text-align:left;color:#666;font-size:11px;text-transform:uppercase;letter-spacing:1px;">
            <th style="padding:8px 10px;border-bottom:1px solid #222;">Name</th>
            <th style="padding:8px 10px;border-bottom:1px solid #222;">Natural Role</th>
            <th style="padding:8px 10px;border-bottom:1px solid #222;">Assigned Role</th>
            <th style="padding:8px 10px;border-bottom:1px solid #222;">Design Team</th>
          </tr>
        </thead>
        <tbody id="empBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- YOUTUBE EXTRACTOR -->
<div class="panel" id="panel-youtube">
  <div class="card">
    <div class="card-title">YouTube Video Extractor</div>
    <p style="font-size:13px;color:#555;margin-bottom:14px;">Paste a YouTube URL — Claude watches it so you don't have to.</p>
    <input type="text" id="ytUrl" placeholder="https://www.youtube.com/watch?v=..." />
    <div class="chip-row" style="margin-top:12px;">
      <button class="chip" onclick="document.getElementById('ytMode').value='wiki';this.style.color='#c9a84c'">Wiki Seed</button>
      <button class="chip" onclick="document.getElementById('ytMode').value='build'">Build Extract</button>
      <button class="chip" onclick="document.getElementById('ytMode').value='summary'">Quick Summary</button>
      <button class="chip" onclick="document.getElementById('ytMode').value='picp'">PICP Intel</button>
    </div>
    <input type="text" id="ytMode" value="wiki" style="display:none;" />
    <button class="btn" onclick="extractYoutube()">Extract</button>
    <div class="result" id="ytResult"></div>
  </div>
</div>

<!-- PASTE CHAT -> TIER 3 -->
<div class="panel" id="panel-tier3">
  <div class="card">
    <div class="card-title">Paste Chat → Tier 3</div>
    <p style="font-size:13px;color:#555;margin-bottom:14px;">Paste an exported Claude conversation — it gets cleaned and filed to the Tier 3 databank.</p>
    <textarea id="tier3Paste" placeholder="Paste the full exported Claude conversation here..."></textarea>
    <button class="btn" onclick="ingestTier3()">Clean &amp; File to Tier 3</button>
    <div class="result" id="tier3Status"></div>
  </div>
</div>

<!-- FINANCE AI -->
<div class="panel" id="panel-finance">
  <div class="card">
    <div class="card-title">Market Query</div>
    <input type="text" id="financeInput" placeholder="Ask about a stock, market, or financial topic..." />
    <button class="btn" onclick="queryFinance()">Search Markets</button>
    <div class="result" id="financeResult"></div>
  </div>
  <div class="card">
    <div class="card-title">Vault Trader Status</div>
    <div style="font-size:13px;color:#555;line-height:1.7;">
      <div>Strategy: RSI + Bollinger Bands + Volume Spike</div>
      <div>Universe: 50 large-cap symbols | Position: 15% | Stop: 3% | TP: 5%</div>
      <div style="margin-top:10px;color:#e05555;">Status: Undeployed — Blocked on Alpaca 2FA (Authy)</div>
    </div>
  </div>
</div>

<!-- TOOLS -->
<div class="panel" id="panel-tools">
  <div class="tools-grid">
    <div class="tool-card" onclick="switchTab('command');document.getElementById('ccMenu').value='niche';ccShow('niche')"><div class="tool-icon">🎯</div><div class="tool-name">Niche Scorer</div><div class="tool-desc">Score any niche against the 5-criteria Factory formula</div></div>
    <div class="tool-card" onclick="switchTab('youtube')"><div class="tool-icon">▶️</div><div class="tool-name">YouTube Extractor</div><div class="tool-desc">Extract concepts from any YouTube video automatically</div></div>
    <div class="tool-card" onclick="switchTab('finance')"><div class="tool-icon">📈</div><div class="tool-name">Finance AI</div><div class="tool-desc">Market queries, stock lookup, Vault Trader status</div></div>
    <div class="tool-card" onclick="switchTab('command');document.getElementById('ccMenu').value='agent';ccShow('agent')"><div class="tool-icon">🧠</div><div class="tool-name">Brain Control</div><div class="tool-desc">Send context to the PICP agent, query the brain</div></div>
    <div class="tool-card" onclick="window.open('https://claude.ai','_blank')"><div class="tool-icon">⚡</div><div class="tool-name">Open Claude</div><div class="tool-desc">Launch Claude in a new tab for working sessions</div></div>
    <div class="tool-card" onclick="window.open('https://github.com/randynutt31','_blank')"><div class="tool-icon">🐙</div><div class="tool-name">GitHub</div><div class="tool-desc">View and manage your repos</div></div>
    <div class="tool-card" onclick="window.open('https://railway.app','_blank')"><div class="tool-icon">🚂</div><div class="tool-name">Railway</div><div class="tool-desc">Monitor deployed services and logs</div></div>
    <div class="tool-card" onclick="window.open('https://supabase.com','_blank')"><div class="tool-icon">🗄️</div><div class="tool-name">Supabase</div><div class="tool-desc">Database management for all products</div></div>
  </div>
</div>

<!-- REPTITERRA FEEDING -->
<div class="panel" id="panel-feeding">
  <div class="rt-wrap">
    <div class="rt-subnav">
      <div class="rt-subtab active" id="rtsub-animals" onclick="rtSwitch('animals')">Animals</div>
      <div class="rt-subtab" id="rtsub-feed" onclick="rtSwitch('feed')">Feed</div>
      <div class="rt-subtab" id="rtsub-supplies" onclick="rtSwitch('supplies')">Supplies</div>
    </div>
    <div class="rt-sub active" id="rtpane-animals">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div class="rt-title">Animals</div><button class="rt-btn" onclick="rtOpenAddAnimal()">+ Animal</button>
      </div>
      <div id="rtAnimals"></div>
    </div>
    <div class="rt-sub" id="rtpane-feed">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div class="rt-title">Feed Inventory</div><button class="rt-btn" onclick="rtOpenDelivery()">+ Delivery</button>
      </div>
      <div id="rtFeed"></div>
    </div>
    <div class="rt-sub" id="rtpane-supplies">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div class="rt-title">Supplies</div><button class="rt-btn" onclick="rtOpenAddSupply()">+ Supply</button>
      </div>
      <div id="rtSupplies"></div>
    </div>
  </div>
  <div class="rt-modal-bg" id="rtModalBg"><div class="rt-modal" id="rtModal"></div></div>
</div>

<script>
// All API calls go through our own server — no CORS issues
async function callServer(endpoint, body) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return res.json();
}

// Check agent status
async function checkAgent() {
  try {
    const data = await fetch('/api/agent-status').then(r => r.json());
    document.getElementById('agentDot').className = 'status-dot ' + (data.online ? 'online' : '');
    document.getElementById('agentStatus').textContent = data.online ? 'Brain online' : 'Brain offline';
  } catch(e) {
    document.getElementById('agentStatus').textContent = 'Status unknown';
  }
}
checkAgent();
setInterval(checkAgent, 30000);
// Sales no longer auto-loads on page/tab entry — it loads only when the Command
// Center "Sales" dropdown option is picked (ccShow), keeping the empty default.

function switchTab(tab) {
  const tabs = ['command','finance','feeding','youtube','tools','tier3'];
  document.querySelectorAll('.nav-tab').forEach((t,i) => t.classList.toggle('active', tabs[i] === tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  if (tab === 'feeding') rtSwitch(RT.sub);
}

// Command Center dropdown: gate Sales / Agent / Marketing / Tracker / Niche below
// the project cards. Empty value ("-- Select --") hides all — the empty default.
// Sales and Marketing load only when their option is selected (not on tab entry).
function ccShow(val) {
  ['cc-sales','cc-agent','cc-marketing','cc-tracker','cc-niche','cc-employees'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  if (!val) return;
  const el = document.getElementById('cc-' + val);
  if (el) el.style.display = 'block';
  if (val === 'sales') loadSalesTracker();
  if (val === 'marketing') loadMarketing();
  if (val === 'employees') loadEmployees();
}

// EMPLOYEES — fixed roster. Name + Natural Role read-only; Assigned Role persists
// in localStorage keyed to employee name; Design Team is locked (Torvalds/Spolsky/Ive).
const EMP_ROSTER = [
  { name: 'Spunky',   natural: 'Project Manager',   design: false },
  { name: 'Dimon',    natural: 'General Manager',   design: false },
  { name: 'Munger',   natural: 'The Destroyer',     design: false },
  { name: 'Leonard',  natural: 'Capital Allocator', design: false },
  { name: 'Brigs',    natural: 'Tax Attorney',      design: false },
  { name: 'Siggy',    natural: 'Lawyer',            design: false },
  { name: 'Deming',   natural: 'Systems Engineer',  design: false },
  { name: 'Ogilvy',   natural: 'Copywriter',        design: false },
  { name: 'Drucker',  natural: 'Operations Mind',   design: false },
  { name: 'Torvalds', natural: 'The Engine',        design: true  },
  { name: 'Spolsky',  natural: 'The Craft',         design: true  },
  { name: 'Ive',      natural: 'The Feel',          design: true  },
];
const EMP_ROLES = ['Sales','Marketing','Operations','Legal','Tax','Engineering','Systems','Copywriting','Capital','—'];
function empKey(name) { return 'emp_role_' + name; }
function empSave(name, val) { localStorage.setItem(empKey(name), val); }
function loadEmployees() {
  const body = document.getElementById('empBody');
  if (!body) return;
  body.innerHTML = EMP_ROSTER.map(e => {
    const saved = localStorage.getItem(empKey(e.name)) || '—';
    const opts = EMP_ROLES.map(r => '<option value="' + r + '"' + (r === saved ? ' selected' : '') + '>' + r + '</option>').join('');
    const design = e.design
      ? '<span style="color:#4caf50;font-weight:700;">✓</span>'
      : '<span style="color:#444;">N/A</span>';
    return '<tr>' +
      '<td style="padding:8px 10px;border-bottom:1px solid #2a2d35;font-weight:600;color:#e0e0e0;">' + e.name + '</td>' +
      '<td style="padding:8px 10px;border-bottom:1px solid #2a2d35;color:#aaa;">' + e.natural + '</td>' +
      '<td style="padding:8px 10px;border-bottom:1px solid #2a2d35;">' +
        '<select onchange="empSave(\'' + e.name + '\', this.value)" ' +
          'style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;">' +
          opts +
        '</select></td>' +
      '<td style="padding:8px 10px;border-bottom:1px solid #2a2d35;">' + design + '</td>' +
      '</tr>';
  }).join('');
}

// AGENT CONTROL — Tier 3 -> Tier 4 belt
const T4_TOKEN = 'RT_DASH_7f39c2a4b8e15d60';

async function runLoad() {
  const result = document.getElementById('loadResult');
  result.textContent = '⬤ Running load...';
  result.className = 'result visible';
  try {
    const res = await fetch('/tier4/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Token': T4_TOKEN}
    });
    const data = await res.json();
    if (!res.ok || data.ok !== true) throw new Error(data.detail || 'Load failed');
    result.className = 'result visible success';
    result.textContent = 'Moved ' + data.moved + ' records from section [' + data.section + '].';
    loadStatus();
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

async function loadStatus() {
  const cat = document.getElementById('catalogBox');
  const gauge = document.getElementById('gaugeBox');
  cat.textContent = 'Loading...';
  try {
    const res = await fetch('/tier4/status', {headers: {'X-API-Token': T4_TOKEN}});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Status failed');
    const secs = data.sections || {};
    const keys = Object.keys(secs).sort();
    cat.textContent = keys.length
      ? keys.map(k => k + '  →  ' + secs[k] + ' records').join('\\n')
      : 'Tier 4 is empty — nothing landed yet.';
    const t3 = data.tier3_count, t4 = data.tier4_count;
    const match = (t3 === t4);
    gauge.innerHTML =
      '<div style="font-size:24px;font-weight:700;">' + t4 + ' / ' + t3 + '</div>' +
      '<div style="font-size:12px;color:#555;margin-top:4px;">Tier 4 landed / Tier 3 source</div>' +
      '<div style="margin-top:8px;font-weight:600;">' +
        (match ? '✓ Belt emptied clean' : '⚠ ' + (t3 - t4) + ' still on the belt') + '</div>';
    gauge.style.borderLeft = '4px solid ' + (match ? '#2ea043' : '#e05555');
    gauge.style.background = match ? '#eaf7ed' : '#fbeaea';
    gauge.style.color = match ? '#1a7f37' : '#b02a2a';
  } catch(e) {
    cat.textContent = 'Could not load catalog: ' + e.message;
    gauge.textContent = 'Gauge unavailable';
  }
}

// MARKETING — read the marketing_status card via /tier3/records, render the agent board
function mktEsc(s) {
  return (s == null ? '' : String(s)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function mktAgentCard(a) {
  const produced = Object.entries(a.produced || {})
    .map(([k,v]) => '<span class="badge active" style="margin-right:6px;">' + mktEsc(k) + ': ' + mktEsc(v) + '</span>')
    .join('') || '<span style="color:#555;font-size:12px;">nothing produced</span>';
  const waiting = (a.waiting_on && a.waiting_on.length)
    ? '<ul class="flags-list" style="margin-top:10px;">' +
        a.waiting_on.map(w => '<li><span class="flag-num">!</span>' + mktEsc(w) + '</li>').join('') +
      '</ul>'
    : '<div style="font-size:12px;color:#4caf50;margin-top:10px;">✓ nothing outstanding</div>';
  return '<div class="project-card" style="margin-bottom:12px;">' +
    '<div class="project-name">' + mktEsc(a.agent) + '</div>' +
    '<div class="project-status" style="color:#888;">last run: ' + mktEsc(a.last_run ? mktHumanDateTime(a.last_run) : '—') + '</div>' +
    '<div style="margin-top:10px;">' + produced + '</div>' +
    waiting +
    '</div>';
}
// Parse a record's manifest, keeping the rec.record -> JSON.parse(rec.content) fallback.
function mktManifest(rec) {
  let mf = rec.record;
  if (!mf && rec.content) { try { mf = JSON.parse(rec.content); } catch(e) {} }
  return mf;
}
// Selector-model state. Records grouped by company slug; each company holds
// dated snapshots. The Company + As-of dropdowns pick which single board renders.
const MKT = { byCompany: {}, order: [], selCompany: null, selDate: null };

// "YYYY-MM-DD" -> "Jul 10, 2026". Parsed by parts to avoid timezone drift; no ISO shown.
function mktHumanDate(ymd) {
  if (!ymd) return '—';
  const p = String(ymd).split('-');
  if (p.length !== 3) return String(ymd);
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const mi = parseInt(p[1], 10) - 1;
  if (isNaN(mi) || mi < 0 || mi > 11) return String(ymd);
  return months[mi] + ' ' + parseInt(p[2], 10) + ', ' + p[0];
}

// ISO timestamp -> "Jul 10, 2026, 7:32 PM" (keeps time-of-day, no raw ISO on the tab).
// Parsed positionally from the string to avoid browser-local timezone drift.
function mktHumanDateTime(iso) {
  if (!iso) return '—';
  const s = String(iso);
  const ymd = s.slice(0, 10);
  const d = mktHumanDate(ymd);
  if (d === ymd) return s;                 // not date-prefixed -> return as-is (rare)
  const hStr = s.slice(11, 13), mStr = s.slice(14, 16);
  const h = parseInt(hStr, 10), mnum = parseInt(mStr, 10);
  if (isNaN(h) || isNaN(mnum) || hStr.length < 2 || mStr.length < 2) return d;  // date only
  const ampm = h >= 12 ? 'PM' : 'AM';
  let hr = h % 12; if (hr === 0) hr = 12;
  return d + ', ' + hr + ':' + mStr + ' ' + ampm;
}

async function loadMarketing() {
  const meta = document.getElementById('marketingMeta');
  const board = document.getElementById('marketingBoard');
  const coSel = document.getElementById('mktCompany');
  const dateSel = document.getElementById('mktDate');
  const prevCompany = MKT.selCompany, prevDate = MKT.selDate;   // preserve across refresh
  meta.className = 'result visible';
  meta.textContent = '⬤ Loading marketing board...';
  try {
    const res = await fetch('/tier3/records?section=marketing_status&limit=500', {
      headers: {'X-API-Token': T4_TOKEN}
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    const recs = data.records || [];
    const byCompany = {};
    for (const rec of recs) {
      const manifest = mktManifest(rec);
      const sid = String(rec.source_id || '');
      const ci = sid.indexOf(':');
      // source_id: "{slug}" (legacy) or "{slug}:{YYYY-MM-DD}" (dated).
      const slug = (ci >= 0 ? sid.slice(0, ci) : sid) || (manifest && manifest.business) || '';
      if (!slug) continue;
      const gen = manifest && manifest.generated_at;
      // Dated record -> date from source_id; legacy -> date part of generated_at.
      const date = ci >= 0 ? sid.slice(ci + 1) : (gen ? String(gen).slice(0, 10) : null);
      if (!date) continue;
      const ts = gen ? Date.parse(gen) : Date.parse(date + 'T00:00:00Z');
      if (!byCompany[slug]) byCompany[slug] = { slug, display: slug, snapshots: [] };
      if (manifest && manifest.display_name) byCompany[slug].display = manifest.display_name;
      byCompany[slug].snapshots.push({ date, ts: isNaN(ts) ? null : ts, manifest });
    }
    // Per company: dedupe by date (keep newest generated_at), sort newest date first.
    for (const slug in byCompany) {
      const seen = {};
      for (const s of byCompany[slug].snapshots) {
        if (!seen[s.date] || (s.ts || 0) > (seen[s.date].ts || 0)) seen[s.date] = s;
      }
      byCompany[slug].snapshots = Object.keys(seen).map(d => seen[d])
        .sort((a, b) => a.date < b.date ? 1 : (a.date > b.date ? -1 : 0));
    }
    const order = Object.keys(byCompany).sort((a, b) => a.localeCompare(b));
    MKT.byCompany = byCompany;
    MKT.order = order;
    if (!order.length) {
      MKT.selCompany = null; MKT.selDate = null;
      if (coSel) coSel.innerHTML = '';
      if (dateSel) dateSel.innerHTML = '';
      board.innerHTML = '';
      meta.className = 'result visible error';
      meta.textContent = 'No marketing_status card filed yet — run handoff.py with the Tier 3 push enabled.';
      return;
    }
    // Preserve selection where possible, else default to first company / newest date.
    MKT.selCompany = (prevCompany && byCompany[prevCompany]) ? prevCompany : order[0];
    const co = byCompany[MKT.selCompany];
    MKT.selDate = (prevDate && co.snapshots.some(s => s.date === prevDate))
      ? prevDate
      : (co.snapshots[0] ? co.snapshots[0].date : null);
    mktFillCompanyDropdown();
    mktFillDateDropdown();
    mktRenderBoard();
    meta.className = 'result visible success';
    meta.textContent = order.length + (order.length === 1 ? ' company' : ' companies') + ' · section marketing_status';
  } catch(e) {
    meta.className = 'result visible error';
    meta.textContent = 'Error: ' + e.message;
  }
}

function mktFillCompanyDropdown() {
  const sel = document.getElementById('mktCompany');
  if (!sel) return;
  sel.innerHTML = MKT.order
    .map(slug => '<option value="' + mktEsc(slug) + '">' + mktEsc(MKT.byCompany[slug].display) + '</option>')
    .join('');
  if (MKT.selCompany != null) sel.value = MKT.selCompany;
}

function mktFillDateDropdown() {
  const sel = document.getElementById('mktDate');
  if (!sel) return;
  const co = MKT.byCompany[MKT.selCompany];
  const snaps = co ? co.snapshots : [];
  sel.innerHTML = snaps
    .map(s => '<option value="' + mktEsc(s.date) + '">' + mktEsc(mktHumanDate(s.date)) + '</option>')
    .join('');
  if (MKT.selDate != null) sel.value = MKT.selDate;
}

function mktOnCompanyChange() {
  const sel = document.getElementById('mktCompany');
  MKT.selCompany = sel.value;
  const co = MKT.byCompany[MKT.selCompany];
  MKT.selDate = (co && co.snapshots[0]) ? co.snapshots[0].date : null;   // default newest
  mktFillDateDropdown();
  mktRenderBoard();
}

function mktRenderBoard() {
  const dateSel = document.getElementById('mktDate');
  if (dateSel && dateSel.value) MKT.selDate = dateSel.value;
  const board = document.getElementById('marketingBoard');
  if (!board) return;
  const co = MKT.byCompany[MKT.selCompany];
  if (!co) { board.innerHTML = ''; return; }
  const snap = co.snapshots.find(s => s.date === MKT.selDate) || co.snapshots[0];
  if (!snap) { board.innerHTML = ''; return; }
  const manifest = snap.manifest;
  // Stale fires on the company's NEWEST snapshot age, independent of the browsed date.
  const newest = co.snapshots[0];
  let stale = '';
  if (newest && newest.ts != null && (Date.now() - newest.ts) / 86400000 > 7) {
    stale = '<span style="display:inline-block;margin-left:8px;padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#f87171;border:1px solid #f87171;">Stale</span>';
  }
  const header =
    '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">' +
      '<div class="card-title" style="margin-bottom:0;">' + mktEsc(co.display) + '</div>' +
      '<span style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;">As of ' + mktEsc(mktHumanDate(snap.date)) + '</span>' +
      stale +
    '</div>';
  const body = (manifest && Array.isArray(manifest.agents))
    ? manifest.agents.map(mktAgentCard).join('')
    : '<div style="font-size:12px;color:#e07e7e;">Card found, but it has no agent manifest to render.</div>';
  board.innerHTML = '<div style="margin-bottom:24px;">' + header + body + '</div>';
}

// ═══════════════ SALES TRACKER (Command Center) ═══════════════
// Reads Tier 3 section sales_status via /tier3/records. Two views off the same
// records: Tier 3 Books (per-company matrix) and Tier 4 Consolidated (aggregation).
const SALES_TOKEN = 'RT_DASH_7f39c2a4b8e15d60';
const SALES_METRICS = [
  {key:'revenue',   label:'Revenue',   money:true},
  {key:'mrr',       label:'MRR',       money:true},
  {key:'customers', label:'Customers'},
  {key:'new',       label:'New'},
  {key:'churned',   label:'Churned',   invert:true},
  {key:'units',     label:'Units'}
];
const SALES_COLORS = { progenyvault:'#c9a84c', reptiterra:'#e85d2f', denofindigos:'#5b8def' };
const SALES_MIX = [
  {slug:'progenyvault', code:'PV'},
  {slug:'reptiterra',   code:'RTL'},
  {slug:'denofindigos', code:'DOI'}
];
const SALES = { view:'books', metric:'revenue', expanded:{}, companies:[], periods:[], idx:{}, names:{}, status:{}, newestUpd:null, loaded:false };

function stEsc(s){ return (s==null?'':String(s)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function stColor(slug){ return SALES_COLORS[slug] || '#7a7e89'; }
// Mix-column code: known SALES_MIX code (PV/RTL/DOI), else first 3-4 letters of
// the company's name/slug, uppercased.
function stCode(slug){
  const mm = SALES_MIX.find(x=>x.slug===slug);
  if (mm) return mm.code;
  return String(SALES.names[slug] || slug || '').slice(0,4).toUpperCase();
}
function stFmt(metric, v){
  if (v==null || v==='') return '—';
  const n = Number(v);
  if (isNaN(n)) return '—';
  if (metric && metric.money) return '$' + Math.round(n).toLocaleString('en-US');
  return Math.round(n).toLocaleString('en-US');
}
function stVal(slug, period, key){
  const rec = SALES.idx[slug] && SALES.idx[slug][period];
  if (!rec) return null;
  const v = rec[key];
  return (v==null || v==='') ? null : Number(v);
}
function stTotal(period, key){
  let sum=0, any=false;
  for (const s of SALES.companies){ const v = stVal(s,period,key); if (v!=null){ sum+=v; any=true; } }
  return any ? sum : null;
}
function stMoM(cur, prev, invert){
  if (cur==null || prev==null || prev===0) return {text:'—', cls:'st-mom-flat'};
  const pct = (cur-prev)/Math.abs(prev)*100;
  let dir = pct>0?1:(pct<0?-1:0);
  if (invert) dir = -dir;
  const cls = dir>0?'st-mom-pos':(dir<0?'st-mom-neg':'st-mom-flat');
  return {pct, cls, text:(pct>0?'+':'')+pct.toFixed(1)+'%'};
}
function stHeat(v, rowMax){
  if (v==null || !rowMax) return '';
  const a = Math.max(0, Math.min(0.82, Math.abs(v)/rowMax*0.82));
  return 'background:rgba(201,168,76,'+a.toFixed(3)+');';
}
function stSpark(vals, color){
  const nums = vals.map(v => v==null?null:Number(v));
  const present = nums.filter(v => v!=null);
  const w=78, h=22, pad=2;
  if (present.length < 2) return `<svg width="${w}" height="${h}"></svg>`;
  const max = Math.max.apply(null, present), min = Math.min.apply(null, present);
  const span = (max-min) || 1;
  const n = nums.length;
  const pts = [];
  nums.forEach((v,i) => {
    if (v==null) return;
    const x = pad + (n===1?0:(i/(n-1))*(w-2*pad));
    const y = h-pad - ((v-min)/span)*(h-2*pad);
    pts.push(x.toFixed(1)+','+y.toFixed(1));
  });
  return `<svg width="${w}" height="${h}" style="display:block"><polyline fill="none" stroke="${color}" stroke-width="1.5" points="${pts.join(' ')}"/></svg>`;
}

function salesSetView(v){ SALES.view=v; salesRender(); }
function salesSetMetric(m){ SALES.metric=m; salesRender(); }
function salesToggleRow(slug){ SALES.expanded[slug] = !SALES.expanded[slug]; salesRender(); }

async function loadSalesTracker(){
  const msg = document.getElementById('stMsg');
  if (msg && !SALES.loaded){ msg.className='st-msg'; msg.textContent='Loading sales…'; }
  try {
    const res = await fetch('/tier3/records?section=sales_status&limit=500', { headers:{'X-API-Token':SALES_TOKEN} });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    const recs = data.records || [];
    const idx={}, names={}, status={}, periodSet={}, coOrder=[];
    let newestUpd=null;
    for (const r of recs){
      let p = r.record;
      if (!p && r.content){ try { p = JSON.parse(r.content); } catch(e){} }
      if (!p || !p.slug || !p.period) continue;
      const slug=p.slug, period=p.period;
      if (!idx[slug]){ idx[slug]={}; coOrder.push(slug); }
      idx[slug][period] = p;
      if (!names[slug]) names[slug] = p.company || slug;
      if (p.status) status[slug] = p.status;
      periodSet[period] = true;
      if (p.updated_at){ const t = Date.parse(p.updated_at); if (!isNaN(t) && (newestUpd==null || t>newestUpd)) newestUpd = t; }
    }
    const periods = Object.keys(periodSet).sort();
    const known = SALES_MIX.map(m => m.slug);
    const companies = known.filter(s => idx[s]).concat(coOrder.filter(s => known.indexOf(s)<0));
    SALES.idx=idx; SALES.names=names; SALES.status=status; SALES.periods=periods; SALES.companies=companies;
    SALES.newestUpd=newestUpd; SALES.loaded=true;
    salesRender();
  } catch(e){
    const body = document.getElementById('stBody');
    if (body) body.innerHTML = '<div class="st-msg err">Error: ' + stEsc(e.message) + '</div>';
  }
}

function salesRender(){
  const tog = document.getElementById('stToggle');
  if (tog){ const b = tog.querySelectorAll('button'); b[0].classList.toggle('active', SALES.view==='books'); b[1].classList.toggle('active', SALES.view==='consolidated'); }
  const asOf = document.getElementById('stAsOf');
  const cur = SALES.periods.length ? SALES.periods[SALES.periods.length-1] : null;
  if (asOf){
    let h = cur ? ('As of ' + stEsc(cur)) : 'No data';
    if (SALES.newestUpd!=null && (Date.now()-SALES.newestUpd)/86400000 > 7) h += ' <span class="st-stale">Stale</span>';
    asOf.innerHTML = h;
  }
  const body = document.getElementById('stBody');
  if (!body) return;
  if (!SALES.periods.length || !SALES.companies.length){
    body.innerHTML = '<div class="st-msg">No sales_status records yet. Push a record to populate the tracker.</div>';
    return;
  }
  body.innerHTML = salesKpis() + salesSwitcher() + (SALES.view==='books' ? salesBooks() : salesConsolidated());
}

function salesKpis(){
  const periods = SALES.periods;
  const cur = periods[periods.length-1];
  const prev = periods.length>1 ? periods[periods.length-2] : null;
  const money = {money:true}, plain = {};
  const revCur = stTotal(cur,'revenue'), revPrev = prev?stTotal(prev,'revenue'):null;
  const rMoM = stMoM(revCur, revPrev, false);
  const mrrCur = stTotal(cur,'mrr'), mrrPrev = prev?stTotal(prev,'mrr'):null;
  const mrrMoM = stMoM(mrrCur, mrrPrev, false);
  const custCur = stTotal(cur,'customers');
  let netNew=0, hasNN=false;
  for (const s of SALES.companies){ const n=stVal(s,cur,'new'), c=stVal(s,cur,'churned'); if(n!=null){netNew+=n;hasNN=true;} if(c!=null){netNew-=c;hasNN=true;} }
  const nn = hasNN ? ((netNew>=0?'+':'')+netNew.toLocaleString('en-US')+' net new') : '—';
  let mover=null;
  if (prev){
    for (const s of SALES.companies){
      const c=stVal(s,cur,'revenue'), p=stVal(s,prev,'revenue');
      if (c==null||p==null||p===0) continue;
      const pct=(c-p)/Math.abs(p)*100;
      if (!mover || pct>mover.pct) mover={slug:s, pct};
    }
  }
  let moverTxt='—';
  if (mover){ const mm=SALES_MIX.find(x=>x.slug===mover.slug); const code=mm?mm.code:(SALES.names[mover.slug]||mover.slug); moverTxt = stEsc(code)+' '+(mover.pct>0?'+':'')+mover.pct.toFixed(1)+'%'; }
  const kpi = (lab,val,sub,cls) => `<div class="st-kpi"><div class="st-kpi-lab">${lab}</div><div class="st-kpi-val">${val}</div><div class="st-kpi-sub ${cls||''}">${sub}</div></div>`;
  return `<div class="st-kpis">
    ${kpi('Portfolio Revenue', stFmt(money,revCur), rMoM.text+' MoM', rMoM.cls)}
    ${kpi('Portfolio MRR', stFmt(money,mrrCur), mrrMoM.text+' MoM', mrrMoM.cls)}
    ${kpi('Portfolio Customers', stFmt(plain,custCur), nn)}
    ${kpi('Top Mover', moverTxt, 'best revenue MoM')}
  </div>`;
}

function salesSwitcher(){
  return '<div class="st-switch">' + SALES_METRICS.map(m =>
    `<button class="${m.key===SALES.metric?'active':''}" onclick="salesSetMetric('${m.key}')">${m.label}</button>`
  ).join('') + '</div>';
}

function salesBooks(){
  const m = SALES_METRICS.find(x=>x.key===SALES.metric);
  const periods = SALES.periods;
  const cur = periods[periods.length-1];
  const prev = periods.length>1?periods[periods.length-2]:null;
  let head = '<tr><th class="st-co">Company</th>';
  for (const per of periods) head += `<th class="${per===cur?'st-cur':''}">${stEsc(per)}</th>`;
  head += '<th>MoM</th><th>Trend</th></tr>';
  let rows='';
  for (const slug of SALES.companies){
    const vals = periods.map(per => stVal(slug,per,m.key));
    const rowMax = Math.max.apply(null, [0].concat(vals.filter(v=>v!=null).map(v=>Math.abs(v))));
    let cells='';
    periods.forEach((per,i) => { cells += `<td class="${per===cur?'st-cur':''}" style="${stHeat(vals[i],rowMax)}">${stFmt(m,vals[i])}</td>`; });
    const mom = stMoM(stVal(slug,cur,m.key), prev?stVal(slug,prev,m.key):null, m.invert);
    const color = stColor(slug);
    rows += `<tr class="st-clickable" onclick="salesToggleRow('${slug}')">
      <td class="st-co"><span class="st-dot" style="background:${color}"></span>${stEsc(SALES.names[slug]||slug)}</td>
      ${cells}<td class="${mom.cls}">${mom.text}</td><td>${stSpark(vals,color)}</td></tr>`;
    if (SALES.expanded[slug]) rows += `<tr class="st-book"><td colspan="${periods.length+3}">${salesBook(slug)}</td></tr>`;
  }
  const totVals = periods.map(per => stTotal(per,m.key));
  let totCells='';
  periods.forEach((per,i) => { totCells += `<td class="${per===cur?'st-cur':''}">${stFmt(m,totVals[i])}</td>`; });
  const totMom = stMoM(stTotal(cur,m.key), prev?stTotal(prev,m.key):null, m.invert);
  rows += `<tr class="st-total"><td class="st-co">Portfolio Total</td>${totCells}<td class="${totMom.cls}">${totMom.text}</td><td>${stSpark(totVals,'#c9a84c')}</td></tr>`;
  return `<div class="st-scroll"><table class="st-matrix"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
}

function salesBook(slug){
  const periods = SALES.periods;
  const cur = periods[periods.length-1];
  const prev = periods.length>1?periods[periods.length-2]:null;
  let head = '<tr><th class="st-co">Metric</th>';
  for (const per of periods) head += `<th class="${per===cur?'st-cur':''}">${stEsc(per)}</th>`;
  head += '<th>MoM</th></tr>';
  let rows='';
  for (const met of SALES_METRICS){
    let cells='';
    periods.forEach(per => { cells += `<td class="${per===cur?'st-cur':''}">${stFmt(met, stVal(slug,per,met.key))}</td>`; });
    const mom = stMoM(stVal(slug,cur,met.key), prev?stVal(slug,prev,met.key):null, met.invert);
    rows += `<tr><td class="st-co">${met.label}</td>${cells}<td class="${mom.cls}">${mom.text}</td></tr>`;
  }
  return `<div class="st-scroll"><table class="st-sub"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
}

function salesConsolidated(){
  const m = SALES_METRICS.find(x=>x.key===SALES.metric);
  const periods = SALES.periods;
  const cur = periods[periods.length-1];
  const totals = periods.map(per => { let sum=0; for (const s of SALES.companies){ const v=stVal(s,per,m.key); if(v!=null&&v>0) sum+=v; } return sum; });
  const maxTot = Math.max.apply(null, [1].concat(totals));
  const W = Math.max(320, periods.length*64), H=180, padB=26, padT=10, padL=8;
  const step = (W-padL*2)/periods.length, bw = step*0.55;
  let bars='';
  periods.forEach((per,i) => {
    const cx = padL + (i+0.5)*step;
    let y = H-padB;
    for (const s of SALES.companies){
      const v = stVal(s,per,m.key);
      if (v==null || v<=0) continue;
      const hh = (v/maxTot)*(H-padB-padT);
      y -= hh;
      bars += `<rect x="${(cx-bw/2).toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}" fill="${stColor(s)}"></rect>`;
    }
    bars += `<text x="${cx.toFixed(1)}" y="${H-padB+14}" fill="#7a7e89" font-size="9" text-anchor="middle">${stEsc(per.slice(5))}</text>`;
  });
  const svg = `<svg width="100%" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="display:block;max-width:100%">${bars}</svg>`;
  const legend = SALES.companies.map(s => `<span><span class="st-dot" style="background:${stColor(s)}"></span>${stEsc(SALES.names[s]||s)}</span>`).join('');
  let rows='';
  periods.slice().reverse().forEach(per => {
    const tot = stTotal(per,m.key);
    const pi = periods.indexOf(per);
    const pv = pi>0 ? stTotal(periods[pi-1], m.key) : null;
    const mom = stMoM(tot, pv, m.invert);
    const mixCells = SALES.companies.map(s => {
      const v = stVal(s, per, m.key);
      if (v==null || tot==null || tot===0) return '<td>—</td>';
      return `<td>${(v/tot*100).toFixed(1)}%</td>`;
    }).join('');
    rows += `<tr class="${per===cur?'st-total':''}"><td class="st-co">${stEsc(per)}</td><td>${stFmt(m,tot)}</td><td class="${mom.cls}">${mom.text}</td>${mixCells}</tr>`;
  });
  const mixHead = SALES.companies.map(s => `<th>${stEsc(stCode(s))} mix</th>`).join('');
  const thead = `<tr><th class="st-co">Month</th><th>Total</th><th>MoM</th>${mixHead}</tr>`;
  return `<div class="st-cons-chart">${svg}<div class="st-legend">${legend}</div></div>
    <div class="st-scroll"><table class="st-matrix"><thead>${thead}</thead><tbody>${rows}</tbody></table></div>`;
}

// NICHE SCORER
async function scoreNiche() {
  const niche = document.getElementById('nicheInput').value.trim();
  if (!niche) return;
  const result = document.getElementById('nicheResult');
  result.textContent = '⬤ Scoring...';
  result.className = 'result visible';
  try {
    const data = await callServer('/api/ai', {
      prompt: `Score this niche for the Progyny Infinite Factory using 5 criteria (0-10 each, max 50):
1. Community exists
2. Ops problem exists (spreadsheets/paper/nothing)
3. No dominant tool
4. Willingness to pay
5. Too small for big players

GO threshold: 35+/50.

Niche: ${niche}

Give score per criterion, total, GO/MAYBE/NO-GO verdict, top opportunity, biggest risk. Be direct.`
    });
    result.className = 'result visible success';
    result.textContent = data.result || data.error;
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

// YOUTUBE EXTRACTOR
async function extractYoutube() {
  const url = document.getElementById('ytUrl').value.trim();
  const mode = document.getElementById('ytMode').value || 'wiki';
  if (!url) return;
  const result = document.getElementById('ytResult');
  result.textContent = '⬤ Extracting...';
  result.className = 'result visible';
  const prompts = {
    wiki: `Extract key concepts from this YouTube video for a business wiki. URL: ${url}. Give: core concepts, actionable techniques, tools mentioned, applicability to micro-SaaS factory build.`,
    build: `Extract everything buildable from this video. URL: ${url}. Give: file structure, code patterns, build order, what to skip.`,
    summary: `Summarize in under 200 words. URL: ${url}. Give: what it is, 3 key points, verdict: worth building from?`,
    picp: `What applies to a Claude Project brain + agent + dashboard system? URL: ${url}. Give: structure ideas, file patterns, automation ideas, prompts to steal.`
  };
  try {
    const data = await callServer('/api/ai', {prompt: prompts[mode], web_search: true});
    result.className = 'result visible success';
    result.textContent = data.result || data.error;
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

// FINANCE AI
async function queryFinance() {
  const q = document.getElementById('financeInput').value.trim();
  if (!q) return;
  const result = document.getElementById('financeResult');
  result.textContent = '⬤ Searching...';
  result.className = 'result visible';
  try {
    const data = await callServer('/api/ai', {prompt: q, web_search: true});
    result.className = 'result visible success';
    result.textContent = data.result || data.error;
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

// PASTE CHAT -> TIER 3
async function ingestTier3() {
  const text = document.getElementById('tier3Paste').value.trim();
  const status = document.getElementById('tier3Status');
  if (!text) { status.className = 'result visible'; status.textContent = 'Paste a conversation first.'; return; }
  status.className = 'result visible';
  status.textContent = '⬤ Cleaning & filing to Tier 3...';
  try {
    const res = await fetch('/tier3/ingest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Token': RT.token},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (data.ok) {
      status.className = 'result visible success';
      status.textContent = '✓ Filed to Tier 3: ' + data.title;
    } else {
      status.className = 'result visible error';
      status.textContent = '✗ ' + (data.error || 'Failed');
    }
  } catch(e) {
    status.className = 'result visible error';
    status.textContent = '✗ ' + e.message;
  }
}

// ═══════════════ ReptiTerra Feeding Dashboard ═══════════════
const RT_DEFAULT_TOKEN = 'RT_DASH_7f39c2a4b8e15d60';
if (!localStorage.getItem('rt_token')) localStorage.setItem('rt_token', RT_DEFAULT_TOKEN);
const RT = { token: localStorage.getItem('rt_token'), sub: 'animals', animals: [], feed: [], bowls: [], hides: [], supplies: [] };
const RT_MOUSE = ['XS Pinky','SM Pinky','LG Pinky','Peach Fuzzy','Fuzzy','Small','Medium','Large','XL'];
const RT_RAT = ['Pinky','Fuzzy','Pup','Weaned','Small','Medium','Large','XL','XXL','XXXL'];
const RT_BOWL = ['XS','Small','Medium','Large','XL','XXL'];
const RT_UNITS = ['each','box','ft','gallon','bag','roll'];
const RT_CATALOG = {
  Heating: ['Ceramic Heat Emitter','Deep Heat Projector','Halogen Flood Bulb','Basking Bulb','Heat Mat','Radiant Heat Panel','Thermostat','Heat Lamp Guard'],
  Substrate: ['Burrow Blend','Tropical Blend','Tunnel Blend'],
  Lighting: ['UVB Bulb / Tube','LED Grow Bar','LED Light Strip','Light Fixture / Hood'],
  Other: ['Thermometer / Hygrometer','Feeding Tongs','Enclosure Lock / Clip','Control Center','Timers','Commercial Power Strips','ReptiTemp Controller']
};
const RT_LOW = 3;

async function rtApi(method, path, body) {
  const opt = { method, headers: { 'Content-Type': 'application/json', 'X-API-Token': RT.token } };
  if (body) opt.body = JSON.stringify(body);
  const res = await fetch('/api/feeding' + path, opt);
  if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
  const t = await res.text();
  return t ? JSON.parse(t) : null;
}
function rtEsc(s) { return (s == null ? '' : String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function rtToday() { return new Date().toISOString().slice(0, 10); }
function rtv(id) { const el = document.getElementById(id); return el ? el.value.trim() : ''; }

function rtSwitch(sub) {
  RT.sub = sub;
  ['animals','feed','supplies'].forEach(s => {
    document.getElementById('rtsub-' + s).classList.toggle('active', s === sub);
    document.getElementById('rtpane-' + s).classList.toggle('active', s === sub);
  });
  if (sub === 'animals') rtLoadAnimals();
  if (sub === 'feed') rtLoadFeed();
  if (sub === 'supplies') rtLoadSupplies();
}
function rtModal(html) { document.getElementById('rtModal').innerHTML = html; document.getElementById('rtModalBg').classList.add('open'); }
function rtCloseModal() { document.getElementById('rtModalBg').classList.remove('open'); }
document.addEventListener('click', function (e) { if (e.target && e.target.id === 'rtModalBg') rtCloseModal(); });

// ---------- ANIMALS ----------
async function rtLoadAnimals() {
  const box = document.getElementById('rtAnimals');
  box.innerHTML = '<div class="rt-muted">Loading…</div>';
  try {
    RT.animals = (await rtApi('GET', '/animals')) || [];
    if (!RT.animals.length) { box.innerHTML = '<div class="rt-card rt-muted">No animals yet. Tap “+ Animal” to start.</div>'; return; }
    box.innerHTML = '';
    for (const a of RT.animals) {
      const logs = (await rtApi('GET', '/logs/' + a.id)) || [];
      box.insertAdjacentHTML('beforeend', rtAnimalCard(a, logs));
    }
  } catch (e) { box.innerHTML = `<div class="rt-card" style="color:var(--rt-danger)">${rtEsc(e.message)}</div>`; }
}
function rtAnimalCard(a, logs) {
  const fed = logs.filter(l => !l.refused);
  const every = a.feed_every_days || 7;
  let pct = 1, col = 'var(--rt-danger)', label = 'No feeding logged';
  if (fed.length) {
    const last = new Date(fed[0].date || fed[0].ts);
    const days = (Date.now() - last.getTime()) / 86400000;
    pct = days / every;
    if (pct < 0.6) col = 'var(--rt-ok)'; else if (pct <= 0.85) col = 'var(--rt-warn)'; else col = 'var(--rt-danger)';
    label = pct > 1 ? ('Overdue ' + Math.ceil(days - every) + 'd') : ('Due in ' + Math.max(0, Math.ceil(every - days)) + 'd');
  }
  const chips = logs.slice(0, 3).map(l => `<span class="rt-chip" title="${rtEsc(l.date || '')}" style="background:${l.refused ? 'var(--rt-danger)' : 'var(--rt-ok)'}"></span>`).join('') || '<span class="rt-muted">—</span>';
  return `<div class="rt-card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div><span style="font-size:22px">${rtEsc(a.emoji || '🦎')}</span> <span class="rt-title">${rtEsc(a.name)}</span>
        <div class="rt-muted">${rtEsc(a.species || '')} · every ${every}d</div></div>
      <button class="rt-btn ghost sm" style="font-size:13px" onclick="rtDeleteAnimal('${a.id}')">✕</button>
    </div>
    <div class="rt-statusbar"><div style="width:${Math.min(pct * 100, 100)}%;height:100%;background:${col}"></div></div>
    <div style="display:flex;justify-content:space-between;align-items:center">
      <div style="font-size:12px;color:${col}">${label}</div><div>${chips}</div>
    </div>
    <button class="rt-btn" style="width:100%;margin-top:10px" onclick="rtOpenLog('${a.id}')">Log Feeding</button>
  </div>`;
}
function rtOpenAddAnimal() {
  rtModal(`<div class="rt-h">Add Animal</div>
    <label class="rt-lab">Name</label><input class="rt-input" id="rtaName">
    <label class="rt-lab">Species</label><input class="rt-input" id="rtaSpecies">
    <label class="rt-lab">Emoji</label><input class="rt-input" id="rtaEmoji" value="🦎">
    <label class="rt-lab">Feed every (days)</label><input class="rt-input" id="rtaEvery" type="number" value="7">
    <label class="rt-lab">Default prey type</label><input class="rt-input" id="rtaPrey" placeholder="e.g. mouse">
    <label class="rt-lab">Default prey size</label><input class="rt-input" id="rtaSize" placeholder="e.g. Adult">
    <div style="display:flex;gap:8px;margin-top:6px"><button class="rt-btn ghost" style="flex:1" onclick="rtCloseModal()">Cancel</button><button class="rt-btn" style="flex:1" onclick="rtSaveAnimal()">Save</button></div>`);
}
async function rtSaveAnimal() {
  const b = { name: rtv('rtaName'), species: rtv('rtaSpecies'), emoji: rtv('rtaEmoji') || '🦎',
    feedEveryDays: parseInt(rtv('rtaEvery')) || 7, preyType: rtv('rtaPrey'), preySize: rtv('rtaSize') };
  if (!b.name) { alert('Name required'); return; }
  try { await rtApi('POST', '/animals', b); rtCloseModal(); rtLoadAnimals(); } catch (e) { alert(e.message); }
}
async function rtDeleteAnimal(id) {
  if (!confirm('Delete this animal and its feeding history?')) return;
  try { await rtApi('DELETE', '/animals/' + id); rtLoadAnimals(); } catch (e) { alert(e.message); }
}
async function rtOpenLog(id) {
  const a = RT.animals.find(x => x.id === id); if (!a) return;
  try { RT.feed = (await rtApi('GET', '/inventory')) || []; } catch (e) {}
  const stock = RT.feed.filter(f => f.count > 0).map(f => `<option value="${f.category}|${rtEsc(f.size)}">${f.category === 'mice' ? 'Mouse' : 'Rat'} — ${rtEsc(f.size)} (${f.count})</option>`).join('');
  rtModal(`<div class="rt-h">Log Feeding — ${rtEsc(a.name)}</div>
    <label class="rt-lab">Date</label><input class="rt-input" id="rtlDate" type="date" value="${rtToday()}">
    <label class="rt-lab">Prey type</label><input class="rt-input" id="rtlPrey" value="${rtEsc(a.prey_type || '')}">
    <label class="rt-lab">Feed size — deducts 1 from stock</label>
    <select class="rt-input" id="rtlSize"><option value="">— none / don't deduct —</option>${stock}</select>
    <label class="rt-lab">Notes</label><textarea class="rt-input" id="rtlNotes" rows="2"></textarea>
    <label class="rt-lab" style="display:flex;align-items:center;gap:8px"><input type="checkbox" id="rtlRefused" style="width:auto"> Refused (no deduction)</label>
    <div style="display:flex;gap:8px;margin-top:10px"><button class="rt-btn ghost" style="flex:1" onclick="rtCloseModal()">Cancel</button><button class="rt-btn" style="flex:1" onclick="rtSaveLog('${id}')">Save</button></div>`);
}
async function rtSaveLog(id) {
  const refused = document.getElementById('rtlRefused').checked;
  const sv = rtv('rtlSize'); let deduct = null, sizeLabel = '';
  if (sv) { const p = sv.split('|'); sizeLabel = p[1]; if (!refused) deduct = { category: p[0], size: p[1] }; }
  const b = { date: rtv('rtlDate'), prey: rtv('rtlPrey'), size: sizeLabel, notes: rtv('rtlNotes'), refused, deduct };
  try { await rtApi('POST', '/logs/' + id, b); rtCloseModal(); rtLoadAnimals(); } catch (e) { alert(e.message); }
}

// ---------- FEED ----------
async function rtLoadFeed() {
  const box = document.getElementById('rtFeed');
  box.innerHTML = '<div class="rt-muted">Loading…</div>';
  try {
    RT.feed = (await rtApi('GET', '/inventory')) || [];
    rtFeedBadge();
    box.innerHTML = rtFeedSection('Mice', RT.feed.filter(f => f.category === 'mice'), RT_MOUSE)
      + rtFeedSection('Rats', RT.feed.filter(f => f.category === 'rats'), RT_RAT);
  } catch (e) { box.innerHTML = `<div class="rt-card" style="color:var(--rt-danger)">${rtEsc(e.message)}</div>`; }
}
function rtFeedSection(title, rows, order) {
  rows = rows.slice().sort((a, b) => order.indexOf(a.size) - order.indexOf(b.size));
  let inner = rows.length ? '' : '<div class="rt-muted">No deliveries logged yet.</div>';
  for (const r of rows) {
    const low = r.count > 0 && r.count <= RT_LOW;
    inner += `<div class="rt-row"><div>${rtEsc(r.size)}${low ? ' <span class="rt-badge low">LOW</span>' : ''}</div>
      <div style="display:flex;align-items:center;gap:8px">
        <button class="rt-btn ghost sm" onclick="rtFeedAdjust('${r.category}','${rtEsc(r.size)}',-1)">−</button>
        <span class="rt-count">${r.count}</span>
        <button class="rt-btn ghost sm" onclick="rtFeedAdjust('${r.category}','${rtEsc(r.size)}',1)">+</button>
      </div></div>`;
  }
  return `<div class="rt-card"><div class="rt-h">${title}</div>${inner}</div>`;
}
function rtFeedBadge() {
  const low = RT.feed.some(f => f.count > 0 && f.count <= RT_LOW);
  document.getElementById('rtsub-feed').innerHTML = 'Feed' + (low ? ' <span class="rt-badge low">!</span>' : '');
}
async function rtFeedAdjust(category, size, delta) {
  try { await rtApi('POST', '/inventory/adjust', { category, size, delta }); rtLoadFeed(); } catch (e) { alert(e.message); }
}
function rtDeliveryInputs(arr) {
  return arr.map(s => `<div class="rt-row"><div>${rtEsc(s)}</div><input class="rt-input" style="width:84px;margin:0" type="number" min="0" value="0" data-size="${rtEsc(s)}"></div>`).join('');
}
function rtOpenDelivery() {
  rtModal(`<div class="rt-h">Log Delivery</div>
    <label class="rt-lab">Category</label>
    <select class="rt-input" id="rtdCat" onchange="rtDeliveryCat(this.value)"><option value="mice">Mice</option><option value="rats">Rats</option></select>
    <div id="rtdSizes">${rtDeliveryInputs(RT_MOUSE)}</div>
    <div style="display:flex;gap:8px;margin-top:10px"><button class="rt-btn ghost" style="flex:1" onclick="rtCloseModal()">Cancel</button><button class="rt-btn" style="flex:1" onclick="rtSaveDelivery()">Add to Stock</button></div>`);
}
function rtDeliveryCat(cat) { document.getElementById('rtdSizes').innerHTML = rtDeliveryInputs(cat === 'mice' ? RT_MOUSE : RT_RAT); }
async function rtSaveDelivery() {
  const cat = rtv('rtdCat'); const items = {};
  document.querySelectorAll('#rtdSizes input').forEach(i => { const q = parseInt(i.value) || 0; if (q > 0) items[i.dataset.size] = q; });
  if (!Object.keys(items).length) { alert('Enter at least one quantity'); return; }
  try { await rtApi('POST', '/inventory/delivery', { category: cat, items }); rtCloseModal(); rtLoadFeed(); } catch (e) { alert(e.message); }
}

// ---------- SUPPLIES ----------
async function rtLoadSupplies() {
  const box = document.getElementById('rtSupplies');
  box.innerHTML = '<div class="rt-muted">Loading…</div>';
  try {
    const r = await Promise.all([rtApi('GET', '/bowls'), rtApi('GET', '/hides'), rtApi('GET', '/supplies')]);
    RT.bowls = r[0] || []; RT.hides = r[1] || []; RT.supplies = r[2] || [];
    box.innerHTML = rtFixed('Water Bowls', 'bowls', RT.bowls) + rtFixed('Hide Boxes', 'hides', RT.hides) + rtSuppliesHtml();
  } catch (e) { box.innerHTML = `<div class="rt-card" style="color:var(--rt-danger)">${rtEsc(e.message)}</div>`; }
}
function rtFixed(title, kind, rows) {
  const by = {}; rows.forEach(r => by[r.size] = r.count);
  let inner = '';
  for (const s of RT_BOWL) {
    const c = by[s] || 0;
    inner += `<div class="rt-row"><div>${s}</div><div style="display:flex;align-items:center;gap:8px">
      <button class="rt-btn ghost sm" onclick="rtFixedAdjust('${kind}','${s}',-1)">−</button>
      <span class="rt-count">${c}</span>
      <button class="rt-btn ghost sm" onclick="rtFixedAdjust('${kind}','${s}',1)">+</button></div></div>`;
  }
  return `<div class="rt-card"><div class="rt-h">${title}</div>${inner}</div>`;
}
async function rtFixedAdjust(kind, size, delta) {
  try { await rtApi('POST', '/' + kind + '/adjust', { size, delta }); rtLoadSupplies(); } catch (e) { alert(e.message); }
}
function rtSuppliesHtml() {
  let html = '';
  for (const cat of ['Heating','Substrate','Lighting','Other']) {
    const items = RT.supplies.filter(s => s.category === cat);
    let inner = items.length ? '' : '<div class="rt-muted">None</div>';
    for (const s of items) {
      const needs = s.reorder_enabled && s.reorder_point != null && (s.qty || 0) <= s.reorder_point;
      inner += `<div class="rt-row">
        <div>${rtEsc(s.name)}${s.spec ? ` <span class="rt-muted">(${rtEsc(s.spec)})</span>` : ''}${needs ? ' <span class="rt-badge danger">REORDER</span>' : ''}
          <div class="rt-muted">${s.qty || 0} ${rtEsc(s.unit || 'each')}</div></div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
          <button class="rt-btn ghost sm" style="font-size:11px" onclick="rtAddStock('${s.id}')">+ Stock</button>
          <button class="rt-btn ghost sm" style="font-size:11px" onclick="rtReplaced('${s.id}')">Replaced</button>
          <button class="rt-btn ghost sm" style="font-size:11px" onclick="rtHistory('${s.id}')">History</button>
          <button class="rt-btn ghost sm" style="font-size:11px" onclick="rtDeleteSupply('${s.id}')">✕</button>
        </div></div>`;
    }
    html += `<div class="rt-card"><div class="rt-h">${cat}</div>${inner}</div>`;
  }
  return html;
}
function rtOpenAddSupply() {
  const cats = Object.keys(RT_CATALOG).map(c => `<option>${c}</option>`).join('');
  const items = RT_CATALOG.Heating.map(i => `<option>${rtEsc(i)}</option>`).join('');
  const units = RT_UNITS.map(u => `<option>${u}</option>`).join('');
  rtModal(`<div class="rt-h">Add Supply</div>
    <label class="rt-lab">Category</label><select class="rt-input" id="rtsCat" onchange="rtSupplyCat(this.value)">${cats}</select>
    <label class="rt-lab">Item</label><select class="rt-input" id="rtsItem">${items}</select>
    <label class="rt-lab">Size / spec (optional)</label><input class="rt-input" id="rtsSpec">
    <label class="rt-lab">Quantity</label><input class="rt-input" id="rtsQty" type="number" value="1">
    <label class="rt-lab">Unit</label><select class="rt-input" id="rtsUnit">${units}</select>
    <label class="rt-lab">Reorder point</label><input class="rt-input" id="rtsReorder" type="number" value="1">
    <label class="rt-lab" style="display:flex;align-items:center;gap:8px"><input type="checkbox" id="rtsAlert" style="width:auto" checked> Reorder alert on</label>
    <div style="display:flex;gap:8px;margin-top:10px"><button class="rt-btn ghost" style="flex:1" onclick="rtCloseModal()">Cancel</button><button class="rt-btn" style="flex:1" onclick="rtSaveSupply()">Save</button></div>`);
}
function rtSupplyCat(cat) { document.getElementById('rtsItem').innerHTML = RT_CATALOG[cat].map(i => `<option>${rtEsc(i)}</option>`).join(''); }
async function rtSaveSupply() {
  const b = { category: rtv('rtsCat'), name: rtv('rtsItem'), spec: rtv('rtsSpec'), qty: parseInt(rtv('rtsQty')) || 0,
    unit: rtv('rtsUnit'), reorderPoint: parseInt(rtv('rtsReorder')) || 0, reorderEnabled: document.getElementById('rtsAlert').checked };
  try { await rtApi('POST', '/supplies', b); rtCloseModal(); rtLoadSupplies(); } catch (e) { alert(e.message); }
}
async function rtDeleteSupply(id) { if (!confirm('Delete this supply?')) return; try { await rtApi('DELETE', '/supplies/' + id); rtLoadSupplies(); } catch (e) { alert(e.message); } }
async function rtAddStock(id) {
  const n = prompt('Add how many?', '1'); if (n == null) return; const qty = parseInt(n) || 0; if (qty <= 0) return;
  try { await rtApi('POST', '/supplies/' + id + '/logs', { type: 'addition', qty, date: rtToday() }); rtLoadSupplies(); } catch (e) { alert(e.message); }
}
async function rtReplaced(id) {
  const reason = prompt('Replaced — reason (wear / broke / used up)?', ''); if (reason == null) return;
  try { await rtApi('POST', '/supplies/' + id + '/logs', { type: 'replacement', qty: 1, reason, date: rtToday() }); rtLoadSupplies(); } catch (e) { alert(e.message); }
}
async function rtHistory(id) {
  try {
    const logs = (await rtApi('GET', '/supplies/' + id + '/logs')) || [];
    const s = RT.supplies.find(x => x.id === id);
    const rows = logs.length ? logs.map(l => `<div class="rt-row"><div>${rtEsc(l.date || '')} · ${rtEsc(l.type)}${l.reason ? ' · ' + rtEsc(l.reason) : ''}</div><div>${l.type === 'addition' ? '+' : '−'}${l.qty}</div></div>`).join('') : '<div class="rt-muted">No history yet</div>';
    rtModal(`<div class="rt-h">History — ${rtEsc(s ? s.name : '')}</div>${rows}<div style="margin-top:10px"><button class="rt-btn ghost" style="width:100%" onclick="rtCloseModal()">Close</button></div>`);
  } catch (e) { alert(e.message); }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    return {"status": "Progyny Infinite Dashboard online", "operator": "Randy Wain Nutt"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
