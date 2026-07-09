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
        "select": "id,section,source_type,source_id,title,ingested_at",
        "section": f"eq.{section}",
        "order": "ingested_at.desc",
        "limit": str(limit),
    }
    if q:
        params["content"] = f"fts(english).{q}"
    res = await _sb_service("GET", "tier3_databank", params=params)
    return {"section": section, "count": len(res or []), "records": res}


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
  <button class="nav-tab" onclick="switchTab('tracker')">Project Tracker</button>
  <button class="nav-tab" onclick="switchTab('agent')">Agent Control</button>
  <button class="nav-tab" onclick="switchTab('niche')">Niche Scorer</button>
  <button class="nav-tab" onclick="switchTab('youtube')">YouTube Extractor</button>
  <button class="nav-tab" onclick="switchTab('finance')">Finance AI</button>
  <button class="nav-tab" onclick="switchTab('tools')">Tools</button>
  <button class="nav-tab" id="rtsub-feed-nav" onclick="switchTab('feeding')">🦎 ReptiTerra</button>
  <button class="nav-tab" onclick="switchTab('tier3')">Tier 3 Paste</button>
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

<!-- PROJECT TRACKER -->
<div class="panel" id="panel-tracker">
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
</div>

<!-- AGENT CONTROL -->
<div class="panel" id="panel-agent">
  <div class="two-col">
    <div>
      <div class="card">
        <div class="card-title">Send Context to Brain</div>
        <div class="chip-row">
          <button class="chip" onclick="document.getElementById('sourceInput').value='session'">Session Extract</button>
          <button class="chip" onclick="document.getElementById('sourceInput').value='decision'">Decision Log</button>
          <button class="chip" onclick="document.getElementById('sourceInput').value='project'">Project Update</button>
        </div>
        <textarea id="contextInput" placeholder="Paste your EXTRACT CONTEXT document here..."></textarea>
        <input type="text" id="sourceInput" value="session" placeholder="Source label" style="margin-top:10px;" />
        <button class="btn" onclick="ingestContext()">Send to Brain</button>
        <div class="result" id="ingestResult"></div>
      </div>
    </div>
    <div>
      <div class="card">
        <div class="card-title">Query Brain</div>
        <div class="chip-row">
          <button class="chip" onclick="document.getElementById('queryInput').value='What is the current status of ProgenyVault?'">PV Status</button>
          <button class="chip" onclick="document.getElementById('queryInput').value='What are all open flags right now?'">Open Flags</button>
          <button class="chip" onclick="document.getElementById('queryInput').value='What is the next niche to build?'">Next Niche</button>
        </div>
        <input type="text" id="queryInput" placeholder="Ask the brain anything..." />
        <button class="btn" onclick="queryBrain()">Ask Brain</button>
        <div class="result" id="queryResult"></div>
      </div>
      <div class="card">
        <div class="card-title">Agent Log</div>
        <div class="log-box" id="logBox">Click refresh to load logs</div>
        <button class="btn btn-ghost" onclick="loadLog()" style="margin-top:10px;">Refresh Log</button>
      </div>
    </div>
  </div>
</div>

<!-- NICHE SCORER -->
<div class="panel" id="panel-niche">
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
    <div class="tool-card" onclick="switchTab('niche')"><div class="tool-icon">🎯</div><div class="tool-name">Niche Scorer</div><div class="tool-desc">Score any niche against the 5-criteria Factory formula</div></div>
    <div class="tool-card" onclick="switchTab('youtube')"><div class="tool-icon">▶️</div><div class="tool-name">YouTube Extractor</div><div class="tool-desc">Extract concepts from any YouTube video automatically</div></div>
    <div class="tool-card" onclick="switchTab('finance')"><div class="tool-icon">📈</div><div class="tool-name">Finance AI</div><div class="tool-desc">Market queries, stock lookup, Vault Trader status</div></div>
    <div class="tool-card" onclick="switchTab('agent')"><div class="tool-icon">🧠</div><div class="tool-name">Brain Control</div><div class="tool-desc">Send context to the PICP agent, query the brain</div></div>
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

function switchTab(tab) {
  const tabs = ['command','tracker','agent','niche','youtube','finance','tools','feeding','tier3'];
  document.querySelectorAll('.nav-tab').forEach((t,i) => t.classList.toggle('active', tabs[i] === tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  if (tab === 'feeding') rtSwitch(RT.sub);
}

// AGENT CONTROL
async function ingestContext() {
  const content = document.getElementById('contextInput').value.trim();
  if (!content) return;
  const source = document.getElementById('sourceInput').value || 'session';
  const result = document.getElementById('ingestResult');
  result.textContent = '⬤ Sending to brain...';
  result.className = 'result visible';
  try {
    const data = await callServer('/api/ingest', {content, source});
    result.className = 'result visible success';
    result.textContent = '✓ Ingested\\n\\n' + data.summary + '\\nDecisions: ' + data.decisions_captured + ' | Flags: ' + data.flags_captured;
    document.getElementById('contextInput').value = '';
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

async function queryBrain() {
  const question = document.getElementById('queryInput').value.trim();
  if (!question) return;
  const result = document.getElementById('queryResult');
  result.textContent = '⬤ Thinking...';
  result.className = 'result visible';
  try {
    const data = await callServer('/api/query', {question});
    result.className = 'result visible success';
    result.textContent = data.answer;
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  }
}

async function loadLog() {
  try {
    const data = await fetch('/api/log').then(r => r.json());
    document.getElementById('logBox').textContent = (data.log || []).join('\\n');
  } catch(e) {
    document.getElementById('logBox').textContent = 'Could not load log';
  }
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
