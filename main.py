"""
Progyny Infinite Dashboard
Hosted on Railway. Open from any device, any browser.
All AI calls route through server — no browser CORS issues.
"""

from fastapi import FastAPI, Request, Header, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import io
import json
import re
import base64
import hashlib
import uuid
from datetime import datetime, timezone
import asyncio
import threading
import anthropic
import httpx
import fitz  # PyMuPDF — renders PDF pages to PNG for the Book Extract tool
from PIL import Image  # downscale page images under Anthropic's 8000px vision limit

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


# ── DOCUMENTS TAB ────────────────────────────────────────────────────────────────
# File store for the Documents tab. Files land in the "documents" storage bucket;
# one metadata row per file goes to tier3_documents. Same service-role key + rt_auth
# token guard as the tier3 routes above. Deliberately isolated: no tier4 reads/writes.
DOCUMENTS_BUCKET = "documents"


async def _sb_storage(method, path, content=None, json=None, content_type=None):
    """Supabase Storage REST using the service-role key (apikey header ONLY, exactly
    like _sb_service — sb_secret_ keys are not JWTs). `path` is appended to
    {SUPABASE_URL}/storage/v1/."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500,
                            detail="Supabase service env vars (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY) not set")
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY}
    if content_type:
        headers["Content-Type"] = content_type
    url = f"{SUPABASE_URL}/storage/v1/{path}"
    async with httpx.AsyncClient(timeout=60) as hc:
        r = await hc.request(method, url, headers=headers, content=content, json=json)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=f"Supabase storage error: {r.text}")
    try:
        return r.json()
    except Exception:
        return {}


# UPLOAD — put the file in the bucket, then file one metadata row to tier3_documents.
@app.post("/documents/upload")
async def documents_upload(
    file: UploadFile = File(...),
    name: str = Form(...),
    _: bool = Depends(rt_auth),
):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "Empty file upload"}, status_code=400)
    original = file.filename or "file"
    # Unique object path so identically-named files never collide in the bucket.
    object_path = f"{uuid.uuid4().hex}/{original}"
    await _sb_storage(
        "POST", f"object/{DOCUMENTS_BUCKET}/{object_path}",
        content=raw,
        content_type=file.content_type or "application/octet-stream",
    )
    row = {
        "name": name,
        "filename": original,
        "file_path": object_path,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    res = await _sb_service(
        "POST", "tier3_documents",
        json=[row],
        prefer="return=representation",
    )
    return {"ok": True, "document": (res or [None])[0]}


# LIST — every row from tier3_documents, newest first.
@app.get("/documents/list")
async def documents_list(_: bool = Depends(rt_auth)):
    res = await _sb_service(
        "GET", "tier3_documents",
        params={
            "select": "id,name,filename,file_path,uploaded_at",
            "order": "uploaded_at.desc",
        },
    )
    return {"count": len(res or []), "documents": res or []}


# URL — a short-lived signed download URL for a given object path in the bucket.
@app.get("/documents/url")
async def documents_url(file_path: str, _: bool = Depends(rt_auth)):
    res = await _sb_storage(
        "POST", f"object/sign/{DOCUMENTS_BUCKET}/{file_path}",
        json={"expiresIn": 3600},
    )
    signed = res.get("signedURL") or res.get("signedUrl")
    if not signed:
        raise HTTPException(status_code=500, detail="No signed URL returned by storage")
    return {"url": f"{SUPABASE_URL}/storage/v1{signed}"}


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


# ── BOOK PDF -> TIER 3 EXTRACTS ──────────────────────────────────────────────────
# Render a captured book PDF to page images with PyMuPDF, read them in batches of 4
# with the SAME Anthropic client/model /api/ai uses, then file each returned extract
# to tier3_databank via the same _sb_service upsert path /tier3/push uses.

# Mode prompts, verbatim. Chosen from the frontend Mode dropdown.
BOOK_MODE_PROMPTS = {
    "excerpt": "Extracting for a private brain file from a book the owner legally owns. Read these page images. Pull the usable SUBSTANCE — anecdotes, decisions, philosophy, patterns. Paraphrase everything in your own words. NEVER reproduce the author's sentences. One short quote under 15 words only where the exact phrase is the point. No whole paragraphs. Page number for each. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
    "principle": "Extracting for a private brain file. Read these page images. Pull all educational substance — rules, definitions, principles, frameworks, procedures — in your own words as bare rules. Exclude the author's explanations, worked examples, and phrasing. Page number for each. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
    "framework": "Extracting for a private brain file. Read these page images. Pull the METHOD for finding and applying current law — frameworks, tests, steps — never a stored legal conclusion. Paraphrase in your own words. Page number for each. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
}


def _book_slug(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "book"


# In-memory job store for background Book Extract runs. Long PDFs mean dozens of
# Anthropic calls, which blow past Railway's request timeout (502) if done inline —
# so the route renders the PDF, kicks a daemon thread, and the client polls status.
# In-memory is fine: a lost job on restart just means re-running the extract.
JOBS = {}


def _extract_book_worker(job_id, raw, book_name, mode, prompt_text):
    """Runs in a daemon thread. Renders the PDF here (not in the request) so the POST
    returns the job_id near-instantly even for a 100-page book, then runs the same
    4-pages-per-call batch loop: updates JOBS[job_id] after each batch and upserts that
    batch to tier3_databank via the async _sb_service (asyncio.run, since we're off the
    event loop)."""
    job = JOBS[job_id]
    sl = _book_slug(book_name)
    idx = 0  # running global index -> keeps the {slug}-{page}-{idx} source_id scheme
    try:
        # Render every page to a 150-dpi PNG, base64-encoded, paired with its 1-based
        # page number. These are image-only Kindle PDFs (no text layer), so each page
        # MUST be rasterized and sent to the model as an image.
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
        except Exception as e:
            job["error"] = f"Could not read PDF: {e}"
            job["status"] = "error"
            return

        # `images` is the ONE list the batch loop iterates. total_pages is derived from
        # it below so the reported count and the loop can never disagree. Each entry is
        # (page_number, base64_jpeg, width, height).
        MAX_SIDE = 1568  # Anthropic's recommended vision max; also well under the 8000px cap
        images = []
        page_count = doc.page_count
        for i in range(page_count):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
                # Downscale so the longest side <= 1568px; a 150-dpi capture is often
                # larger than the 8000px limit, which 400s the whole batch.
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                w, h = img.size
                if max(w, h) > MAX_SIDE:
                    scale = MAX_SIDE / float(max(w, h))
                    w, h = int(round(w * scale)), int(round(h * scale))
                    img = img.resize((w, h), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)  # JPEG is fine for text pages, smaller
                b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
                images.append((i + 1, b64, w, h))
            except Exception as e:
                # Log and skip the bad page rather than silently losing the whole book.
                print(f"[book-extract] page {i + 1}/{page_count} render FAILED: {e}", flush=True)
        doc.close()

        total_pages = len(images)
        job["total_pages"] = total_pages
        num_batches = (total_pages + 3) // 4
        print(f"[book-extract] total_pages={total_pages} rendered_images={len(images)} "
              f"batches={num_batches}", flush=True)

        if not images:
            job["error"] = "PDF rendered no page images"
            job["status"] = "error"
            return

        for start in range(0, len(images), 4):
            batch = images[start:start + 4]
            batch_no = start // 4 + 1
            # Attach each page as a base64 PNG IMAGE block (vision). These PDFs are
            # image-only Kindle screenshots with NO text layer, so the model has to
            # SEE the pages — a "Page N:" label precedes each image so extracts carry
            # the right page number, then the mode prompt is the final text block.
            content = []
            num_imgs = 0
            for (pnum, b64, w, h) in batch:
                content.append({"type": "text", "text": f"Page {pnum}:"})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}})
                num_imgs += 1
            content.append({"type": "text", "text": prompt_text})
            dims = f"{batch[0][2]}x{batch[0][3]}" if batch else "0x0"
            text = ""
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": content}],
                )
                text = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
            except Exception as e:
                # Surface the real reason in Railway logs instead of silently skipping.
                print(f"[book-extract] batch {batch_no}: API call FAILED: {e}", flush=True)
            resp_chars = len(text)  # raw model response length, before fence-stripping

            # Strip a ```json fence if Claude wrapped the array.
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            batch_extracts = []
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("point"):
                            batch_extracts.append({
                                "point": item.get("point"),
                                "page": item.get("page"),
                                "theme": item.get("theme") or "",
                            })

            # One line per batch so 0-extract runs are diagnosable in Railway logs.
            print(f"[book-extract] batch {batch_no}: imgs={num_imgs} dims={dims} "
                  f"respChars={resp_chars} extracted={len(batch_extracts)}", flush=True)

            # File this batch to tier3_databank. Upsert on (section, source_id) so
            # re-running the same book never duplicates.
            rows = [{
                "section":     "book_extracts",
                "source_type": "book_extract",
                "source_id":   f"{sl}-{e.get('page')}-{idx + j}",
                "title":       f"{book_name} — {e.get('theme') or ''}",
                "content":     e.get("point"),
                "record":      {"book": book_name, "mode": mode, "page": e.get("page"),
                                "theme": e.get("theme"), "point": e.get("point")},
            } for j, e in enumerate(batch_extracts)]
            if rows:
                res = asyncio.run(_sb_service(
                    "POST", "tier3_databank",
                    params={"on_conflict": "section,source_id"},
                    json=rows,
                    prefer="resolution=merge-duplicates,return=representation",
                ))
                job["written"] += len(res or [])

            idx += len(batch_extracts)
            job["extracts"].extend(batch_extracts)
            job["done_pages"] = min(start + 4, len(images))
        job["status"] = "done"
    except Exception as e:
        # Keep whatever was already written; just flag the failure.
        job["error"] = str(e)
        job["status"] = "error"


@app.post("/api/extract-book")
async def extract_book(
    pdf: UploadFile = File(...),
    book_name: str = Form(...),
    mode: str = Form("excerpt"),
    _: bool = Depends(rt_auth),
):
    if not client:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=500)
    prompt_text = BOOK_MODE_PROMPTS.get(mode, BOOK_MODE_PROMPTS["excerpt"])
    # Only read the upload bytes in-request (fast). Rendering the PDF to page images is
    # the slow part, so it happens in the worker — the POST returns near-instantly.
    raw = await pdf.read()
    if not raw:
        return JSONResponse({"error": "Empty PDF upload"}, status_code=400)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "book": book_name,
        "mode": mode,
        "total_pages": 0,  # set once the worker finishes rendering
        "done_pages": 0,
        "written": 0,
        "extracts": [],
        "error": None,
    }
    threading.Thread(
        target=_extract_book_worker,
        args=(job_id, raw, book_name, mode, prompt_text),
        daemon=True,
    ).start()
    return {"job_id": job_id, "total_pages": 0}


@app.get("/api/extract-status/{job_id}")
async def extract_status(job_id: str, _: bool = Depends(rt_auth)):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job_id"}, status_code=404)
    return {
        "job_id":      job["id"],
        "status":      job["status"],
        "book":        job["book"],
        "mode":        job["mode"],
        "total_pages": job["total_pages"],
        "done_pages":  job["done_pages"],
        "written":     job["written"],
        "count":       len(job["extracts"]),
        "extracts":    job["extracts"],
        "error":       job["error"],
    }


# ── BOOK OCR TEXT -> TIER 3 EXTRACTS (second option; image path above is untouched) ──
# Same JOBS store + /api/extract-status polling + tier3_databank row shape as the image
# path — only the input differs (raw OCR text instead of page images). Paraphrase is
# MANDATORY here: the prompts forbid reproducing the author's sentences. That is the
# copyright line — we never store verbatim book text.
BOOK_MODE_PROMPTS_TEXT = {
    "excerpt": "Below is OCR page text from a book the owner legally owns. Pull the usable SUBSTANCE — anecdotes, decisions, philosophy, patterns. Paraphrase everything in your own words. NEVER reproduce the author's sentences or copy phrasing from the text. One short quote under 15 words only where the exact phrase is the point. No whole paragraphs. Use the 'Page N:' labels to attribute a page number to each point. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
    "principle": "Below is OCR page text from a book the owner legally owns. Pull all educational substance — rules, definitions, principles, frameworks, procedures — in your own words as bare rules. Exclude the author's explanations, worked examples, and phrasing; never copy sentences from the text. Use the 'Page N:' labels to attribute a page number to each point. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
    "framework": "Below is OCR page text from a book the owner legally owns. Pull the METHOD for finding and applying current law — frameworks, tests, steps — never a stored legal conclusion. Paraphrase in your own words; never copy sentences from the text. Use the 'Page N:' labels to attribute a page number to each point. Return ONLY a JSON array: [{\"point\":\"...\",\"page\":N,\"theme\":\"...\"}].",
}

BOOK_TEXT_CHUNK_CHARS = 8000  # bounded per-call text size for the OCR/text path


def _chunk_book_text(text):
    """Split OCR text into bounded chunks. Prefers '===== PAGE N =====' markers
    (grouping whole pages up to the char budget) and falls back to fixed char windows
    when no markers are present. Returns a list of (chunk_text, first_page_label)."""
    marker_re = re.compile(r"=+\s*PAGE\s+(\d+)\s*=+", re.IGNORECASE)
    matches = list(marker_re.finditer(text))
    chunks = []
    if matches:
        pages = []
        for i, m in enumerate(matches):
            page_no = int(m.group(1))
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                pages.append((page_no, body))
        cur, cur_len, cur_first = [], 0, None
        for (pno, body) in pages:
            block = f"Page {pno}:\n{body}"
            if cur and cur_len + len(block) > BOOK_TEXT_CHUNK_CHARS:
                chunks.append(("\n\n".join(cur), cur_first))
                cur, cur_len, cur_first = [], 0, None
            if not cur:
                cur_first = pno
            cur.append(block)
            cur_len += len(block)
        if cur:
            chunks.append(("\n\n".join(cur), cur_first))
    else:
        page = 1
        for start in range(0, len(text), BOOK_TEXT_CHUNK_CHARS):
            window = text[start:start + BOOK_TEXT_CHUNK_CHARS].strip()
            if window:
                chunks.append((f"Page {page}:\n{window}", page))
            page += 1
    return chunks


def _extract_book_text_worker(job_id, text, book_name, mode, prompt_text):
    """Text twin of _extract_book_worker: chunk OCR text, paraphrase each chunk with the
    SAME model, write the SAME tier3_databank row shape via the SAME _sb_service upsert,
    and update the SAME job fields so /api/extract-status drives the queue identically.
    'pages' in the status payload map onto chunks so the pill reads 'reading X/Y'."""
    job = JOBS[job_id]
    sl = _book_slug(book_name)
    idx = 0  # running global index -> keeps the {slug}-{page}-{idx} source_id scheme
    try:
        parts = _chunk_book_text(text)
        total = len(parts)
        job["total_pages"] = total
        print(f"[book-extract-text] book={book_name!r} chars={len(text)} chunks={total}", flush=True)
        if not parts:
            job["error"] = "No text to extract"
            job["status"] = "error"
            return

        for i, (chunk_text, page_label) in enumerate(parts):
            chunk_no = i + 1
            out = ""
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": chunk_text + "\n\n" + prompt_text}],
                )
                out = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
            except Exception as e:
                print(f"[book-extract-text] chunk {chunk_no}: API call FAILED: {e}", flush=True)
            resp_chars = len(out)  # raw model response length, before fence-stripping

            # Strip a ```json fence if Claude wrapped the array.
            if out.startswith("```"):
                out = out.split("```")[1]
                if out.startswith("json"):
                    out = out[4:]
                out = out.strip()
            chunk_extracts = []
            if out:
                try:
                    parsed = json.loads(out)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("point"):
                            chunk_extracts.append({
                                "point": item.get("point"),
                                "page": item.get("page") if item.get("page") is not None else page_label,
                                "theme": item.get("theme") or "",
                            })

            print(f"[book-extract-text] chunk {chunk_no}/{total}: "
                  f"respChars={resp_chars} extracted={len(chunk_extracts)}", flush=True)

            # SAME row shape + SAME upsert as the image path.
            rows = [{
                "section":     "book_extracts",
                "source_type": "book_extract",
                "source_id":   f"{sl}-{e.get('page')}-{idx + j}",
                "title":       f"{book_name} — {e.get('theme') or ''}",
                "content":     e.get("point"),
                "record":      {"book": book_name, "mode": mode, "page": e.get("page"),
                                "theme": e.get("theme"), "point": e.get("point")},
            } for j, e in enumerate(chunk_extracts)]
            if rows:
                res = asyncio.run(_sb_service(
                    "POST", "tier3_databank",
                    params={"on_conflict": "section,source_id"},
                    json=rows,
                    prefer="resolution=merge-duplicates,return=representation",
                ))
                job["written"] += len(res or [])

            idx += len(chunk_extracts)
            job["extracts"].extend(chunk_extracts)
            job["done_pages"] = chunk_no
        job["status"] = "done"
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "error"


class BookTextRequest(BaseModel):
    book_name: str
    text: str
    mode: str = "excerpt"


@app.post("/api/extract-book-text")
async def extract_book_text(req: BookTextRequest, _: bool = Depends(rt_auth)):
    if not client:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=500)
    prompt_text = BOOK_MODE_PROMPTS_TEXT.get(req.mode, BOOK_MODE_PROMPTS_TEXT["excerpt"])
    text = req.text or ""
    if not text.strip():
        return JSONResponse({"error": "Empty text upload"}, status_code=400)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "book": req.book_name,
        "mode": req.mode,
        "total_pages": 0,  # set to chunk count once the worker starts
        "done_pages": 0,
        "written": 0,
        "extracts": [],
        "error": None,
    }
    threading.Thread(
        target=_extract_book_text_worker,
        args=(job_id, text, req.book_name, req.mode, prompt_text),
        daemon=True,
    ).start()
    return {"job_id": job_id, "total_pages": 0}


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
  .tdy-sub-title { font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .tdy-item { font-size: 13px; color: #e0e0e0; padding: 6px 0; border-bottom: 1px solid #1e1e1e; }
  .tdy-item:last-child { border-bottom: none; }
  .tdy-time { color: #c9a84c; font-family: monospace; margin-right: 10px; }
  .tdy-empty { font-size: 13px; color: #555; padding: 6px 0; }
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
  .cc-view-card.cc-view-active { border-color: #c9a84c; background: #171207; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
  .log-box { background: #060606; border: 1px solid #1a1a1a; border-radius: 6px; padding: 14px; font-size: 12px; color: #555; font-family: monospace; height: 200px; overflow-y: auto; line-height: 1.6; }
  /* ── Book Extract holding-cell (scoped) ── */
  #panel-extract .bx-drop { border: 1px dashed #333; border-radius: 8px; background: #0a0a0a; padding: 30px 18px; text-align: center; color: #666; cursor: pointer; transition: border-color 0.2s, background 0.2s; font-size: 13px; }
  #panel-extract .bx-drop:hover { border-color: #c9a84c; color: #888; }
  #panel-extract .bx-drop.dragover { border-color: #c9a84c; background: #171207; color: #c9a84c; }
  #panel-extract .bx-drop b { color: #c9a84c; }
  #panel-extract .bx-queue { margin-top: 16px; }
  #panel-extract .bx-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; padding: 12px 0; border-bottom: 1px solid #1a1a1a; font-size: 13px; }
  #panel-extract .bx-row:last-child { border-bottom: none; }
  #panel-extract .bx-file { color: #888; font-family: monospace; font-size: 11px; }
  #panel-extract .bx-book { color: #e0e0e0; font-weight: 600; margin-top: 2px; }
  #panel-extract .bx-right { display: flex; align-items: center; gap: 10px; }
  #panel-extract .bx-pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; white-space: nowrap; background: #1a1a1a; color: #777; border: 1px solid #33333344; }
  #panel-extract .bx-pill.waiting { background: #1a1a1a; color: #777; border-color: #33333344; }
  #panel-extract .bx-pill.uploading, #panel-extract .bx-pill.reading { background: #1a1208; color: #c9a84c; border-color: #c9a84c44; }
  #panel-extract .bx-pill.done { background: #0d1a0d; color: #4caf50; border-color: #4caf5044; }
  #panel-extract .bx-pill.failed { background: #1a0d0d; color: #e05555; border-color: #e0555544; }
  #panel-extract .bx-retry { background: transparent; border: 1px solid #c9a84c; color: #c9a84c; border-radius: 6px; padding: 4px 12px; font-size: 11px; font-weight: 700; font-family: inherit; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; }
  #panel-extract .bx-retry:hover { background: #c9a84c; color: #0a0a0a; }
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
  <button class="nav-tab" onclick="switchTab('employees')">Employees</button>
  <button class="nav-tab" onclick="switchTab('tools')">Tools</button>
  <button class="nav-tab" onclick="switchTab('documents')">Documents</button>
  <!-- Tier 3 Paste tab hidden from UI. Backend /tier3/ingest route and #panel-tier3 stay intact, just unreachable from the nav.
  <button class="nav-tab" onclick="switchTab('tier3')">Tier 3 Paste</button>
  -->
</nav>

<!-- COMMAND CENTER -->
<div class="panel active" id="panel-command">
  <!-- TODAY — calendar + reminders synced from the Mac, with a local pending overlay. -->
  <div class="two-col" style="margin-bottom:20px;">
    <div class="card">
      <div class="card-title">Today</div>
      <div id="todayBody">
        <div class="result visible" style="min-height:auto;">Loading…</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">+ Add</div>
      <div style="display:flex;gap:8px;margin-bottom:14px;">
        <button type="button" id="addTypeReminder" class="tdy-type btn" onclick="tdySetType('reminder')" style="margin-top:0;">Reminder</button>
        <button type="button" id="addTypeEvent" class="tdy-type btn btn-ghost" onclick="tdySetType('event')" style="margin-top:0;">Event</button>
      </div>
      <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Title</label>
      <input type="text" id="addTitle" style="margin-bottom:14px;">
      <div id="addReminderFields">
        <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Due date</label>
        <input type="date" id="addDue" style="width:100%;background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:11px 14px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;color-scheme:dark;">
      </div>
      <div id="addEventFields" style="display:none;">
        <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Date</label>
        <input type="date" id="addDate" style="width:100%;background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:11px 14px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;color-scheme:dark;margin-bottom:14px;">
        <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Time</label>
        <input type="time" id="addTime" style="width:100%;background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:11px 14px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;color-scheme:dark;">
      </div>
      <button class="btn" id="addSubmit" onclick="tdyAdd()">Submit</button>
      <div class="result" id="addResult"></div>
    </div>
  </div>
  <!-- Command Center selector: card grid gates Sales / Agent / Marketing / Tracker / Niche below the project cards. No card active by default. -->
  <div style="margin-top:22px;">
    <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">View</label>
    <div class="tools-grid">
      <div class="tool-card cc-view-card" onclick="ccPick(this,'sales')"><div class="tool-icon">📊</div><div class="tool-name">Sales</div><div class="tool-desc">Revenue and customer metrics across the portfolio</div></div>
      <div class="tool-card cc-view-card" onclick="ccPick(this,'agent')"><div class="tool-icon">🤖</div><div class="tool-name">Agent Control</div><div class="tool-desc">Run Load, Catalog, and Gauge for Tier 4</div></div>
      <div class="tool-card cc-view-card" onclick="ccPick(this,'marketing')"><div class="tool-icon">📣</div><div class="tool-name">Marketing</div><div class="tool-desc">Drafts and campaign status</div></div>
      <div class="tool-card cc-view-card" onclick="ccPick(this,'tracker')"><div class="tool-icon">✅</div><div class="tool-name">Project Tracker</div><div class="tool-desc">Open flags and build status</div></div>
      <div class="tool-card cc-view-card" onclick="ccPick(this,'niche')"><div class="tool-icon">🎯</div><div class="tool-name">Niche Scorer</div><div class="tool-desc">Score a niche against the 5-criteria formula</div></div>
    </div>
  </div>
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

<!-- DOCUMENTS -->
<div class="panel" id="panel-documents">
  <div class="card">
    <div class="card-title">Upload Document</div>
    <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">File</label>
    <input type="file" id="docFile" style="width:100%;background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:11px 14px;color:#e0e0e0;font-size:13px;font-family:inherit;outline:none;margin-bottom:14px;" />
    <label style="display:block;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Name</label>
    <input type="text" id="docName" placeholder="Name this document..." />
    <button class="btn" id="docSaveBtn" onclick="docUpload()">Save</button>
    <div class="result" id="docStatus"></div>
  </div>
  <div class="card">
    <div class="card-title">Documents</div>
    <div id="docList"><div class="tdy-empty">Loading…</div></div>
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

<!-- BOOK EXTRACT — drag-and-drop holding cell -->
<div class="panel" id="panel-extract">
  <div class="card">
    <div class="card-title">Book Extract — Holding Cell</div>
    <p style="font-size:13px;color:#555;margin-bottom:14px;">Drop one or more book PDFs. Each is named from its file, read page by page by Claude, and filed to the Tier 3 databank — one book at a time, in order.</p>
    <select id="bxMode" style="width:100%;margin-bottom:12px;background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:11px 14px;color:#e0e0e0;font-family:inherit;font-size:14px;outline:none;">
      <option value="excerpt">Excerpt (biography/narrative)</option>
      <option value="principle">Principle (textbook/technical)</option>
      <option value="framework">Framework (legal)</option>
    </select>
    <div class="bx-drop" id="bxDrop" onclick="document.getElementById('bxFile').click()">
      Drop PDFs or .txt here, or <b>click to choose</b> — multiple at once
    </div>
    <input type="file" id="bxFile" accept=".pdf,.txt" multiple style="display:none;" onchange="bxAddFiles(this.files)" />
    <div class="bx-queue" id="bxQueue"></div>
    <button class="btn btn-ghost" onclick="bxClearQueue()">Clear Queue</button>
  </div>
</div>

<!-- TOOLS -->
<div class="panel" id="panel-tools">
  <div class="tools-grid">
    <div class="tool-card" onclick="switchTab('youtube')"><div class="tool-icon">▶️</div><div class="tool-name">YouTube Extractor</div><div class="tool-desc">Extract concepts from any YouTube video automatically</div></div>
    <div class="tool-card" onclick="switchTab('finance')"><div class="tool-icon">📈</div><div class="tool-name">Finance AI</div><div class="tool-desc">Market queries, stock lookup, Vault Trader status</div></div>
    <div class="tool-card" onclick="switchTab('extract')"><div class="tool-icon">📖</div><div class="tool-name">Book Extract</div><div class="tool-desc">Turn a captured book PDF into brain extracts</div></div>
    <div class="tool-card" onclick="window.open('https://claude.ai','_blank')"><div class="tool-icon">⚡</div><div class="tool-name">Open Claude</div><div class="tool-desc">Launch Claude in a new tab for working sessions</div></div>
    <div class="tool-card" onclick="window.open('https://github.com/randynutt31','_blank')"><div class="tool-icon">🐙</div><div class="tool-name">GitHub</div><div class="tool-desc">View and manage your repos</div></div>
    <div class="tool-card" onclick="window.open('https://railway.app','_blank')"><div class="tool-icon">🚂</div><div class="tool-name">Railway</div><div class="tool-desc">Monitor deployed services and logs</div></div>
    <div class="tool-card" onclick="window.open('https://supabase.com','_blank')"><div class="tool-icon">🗄️</div><div class="tool-name">Supabase</div><div class="tool-desc">Database management for all products</div></div>
  </div>
</div>

<!-- EMPLOYEES -->
<div class="panel" id="panel-employees">
  <div class="card">
    <div class="card-title">Employees</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0e0e0;">
      <thead>
        <tr style="text-align:left;color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;">
          <th style="padding:8px 10px;border-bottom:1px solid #222;">Name</th>
          <th style="padding:8px 10px;border-bottom:1px solid #222;">Natural Role</th>
          <th style="padding:8px 10px;border-bottom:1px solid #222;">Assigned Role</th>
          <th style="padding:8px 10px;border-bottom:1px solid #222;">Design Team</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Spunky</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Project Manager</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Spunky" onchange="empSave('Spunky', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Spunky" onchange="empSaveDt('Spunky', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Dimon</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">General Manager</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Dimon" onchange="empSave('Dimon', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Dimon" onchange="empSaveDt('Dimon', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Munger</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">The Destroyer</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Munger" onchange="empSave('Munger', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Munger" onchange="empSaveDt('Munger', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Leonard</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Capital Allocator</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Leonard" onchange="empSave('Leonard', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Leonard" onchange="empSaveDt('Leonard', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Brigs</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Tax Attorney</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Brigs" onchange="empSave('Brigs', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Brigs" onchange="empSaveDt('Brigs', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Siggy</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Lawyer</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Siggy" onchange="empSave('Siggy', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Siggy" onchange="empSaveDt('Siggy', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Deming</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Systems Engineer</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Deming" onchange="empSave('Deming', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Deming" onchange="empSaveDt('Deming', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Ogilvy</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Copywriter</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Ogilvy" onchange="empSave('Ogilvy', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Ogilvy" onchange="empSaveDt('Ogilvy', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Drucker</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">Operations Mind</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Drucker" onchange="empSave('Drucker', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Drucker" onchange="empSaveDt('Drucker', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="N/A" selected>N/A</option><option value="YES">YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Torvalds</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">The Engine</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Torvalds" onchange="empSave('Torvalds', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Torvalds" onchange="empSaveDt('Torvalds', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#4ade80;font-family:inherit;font-size:13px;outline:none;"><option value="N/A">N/A</option><option value="YES" selected>YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Spolsky</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">The Craft</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Spolsky" onchange="empSave('Spolsky', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Spolsky" onchange="empSaveDt('Spolsky', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#4ade80;font-family:inherit;font-size:13px;outline:none;"><option value="N/A">N/A</option><option value="YES" selected>YES</option></select></td>
        </tr>
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #222;font-weight:600;">Ive</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;color:#888;">The Feel</td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="emp-Ive" onchange="empSave('Ive', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#e0e0e0;font-family:inherit;font-size:13px;outline:none;"><option value="-" selected>-</option><option value="Sales">Sales</option><option value="Marketing">Marketing</option><option value="Operations">Operations</option><option value="Legal">Legal</option><option value="Tax">Tax</option><option value="Engineering">Engineering</option><option value="Systems">Systems</option><option value="Copywriting">Copywriting</option><option value="Capital">Capital</option></select></td>
          <td style="padding:8px 10px;border-bottom:1px solid #222;"><select id="empdt-Ive" onchange="empSaveDt('Ive', this.value)" style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:6px 8px;color:#4ade80;font-family:inherit;font-size:13px;outline:none;"><option value="N/A">N/A</option><option value="YES" selected>YES</option></select></td>
        </tr>
      </tbody>
    </table>
  </div>
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
  const tabs = ['command','finance','employees','tools','documents','tier3'];
  document.querySelectorAll('.nav-tab').forEach((t,i) => t.classList.toggle('active', tabs[i] === tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  if (tab === 'employees') empRestore();
  if (tab === 'documents') docLoad();
}

// EMPLOYEES tab -- static roster. Assigned Role persists in localStorage keyed by name.
var EMP_NAMES = ['Spunky','Dimon','Munger','Leonard','Brigs','Siggy','Deming','Ogilvy','Drucker','Torvalds','Spolsky','Ive'];
var EMP_DT_YES = { Torvalds: 1, Spolsky: 1, Ive: 1 };
function empSave(name, value) { try { localStorage.setItem('emp_role_' + name, value); } catch (e) {} }
function empSaveDt(name, value) {
  try { localStorage.setItem('emp_dt_' + name, value); } catch (e) {}
  var el = document.getElementById('empdt-' + name);
  if (el) el.style.color = (value === 'YES' ? '#4ade80' : '#e0e0e0');
}
function empRestore() {
  try {
    for (var i = 0; i < EMP_NAMES.length; i++) {
      var el = document.getElementById('emp-' + EMP_NAMES[i]);
      if (!el) continue;
      var saved = localStorage.getItem('emp_role_' + EMP_NAMES[i]);
      el.value = (saved === null ? '-' : saved);
    }
    for (var j = 0; j < EMP_NAMES.length; j++) {
      var name = EMP_NAMES[j];
      var dt = document.getElementById('empdt-' + name);
      if (!dt) continue;
      var savedDt = localStorage.getItem('emp_dt_' + name);
      var def = EMP_DT_YES[name] ? 'YES' : 'N/A';
      dt.value = (savedDt === null ? def : savedDt);
      dt.style.color = (dt.value === 'YES' ? '#4ade80' : '#e0e0e0');
    }
  } catch (e) {}
}

// Command Center dropdown: gate Sales / Agent / Marketing / Tracker / Niche below
// the project cards. Empty value ("-- Select --") hides all — the empty default.
// Sales and Marketing load only when their option is selected (not on tab entry).
function ccShow(val) {
  ['cc-sales','cc-agent','cc-marketing','cc-tracker','cc-niche'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  if (!val) return;
  const el = document.getElementById('cc-' + val);
  if (el) el.style.display = 'block';
  if (val === 'sales') loadSalesTracker();
  if (val === 'marketing') loadMarketing();
}

// Command Center view cards: move the active accent to the clicked card, then
// delegate to ccShow (hide/show logic unchanged). No card active on load.
function ccPick(el, val) {
  document.querySelectorAll('.cc-view-card').forEach(c => c.classList.remove('cc-view-active'));
  el.classList.add('cc-view-active');
  ccShow(val);
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

// TODAY — Calendar + Reminders synced from the Mac, plus a locally-added pending overlay.
// Reads the same way the Marketing tab does: /tier3/records by section, then the
// mktManifest rec.record -> JSON.parse(rec.content) fallback. Reuses T4_TOKEN.
const TODAY = { addType: 'reminder' };

// Read one record out of a Tier 3 section by source_id. Returns its parsed manifest,
// or null when the section/record is absent or the request fails (treated as empty).
async function tdyReadRecord(section, sourceId) {
  try {
    const res = await fetch('/tier3/records?section=' + encodeURIComponent(section) + '&limit=500', {
      headers: {'X-API-Token': T4_TOKEN}
    });
    if (!res.ok) return null;
    const data = await res.json();
    const recs = (data && data.records) || [];
    const rec = recs.find(r => String(r.source_id || '') === sourceId);
    return rec ? mktManifest(rec) : null;
  } catch(e) { return null; }
}

async function loadToday() {
  const body = document.getElementById('todayBody');
  if (!body) return;
  const mac = await tdyReadRecord('reminders_calendar', 'mac');
  const pendingRaw = await tdyReadRecord('reminders_calendar_pending', 'dashboard');
  const pending = Array.isArray(pendingRaw) ? pendingRaw : [];
  tdyRenderToday(body, mac, pending);
}

function tdyRenderToday(body, mac, pending) {
  const hasMac = mac && typeof mac === 'object';
  if (!hasMac && !pending.length) {
    body.innerHTML = '<div class="result visible" style="min-height:auto;">Waiting for first sync from Mac.</div>';
    return;
  }
  const events = (hasMac && Array.isArray(mac.events)) ? mac.events : [];
  const reminders = (hasMac && Array.isArray(mac.reminders)) ? mac.reminders : [];
  const pendEvents = pending.filter(p => p && p.type === 'event');
  const pendReminders = pending.filter(p => p && p.type === 'reminder');

  // Calendar: synced events (time + title) then any pending events, tagged "pending sync".
  let cal = events.map(e =>
    '<div class="tdy-item"><span class="tdy-time">' + mktEsc(e.time || '') + '</span>' + mktEsc(e.title || '') + '</div>');
  cal = cal.concat(pendEvents.map(p =>
    '<div class="tdy-item"><span class="badge backburner" style="margin-right:8px;">pending sync</span>' +
    (p.time ? '<span class="tdy-time">' + mktEsc(p.time) + '</span>' : '') + mktEsc(p.title || '') + '</div>'));
  const calHtml = cal.length ? cal.join('') : '<div class="tdy-empty">Nothing today</div>';

  // Reminders: synced reminders (red "overdue" tag when overdue) then pending reminders.
  let rem = reminders.map(r =>
    '<div class="tdy-item">' + (r.overdue ? '<span class="badge blocked" style="margin-right:8px;">overdue</span>' : '') +
    mktEsc(r.title || '') + '</div>');
  rem = rem.concat(pendReminders.map(p =>
    '<div class="tdy-item"><span class="badge backburner" style="margin-right:8px;">pending sync</span>' +
    mktEsc(p.title || '') + '</div>'));
  const remHtml = rem.length ? rem.join('') : '<div class="tdy-empty">Nothing today</div>';

  body.innerHTML =
    '<div class="tdy-sub-title">Calendar</div>' + calHtml +
    '<div class="tdy-sub-title" style="margin-top:16px;">Reminders</div>' + remHtml;
}

// + Add card — Reminder/Event toggle swaps the gold-fill accent and the due vs date+time fields.
function tdySetType(t) {
  TODAY.addType = t;
  const rBtn = document.getElementById('addTypeReminder');
  const eBtn = document.getElementById('addTypeEvent');
  if (rBtn) rBtn.className = 'tdy-type btn' + (t === 'reminder' ? '' : ' btn-ghost');
  if (eBtn) eBtn.className = 'tdy-type btn' + (t === 'event' ? '' : ' btn-ghost');
  const rf = document.getElementById('addReminderFields');
  const ef = document.getElementById('addEventFields');
  if (rf) rf.style.display = (t === 'reminder' ? 'block' : 'none');
  if (ef) ef.style.display = (t === 'event' ? 'block' : 'none');
}

function tdyUuid() {
  try { if (window.crypto && crypto.randomUUID) return crypto.randomUUID(); } catch(e) {}
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.floor(Math.random() * 16), v = (c === 'x') ? r : ((r & 0x3) | 0x8);
    return v.toString(16);
  });
}

// Read-modify-write the whole pending array, then push it back to the SAME
// section/source_id via /tier3/push (existing route + token — no new endpoint).
async function tdyAdd() {
  const result = document.getElementById('addResult');
  const btn = document.getElementById('addSubmit');
  const title = (document.getElementById('addTitle').value || '').trim();
  const type = TODAY.addType;
  result.className = 'result visible';
  if (!title) { result.className = 'result visible error'; result.textContent = 'Title is required.'; return; }
  const item = { id: tdyUuid(), type: type, title: title, created_at: new Date().toISOString() };
  if (type === 'reminder') {
    const due = document.getElementById('addDue').value;
    if (!due) { result.className = 'result visible error'; result.textContent = 'Due date is required.'; return; }
    item.due = due;
  } else {
    const date = document.getElementById('addDate').value;
    const time = document.getElementById('addTime').value;
    if (!date) { result.className = 'result visible error'; result.textContent = 'Date is required.'; return; }
    item.date = date; item.time = time;
  }
  btn.disabled = true;
  result.textContent = '⬤ Saving…';
  try {
    const existing = await tdyReadRecord('reminders_calendar_pending', 'dashboard');
    const arr = Array.isArray(existing) ? existing.slice() : [];
    arr.push(item);
    const res = await fetch('/tier3/push', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Token': T4_TOKEN},
      body: JSON.stringify({ records: [{
        section: 'reminders_calendar_pending',
        source_id: 'dashboard',
        source_type: 'dashboard_pending',
        title: 'Dashboard pending reminders/events',
        content: JSON.stringify(arr),
        record: arr
      }]})
    });
    const data = await res.json();
    if (!res.ok || data.ok !== true) throw new Error(data.detail || 'Push failed');
    document.getElementById('addTitle').value = '';
    document.getElementById('addDue').value = '';
    document.getElementById('addDate').value = '';
    document.getElementById('addTime').value = '';
    result.className = 'result visible success';
    result.textContent = 'Added.';
    loadToday();
  } catch(e) {
    result.className = 'result visible error';
    result.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

// Command Center is the default-active tab, so load once now, then poll both records
// every 5 minutes while the tab is open.
loadToday();
setInterval(() => {
  const p = document.getElementById('panel-command');
  if (p && p.classList.contains('active')) loadToday();
}, 300000);

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

// BOOK EXTRACT — drag-and-drop holding cell. Files queue up and process strictly
// ONE AT A TIME (only one upload in flight), so long PDFs never pile up into a
// timeout. Each file is auto-named from its filename; the Mode dropdown applies to
// the whole queue.
var BX_QUEUE = [];   // {id, file, fileName, book, status, done, total, count, error}
var BX_SEQ = 0;
var BX_RUNNING = false;

// Musk_pt1.pdf -> "Musk": drop ".pdf"/".txt", then a trailing part marker (_pt1 / pt3 / -pt4).
function bxDeriveBook(fileName) {
  var n = String(fileName || '').replace(/\\.(pdf|txt)$/i, '');
  n = n.replace(/[ _-]+pt\\s*\\d+$/i, '');
  return n.trim() || 'Book';
}

function bxPillText(item) {
  if (item.status === 'waiting') return 'waiting';
  if (item.status === 'uploading') return 'uploading';
  if (item.status === 'reading') return 'reading ' + (item.done || 0) + '/' + (item.total || 0);
  if (item.status === 'done') return 'done (' + (item.count || 0) + ' extracts)';
  if (item.status === 'failed') return 'failed';
  return item.status;
}

function bxRenderQueue() {
  const q = document.getElementById('bxQueue');
  if (!q) return;
  if (!BX_QUEUE.length) { q.innerHTML = ''; return; }
  q.innerHTML = BX_QUEUE.map(function(item) {
    const retry = item.status === 'failed'
      ? '<button class="bx-retry" onclick="bxRetry(' + item.id + ')">Retry</button>' : '';
    const err = (item.status === 'failed' && item.error)
      ? '<div class="bx-file" style="color:#e05555;margin-top:3px;">' + rtEsc(item.error) + '</div>' : '';
    return '<div class="bx-row"><div>' +
        '<div class="bx-book">' + rtEsc(item.book) + '</div>' +
        '<div class="bx-file">' + rtEsc(item.fileName) + '</div>' + err +
      '</div><div class="bx-right">' +
        '<span class="bx-pill ' + item.status + '">' + rtEsc(bxPillText(item)) + '</span>' +
        retry +
      '</div></div>';
  }).join('');
}

function bxAddFiles(files) {
  const list = Array.prototype.slice.call(files || []);
  list.forEach(function(f) {
    if (!/\\.(pdf|txt)$/i.test(f.name)) return;  // PDFs and .txt only
    BX_QUEUE.push({
      id: ++BX_SEQ, file: f, fileName: f.name, book: bxDeriveBook(f.name),
      status: 'waiting', done: 0, total: 0, count: 0, error: null
    });
  });
  // Reset the input so choosing the same file again still fires onchange.
  const fileEl = document.getElementById('bxFile');
  if (fileEl) fileEl.value = '';
  bxRenderQueue();
  bxRunner();
}

function bxRetry(id) {
  const item = BX_QUEUE.find(function(it) { return it.id === id; });
  if (!item) return;
  item.status = 'waiting'; item.error = null; item.done = 0; item.total = 0; item.count = 0;
  bxRenderQueue();
  bxRunner();
}

// Empty ONLY the UI holding-cell array so files stop stacking between runs. Never
// touches Tier 3 book_extracts (those are written server-side and are the product).
// Guard: keep any actively-processing item ('uploading'/'reading') so an in-flight
// extraction is not wiped mid-run; everything else (waiting/done/failed) is cleared.
function bxClearQueue() {
  BX_QUEUE = BX_QUEUE.filter(function(it) {
    return it.status === 'uploading' || it.status === 'reading';
  });
  bxRenderQueue();
}

// Strictly sequential: pull the next waiting item, fully finish it, then the next.
// The BX_RUNNING guard means adding files or hitting Retry mid-run never starts a
// second concurrent runner — the existing loop re-scans the queue and picks them up.
async function bxRunner() {
  if (BX_RUNNING) return;
  BX_RUNNING = true;
  try {
    while (true) {
      const item = BX_QUEUE.find(function(it) { return it.status === 'waiting'; });
      if (!item) break;
      await bxProcess(item);
    }
  } finally {
    BX_RUNNING = false;
  }
}

async function bxProcess(item) {
  item.status = 'uploading'; item.done = 0; item.total = 0; bxRenderQueue();
  var jobId;
  try {
    const mode = document.getElementById('bxMode').value;
    var res;
    if (/\\.txt$/i.test(item.fileName)) {
      // TEXT path: browser reads the whole .txt and POSTs it once as JSON; the
      // backend chunks + paraphrases. Same {job_id} + status polling as PDFs.
      const text = await item.file.text();
      res = await fetch('/api/extract-book-text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-API-Token': RT.token},
        body: JSON.stringify({ book_name: item.book, text: text, mode: mode })
      });
    } else {
      // IMAGE path — unchanged: multipart PDF upload to the existing route.
      const fd = new FormData();
      fd.append('pdf', item.file);
      fd.append('book_name', item.book);
      fd.append('mode', mode);
      res = await fetch('/api/extract-book', {
        method: 'POST',
        headers: {'X-API-Token': RT.token},
        body: fd
      });
    }
    const data = await res.json();
    if (!res.ok || data.error || !data.job_id) {
      throw new Error(data.error || data.detail || ('HTTP ' + res.status));
    }
    jobId = data.job_id;
  } catch(e) {
    item.status = 'failed'; item.error = e.message; bxRenderQueue(); return;
  }
  item.status = 'reading'; bxRenderQueue();
  await bxPollJob(item, jobId);
}

// Poll one job every 3s until it reaches a terminal state, then resolve so the
// runner advances. Any error (network, 502, or job status "error") -> failed row.
function bxPollJob(item, jobId) {
  return new Promise(function(resolve) {
    const iv = setInterval(async function() {
      try {
        const res = await fetch('/api/extract-status/' + jobId, {
          headers: {'X-API-Token': RT.token}
        });
        const data = await res.json();
        if (!data.status) { throw new Error(data.error || data.detail || ('HTTP ' + res.status)); }
        item.done = data.done_pages || 0;
        item.total = data.total_pages || 0;
        item.count = data.written || 0;
        if (data.status === 'running') { bxRenderQueue(); return; }
        clearInterval(iv);
        if (data.status === 'error') {
          item.status = 'failed';
          item.error = (data.error || 'Job failed') + ' — ' + (data.written || 0) + ' partial saved to Tier 3';
        } else {
          item.status = 'done';
          item.count = data.written || 0;
        }
        bxRenderQueue();
        resolve();
      } catch(e) {
        clearInterval(iv);
        item.status = 'failed'; item.error = e.message; bxRenderQueue(); resolve();
      }
    }, 3000);
  });
}

// Drag-and-drop wiring on the drop zone.
(function() {
  const drop = document.getElementById('bxDrop');
  if (!drop) return;
  drop.addEventListener('dragover', function(e) { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', function(e) { e.preventDefault(); drop.classList.remove('dragover'); });
  drop.addEventListener('drop', function(e) {
    e.preventDefault(); drop.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files) bxAddFiles(e.dataTransfer.files);
  });
})();

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

// ═══════════════ Documents ═══════════════
// Same X-API-Token guard (RT.token) as every other tab. Upload -> "documents"
// bucket + tier3_documents row; list newest-first; each row opens a signed URL.
function docFmtDate(s) {
  if (!s) return '';
  try { return new Date(s).toLocaleString(); } catch(e) { return s; }
}

async function docLoad() {
  const list = document.getElementById('docList');
  list.innerHTML = '<div class="tdy-empty">Loading…</div>';
  try {
    const res = await fetch('/documents/list', { headers: {'X-API-Token': RT.token} });
    if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
    const data = await res.json();
    const docs = data.documents || [];
    if (!docs.length) { list.innerHTML = '<div class="tdy-empty">No documents yet.</div>'; return; }
    list.innerHTML = '';
    docs.forEach(d => {
      const row = document.createElement('div');
      row.className = 'tracker-row';
      row.style.gridTemplateColumns = '2fr 2fr 1.5fr auto';
      const name = document.createElement('div');
      name.style.color = '#e0e0e0'; name.style.fontWeight = '600';
      name.textContent = d.name || '(unnamed)';
      const fname = document.createElement('div');
      fname.style.color = '#888'; fname.style.fontFamily = 'monospace'; fname.style.fontSize = '11px';
      fname.textContent = d.filename || '';
      const date = document.createElement('div');
      date.style.color = '#555'; date.style.fontSize = '12px';
      date.textContent = docFmtDate(d.uploaded_at);
      const btnWrap = document.createElement('div');
      const btn = document.createElement('button');
      btn.className = 'btn btn-ghost'; btn.style.marginTop = '0'; btn.style.padding = '6px 14px';
      btn.textContent = 'Open';
      btn.onclick = () => docOpen(d.file_path, btn);
      btnWrap.appendChild(btn);
      row.appendChild(name); row.appendChild(fname); row.appendChild(date); row.appendChild(btnWrap);
      list.appendChild(row);
    });
  } catch(e) {
    list.innerHTML = '<div class="result visible error">✗ ' + e.message + '</div>';
  }
}

async function docOpen(filePath, btn) {
  const label = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '…'; }
  try {
    const res = await fetch('/documents/url?file_path=' + encodeURIComponent(filePath), { headers: {'X-API-Token': RT.token} });
    if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
    const data = await res.json();
    if (data.url) window.open(data.url, '_blank');
    else throw new Error('No URL returned');
  } catch(e) {
    alert('Could not open file: ' + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

async function docUpload() {
  const fileInput = document.getElementById('docFile');
  const nameInput = document.getElementById('docName');
  const status = document.getElementById('docStatus');
  const btn = document.getElementById('docSaveBtn');
  const file = fileInput.files && fileInput.files[0];
  const name = nameInput.value.trim();
  if (!file) { status.className = 'result visible error'; status.textContent = '✗ Choose a file first.'; return; }
  if (!name) { status.className = 'result visible error'; status.textContent = '✗ Enter a name first.'; return; }
  btn.disabled = true;
  status.className = 'result visible';
  status.textContent = '⬤ Uploading…';
  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('name', name);
    const res = await fetch('/documents/upload', {
      method: 'POST',
      headers: {'X-API-Token': RT.token},
      body: fd
    });
    if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
    const data = await res.json();
    if (data.ok) {
      status.className = 'result visible success';
      status.textContent = '✓ Saved: ' + name;
      fileInput.value = '';
      nameInput.value = '';
      docLoad();
    } else {
      throw new Error(data.error || 'Failed');
    }
  } catch(e) {
    status.className = 'result visible error';
    status.textContent = '✗ ' + e.message;
  } finally {
    btn.disabled = false;
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
