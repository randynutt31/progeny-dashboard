"""
Progyny Infinite Dashboard
Hosted on Railway. Open from any device, any browser.
All AI calls route through server — no browser CORS issues.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
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
  const tabs = ['command','tracker','agent','niche','youtube','finance','tools'];
  document.querySelectorAll('.nav-tab').forEach((t,i) => t.classList.toggle('active', tabs[i] === tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
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
