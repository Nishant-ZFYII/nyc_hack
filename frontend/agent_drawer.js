/* ═══════════════════════════════════════════════════════════════════════
   agent_drawer.js — Right-side slide-out panel for the NeMo agent query.
   ═════════════════════════════════════════════════════════════════════════
   Usage:
       AgentDrawer.mount({
         side: 'user',                  // or 'admin'
         endpoint: '/api/agent/nat',    // or '/api/admin/agent/nat'
         cannedQuery: 'Find me a shelter in Brooklyn tonight',
         title: 'NeMo Agent',
       });
   ═════════════════════════════════════════════════════════════════════════ */
(function () {
  const state = { opts: null, drawerEl: null, toggleBtn: null };

  function mount(opts) {
    state.opts = Object.assign({
      side: 'user',
      endpoint: '/api/agent/nat',
      cannedQuery: 'Find me a shelter in Brooklyn tonight',
      title: 'NeMo Agent',
    }, opts || {});

    // Toggle button (top-left so it doesn't collide with stats-hud at top-right)
    const b = document.createElement('button');
    b.className = 'agent-drawer-toggle';
    b.innerHTML = '&#129302; ' + state.opts.title.toUpperCase();
    b.onclick = () => toggle();
    document.body.appendChild(b);
    state.toggleBtn = b;

    const d = document.createElement('div');
    d.className = 'agent-drawer';
    d.id = 'agent-drawer';
    d.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
        <span style="font-size:22px">&#129302;</span>
        <div style="flex:1">
          <div style="font-weight:600;font-size:14px">${state.opts.title}</div>
          <div style="font-size:10px;color:var(--t3);font-family:var(--mono);letter-spacing:0.12em">LOCAL · LLAMA3 · NVIDIA NEMO AGENT TOOLKIT</div>
        </div>
        <button class="topbar-back" onclick="AgentDrawer.close()" style="padding:4px 10px">×</button>
      </div>
      <div style="color:var(--t2);font-size:12px;line-height:1.6;margin-bottom:12px">
        Ask a question — the agent plans + calls real tools on the resource graph.
        Not on the recording loop; fires only when you click.
      </div>
      <button id="agent-drawer-canned" class="scenario-pill active" style="width:100%;padding:12px 14px;margin-bottom:14px;font-family:var(--mono);font-size:11px">
        &rarr; ${state.opts.cannedQuery}
      </button>
      <div id="agent-drawer-status" style="flex:1;overflow-y:auto;font-size:12px;line-height:1.6;color:var(--t2);background:rgba(7,8,12,0.6);border:1px solid var(--border);border-radius:8px;padding:12px;white-space:pre-wrap"></div>
    `;
    document.body.appendChild(d);
    state.drawerEl = d;

    d.querySelector('#agent-drawer-canned').onclick = () => ask(state.opts.cannedQuery);
  }

  function open() { if (state.drawerEl) state.drawerEl.classList.add('open'); }
  function close() { if (state.drawerEl) state.drawerEl.classList.remove('open'); }
  function toggle() {
    if (!state.drawerEl) return;
    state.drawerEl.classList.toggle('open');
  }

  async function ask(query) {
    const el = document.getElementById('agent-drawer-status');
    if (!el) return;
    el.textContent = '◎ planning + dispatching tools…';
    try {
      const r = await fetch(state.opts.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const d = await r.json();
      if (d.answer) {
        el.textContent = d.answer;
      } else if (d.error) {
        el.innerHTML = `<span style="color:var(--amber)">Agent offline</span>\n\n${d.error}\n\n<span style="color:var(--t3)">Tip: run <code>ollama pull llama3</code> + ensure Ollama is on 11434.</span>`;
      } else {
        el.textContent = JSON.stringify(d, null, 2);
      }
    } catch (e) {
      el.innerHTML = `<span style="color:var(--amber)">Network error.</span>\n\n${e.message}`;
    }
  }

  window.AgentDrawer = { mount, open, close, toggle, ask };
})();
