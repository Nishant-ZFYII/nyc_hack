/* ═══════════════════════════════════════════════════════════════════════
   scenario_player.js — 3D Live-Ops Map + auto-play scenario loop
   ═════════════════════════════════════════════════════════════════════════
   Self-contained module. Drop a <div id="liveMapEl"> on the page, then:
       ScenarioPlayer.init('liveMapEl')
       ScenarioPlayer.startLoop()          // begins cold→migrant→reset cycle
       ScenarioPlayer.setAuto(true)        // or false to stop autoplay

   The HUD + title + pill controls are injected automatically by .init().
   The module talks to the backend via:
       GET  /api/resources           all 7 K sites (backdrop columns)
       GET  /api/vulnerability       ~2 K hex grid (heatmap)
       POST /api/scenario/{name}     run a scenario, returns demand/sites/arcs
   ═════════════════════════════════════════════════════════════════════════ */

(function () {
  const NYC = [-73.98, 40.74];

  // Neon palette by resource_type (RGBA, 0-255)
  const TYPE_COLORS_SP = {
    shelter:          [0, 229, 255, 220],
    food_bank:        [255, 159, 67, 220],
    food_pantry:      [255, 159, 67, 220],
    hospital:         [255, 85, 119, 230],
    clinic:           [255, 123, 123, 220],
    school:           [77, 171, 255, 210],
    childcare:        [178, 75, 255, 210],
    benefits_center:  [92, 255, 177, 220],
    domestic_violence:[255, 204, 51, 220],
    legal_aid:        [51, 240, 216, 210],
    senior_services:  [178, 75, 255, 200],
    mental_health:    [255, 204, 51, 200],
    cooling_center:   [77, 171, 255, 210],
    community_center: [178, 75, 255, 200],
    youth_services:   [92, 255, 177, 180],
    transit_station:  [150, 170, 200, 170],
    emergency_services:[255, 85, 119, 220],
    nycha:            [92, 255, 177, 160],
    other:            [150, 170, 200, 180],
  };
  const COLOR_DEFAULT_SP = [150, 170, 200, 180];

  // ─────────────────────────────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────────────────────────────
  const state = {
    map: null,
    overlay: null,
    containerId: null,
    allResources: [],
    vulnerabilityHexes: [],
    buildings: [],               // sampled PLUTO lots with floor counts
    currentScenario: null,
    auto: true,
    loopHandle: null,
    orbitHandle: null,
    orbitAngle: -20,
    orbitPaused: false,
    statsEl: null,
    titleEl: null,
    pillEls: {},
    tickStart: Date.now(),
    peopleServed: 0,
  };

  const PHASES = ['cold_emergency', 'migrant_bus', 'reset'];
  const PHASE_DURATIONS = { cold_emergency: 10000, migrant_bus: 10000, reset: 3000 };

  // ─────────────────────────────────────────────────────────────────────
  // Map init
  // ─────────────────────────────────────────────────────────────────────
  function init(containerId) {
    state.containerId = containerId;
    const el = document.getElementById(containerId);
    if (!el) { console.warn('[ScenarioPlayer] container missing:', containerId); return; }
    if (state.map) return;

    state.map = new maplibregl.Map({
      container: containerId,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: NYC,
      zoom: 10.8,
      pitch: 58,
      bearing: state.orbitAngle,
      antialias: true,
    });
    state.map.addControl(new maplibregl.NavigationControl(), 'top-left');
    state.map.on('load', async () => {
      state.map.resize();
      await Promise.all([loadResources(), loadVulnerability(), loadBuildings()]);
      renderLayers();
    });
    state.map.on('mousedown', () => { state.orbitPaused = true; });
    state.map.on('touchstart', () => { state.orbitPaused = true; });

    injectOverlays();
    startOrbit();
  }

  async function loadResources() {
    try {
      const r = await fetch('/api/resources');
      if (!r.ok) return;
      const arr = await r.json();
      state.allResources = arr
        .filter(x => x && (x.lat || x.latitude) && (x.lon || x.longitude))
        .map(x => ({
          ...x,
          lat: Number(x.lat || x.latitude),
          lon: Number(x.lon || x.longitude),
          resource_type: x.type || x.resource_type || 'other',
          capacity: Number(x.capacity || x.n_beds || x.beds || 10),
        }))
        .filter(x => isFinite(x.lat) && isFinite(x.lon));
    } catch (e) { /* silent */ }
  }

  async function loadVulnerability() {
    try {
      const r = await fetch('/api/vulnerability');
      if (!r.ok) return;
      const d = await r.json();
      state.vulnerabilityHexes = (d.hexes || []).filter(h => isFinite(h.lat) && isFinite(h.lon));
    } catch (e) { /* silent */ }
  }

  async function loadBuildings() {
    try {
      const r = await fetch('/api/buildings?limit=28000');
      if (!r.ok) return;
      const arr = await r.json();
      state.buildings = (arr || [])
        .filter(b => b && isFinite(b.lat) && isFinite(b.lon))
        .map(b => ({ lat: b.lat, lon: b.lon, floors: Math.max(1, Math.min(100, b.floors || 2)) }));
    } catch (e) { /* silent */ }
  }

  // ─────────────────────────────────────────────────────────────────────
  // Layer factory
  // ─────────────────────────────────────────────────────────────────────
  function renderLayers() {
    if (!state.map) return;
    const layers = [];

    // Very subtle ambient violet glow — NOT a heatmap of truth.
    if (state.vulnerabilityHexes.length) {
      layers.push(new deck.HeatmapLayer({
        id: 'sp-vuln',
        data: state.vulnerabilityHexes,
        getPosition: d => [d.lon, d.lat],
        getWeight: d => d.weight || 1,
        radiusPixels: 35,
        intensity: 0.28,
        threshold: 0.08,
        colorRange: [[0,0,0,0],[30,10,50,25],[60,20,90,40],[90,30,120,55],[120,40,150,70]],
      }));
    }

    // REAL NYC SKYLINE — 28K PLUTO lots rendered as tiny square building
    // extrusions with actual numfloors heights. This is the city underneath;
    // resources + arcs sit on top.
    if (state.buildings && state.buildings.length) {
      layers.push(new deck.ColumnLayer({
        id: 'sp-buildings',
        data: state.buildings,
        diskResolution: 4,          // square footprint — looks like a building
        radius: 16,
        angle: 45,
        extruded: true,
        pickable: false,
        getPosition: d => [d.lon, d.lat],
        // 3.2 m per floor ≈ NYC average
        getElevation: d => Math.max(12, (d.floors || 2) * 3.4),
        // Dim cool grey-blue so it reads as "the city" not data
        getFillColor: d => {
          const tall = Math.min(1, (d.floors || 2) / 40);
          return [
            70 + Math.round(tall * 40),
            92 + Math.round(tall * 50),
            130 + Math.round(tall * 60),
            225,
          ];
        },
        material: { ambient: 0.35, diffuse: 0.6, shininess: 60, specularColor: [120,160,220] },
      }));
    }

    // All ~2.5K resources as glowing neon "beacon" columns — hexagonal
    // footprint to distinguish them from the square buildings underneath.
    if (state.allResources.length) {
      layers.push(new deck.ColumnLayer({
        id: 'sp-city',
        data: state.allResources,
        diskResolution: 6,          // hex — reads as infrastructure
        radius: 38,
        extruded: true,
        pickable: true,
        getPosition: d => [d.lon, d.lat],
        getFillColor: d => {
          const c = TYPE_COLORS_SP[d.resource_type] || COLOR_DEFAULT_SP;
          return [c[0], c[1], c[2], 210];
        },
        // Heights between 60 m and ~450 m so beacons sit above the skyline
        getElevation: d => 120 + Math.log1p(d.capacity || 20) * 110,
        material: { ambient: 0.55, diffuse: 0.8, shininess: 90, specularColor: [200, 230, 255] },
        onHover: info => showTooltip(info),
      }));
    }

    // Scenario-specific layers
    const sc = state.currentScenario;
    if (sc && sc.arcs && sc.arcs.length) {
      // Bright columns at each active site
      if (sc.sites && sc.sites.length) {
        layers.push(new deck.ColumnLayer({
          id: 'sp-sites',
          data: sc.sites,
          diskResolution: 18,
          radius: 90,
          extruded: true,
          pickable: false,
          getPosition: d => [d.lon, d.lat],
          getFillColor: d => {
            const c = TYPE_COLORS_SP[d.type] || COLOR_DEFAULT_SP;
            const saturate = 0.7 + 0.3 * (d.used / Math.max(1, d.cap));
            return [
              Math.min(255, Math.floor(c[0] * saturate)),
              Math.min(255, Math.floor(c[1] * saturate)),
              Math.min(255, Math.floor(c[2] * saturate)),
              240,
            ];
          },
          getElevation: d => Math.log1p(d.used || 1) * 300 + 120,
          material: { ambient: 0.6, diffuse: 0.9, shininess: 120, specularColor: [255, 255, 255] },
        }));
      }
      // Demand scatter — each person as a small glowing dot
      if (sc.demand && sc.demand.length) {
        layers.push(new deck.ScatterplotLayer({
          id: 'sp-demand',
          data: sc.demand,
          getPosition: d => [d.lon, d.lat],
          getRadius: 40,
          radiusMinPixels: 3,
          radiusMaxPixels: 7,
          getFillColor: sc.phase === 'migrant_bus' ? [178, 75, 255, 230] : [255, 204, 51, 230],
          stroked: false,
        }));
      }
      // Arcs — cyan for cold-emergency, magenta for migrant
      layers.push(new deck.ArcLayer({
        id: 'sp-arcs',
        data: sc.arcs,
        getSourcePosition: d => d.from,
        getTargetPosition: d => d.to,
        getSourceColor: d => d.color || [0, 229, 255, 200],
        getTargetColor: d => {
          const c = d.color || [0, 229, 255, 200];
          return [Math.min(255, c[0]+40), Math.min(255, c[1]+40), Math.min(255, c[2]+40), 255];
        },
        getWidth: d => Math.max(1.2, 2.2 * (d.weight || 1)),
        widthMinPixels: 1.2,
        greatCircle: false,
      }));
    }

    const props = { layers };
    if (state.overlay) state.overlay.setProps(props);
    else { state.overlay = new deck.MapboxOverlay(props); state.map.addControl(state.overlay); }
  }

  // ─────────────────────────────────────────────────────────────────────
  // Scenario loop
  // ─────────────────────────────────────────────────────────────────────
  let phaseIdx = 0;
  let progressHandle = null;
  function animateProgress(durationMs) {
    const bar = document.getElementById('sp-progress-bar');
    if (!bar) return;
    bar.style.transition = 'none';
    bar.style.width = '0%';
    // Force reflow before restoring transition
    void bar.offsetWidth;
    bar.style.transition = `width ${durationMs}ms linear`;
    requestAnimationFrame(() => { bar.style.width = '100%'; });
  }
  function startLoop() {
    stopLoop();
    const tick = async () => {
      if (!state.auto) return;
      const name = PHASES[phaseIdx % PHASES.length];
      phaseIdx += 1;
      const dur = PHASE_DURATIONS[name] || 8000;
      animateProgress(dur);
      await runScenario(name);
      state.loopHandle = setTimeout(tick, dur);
    };
    tick();
  }

  function stopLoop() {
    if (state.loopHandle) { clearTimeout(state.loopHandle); state.loopHandle = null; }
  }

  async function runScenario(name) {
    try {
      const r = await fetch(`/api/scenario/${name}`, { method: 'POST' });
      if (!r.ok) return;
      const payload = await r.json();
      state.currentScenario = payload;
      // Update HUD + title
      if (payload.phase !== 'reset') {
        state.peopleServed += (payload.stats?.served || 0);
      }
      updateTitle(payload.title, payload.subtitle);
      updateHud(payload);
      highlightPill(name);
      renderLayers();
      // Fly in on the action so the arcs are actually visible
      if (payload.demand && payload.demand.length && state.map) {
        const cx = payload.demand.reduce((s, d) => s + d.lon, 0) / payload.demand.length;
        const cy = payload.demand.reduce((s, d) => s + d.lat, 0) / payload.demand.length;
        state.map.easeTo({ center: [cx, cy], zoom: 11.8, pitch: 58, duration: 1400 });
      } else if (state.map && name === 'reset') {
        state.map.easeTo({ center: NYC, zoom: 10.8, pitch: 58, duration: 1400 });
      }
    } catch (e) { /* silent */ }
  }

  // ─────────────────────────────────────────────────────────────────────
  // Auto-orbit camera
  // ─────────────────────────────────────────────────────────────────────
  function startOrbit() {
    if (state.orbitHandle) clearInterval(state.orbitHandle);
    state.orbitHandle = setInterval(() => {
      if (!state.map || state.orbitPaused) return;
      state.orbitAngle = (state.orbitAngle + 0.2) % 360;
      try { state.map.setBearing(state.orbitAngle); } catch (e) {}
    }, 120);
    // Allow orbit to resume after period of inactivity
    let idleTimer = null;
    const resumeAfterIdle = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { state.orbitPaused = false; }, 5000);
    };
    const container = document.getElementById(state.containerId);
    if (container) {
      ['mousedown','touchstart','wheel','keydown'].forEach(ev => container.addEventListener(ev, resumeAfterIdle));
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  // HUD + title + pills
  // ─────────────────────────────────────────────────────────────────────
  // Hover tooltip — tells the viewer what a column is
  function showTooltip(info) {
    let el = document.getElementById('sp-tip');
    if (!el) {
      el = document.createElement('div');
      el.id = 'sp-tip';
      el.className = 'hud-panel';
      el.style.cssText = 'position:fixed;padding:8px 12px;font-family:var(--mono);font-size:11px;pointer-events:none;z-index:950;max-width:240px;display:none;line-height:1.55';
      document.body.appendChild(el);
    }
    if (!info || !info.object) { el.style.display = 'none'; return; }
    const o = info.object;
    el.innerHTML = `
      <div style="color:var(--accent);font-weight:700;margin-bottom:4px">${(o.resource_type || o.type || 'resource').replace(/_/g,' ').toUpperCase()}</div>
      <div style="color:var(--t1);font-size:12px">${o.name || '(unnamed)'}</div>
      <div style="color:var(--t3);font-size:10px;margin-top:3px">${o.address || ''} ${o.borough ? '· ' + o.borough : ''}</div>
      <div style="color:var(--t2);font-size:10px;margin-top:4px">capacity: <b class="neon-cyan">${o.capacity || '—'}</b></div>
    `;
    el.style.left = (info.x + 14) + 'px';
    el.style.top  = (info.y + 14) + 'px';
    el.style.display = 'block';
  }

  // Legend card — explains what each visual element MEANS
  function injectLegend() {
    const el = document.createElement('div');
    el.className = 'hud-panel';
    el.id = 'sp-legend';
    el.style.cssText = 'position:fixed;bottom:16px;right:16px;padding:12px 14px;font-family:var(--mono);font-size:10.5px;letter-spacing:0.04em;line-height:1.65;z-index:700;max-width:260px';
    el.innerHTML = `
      <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em;margin-bottom:8px">LEGEND</div>
      <div style="display:grid;grid-template-columns:22px 1fr;gap:6px 10px;align-items:center">
        <div style="width:8px;height:14px;border-radius:2px;background:#00e5ff;box-shadow:0 0 10px #00e5ff;margin-left:6px"></div>
        <div>3D columns = <b class="neon-cyan">NYC resources</b> (height = capacity, color = type)</div>
        <div style="width:8px;height:8px;border-radius:50%;background:#ffcc33;box-shadow:0 0 10px #ffcc33;margin-left:6px"></div>
        <div>Dots = <b style="color:#ffcc33">people in need</b> (synthesized demand)</div>
        <div style="width:16px;height:2px;background:linear-gradient(90deg,#00e5ff,#b24bff);margin-left:2px"></div>
        <div>Arcs = <b class="neon-cyan">routing</b> person → nearest resource</div>
        <div style="width:10px;height:10px;border-radius:2px;background:radial-gradient(circle,#6640a0,transparent);margin-left:5px"></div>
        <div>Violet glow = <b style="color:#b24bff">service gaps</b> (low resource density)</div>
      </div>
      <div style="color:var(--t3);font-size:9px;margin-top:9px;letter-spacing:0.1em;border-top:1px solid rgba(255,255,255,0.08);padding-top:7px">
        Hover a cylinder for details. &nbsp;·&nbsp; Running locally, no cloud.
      </div>
    `;
    document.body.appendChild(el);
  }

  function injectOverlays() {
    injectLegend();
    // Phase title + progress bar
    const t = document.createElement('div');
    t.className = 'phase-title';
    t.id = 'sp-title';
    t.innerHTML = '<div class="main" style="font-size:15px;font-weight:600"></div><span class="sub"></span><div id="sp-progress" style="margin-top:10px;height:3px;background:rgba(255,255,255,0.1);border-radius:2px;overflow:hidden"><div id="sp-progress-bar" style="height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width 120ms linear;box-shadow:0 0 10px rgba(0,229,255,0.4)"></div></div>';
    document.body.appendChild(t);
    state.titleEl = t;

    // Stats HUD (top-right)
    const h = document.createElement('div');
    h.className = 'stats-hud';
    h.id = 'sp-hud';
    h.innerHTML = `
      <div class="label">PHASE</div><div class="value" id="sp-h-phase">—</div>
      <div class="label">SERVED (CUM)</div><div class="value" id="sp-h-served">0</div>
      <div class="label">ACTIVE SITES</div><div class="value mag" id="sp-h-sites">0</div>
      <div class="label">AVG RESPONSE</div><div class="value" id="sp-h-avg">— km</div>
      <div class="label">LAT. (ms)</div><div class="value" id="sp-h-latency">0</div>
      <div class="label">RESOURCES</div><div class="value" id="sp-h-res">0</div>
    `;
    document.body.appendChild(h);
    state.statsEl = h;

    // Pill controls (bottom-center)
    const p = document.createElement('div');
    p.className = 'scenario-pills';
    p.id = 'sp-pills';
    const mk = (id, label) => {
      const b = document.createElement('button');
      b.className = 'scenario-pill';
      b.textContent = label;
      b.dataset.id = id;
      b.onclick = () => {
        state.auto = false;
        if (id === 'auto') { state.auto = true; startLoop(); return; }
        runScenario(id);
      };
      state.pillEls[id] = b;
      return b;
    };
    p.appendChild(mk('cold_emergency', 'Cold Snap'));
    p.appendChild(mk('migrant_bus', 'Migrant Bus'));
    p.appendChild(mk('reset', 'Reset'));
    const autoPill = mk('auto', 'AUTO');
    autoPill.classList.add('active');
    p.appendChild(autoPill);
    document.body.appendChild(p);

    // Once the map initially loads some data, show the resources count
    const resPoll = setInterval(() => {
      if (state.allResources.length) {
        const e = document.getElementById('sp-h-res');
        if (e) e.textContent = state.allResources.length.toLocaleString();
        clearInterval(resPoll);
      }
    }, 300);
  }

  function updateTitle(title, subtitle) {
    if (!state.titleEl) return;
    if (!title) { state.titleEl.classList.remove('show'); return; }
    state.titleEl.querySelector('.main').textContent = title;
    state.titleEl.querySelector('.sub').textContent = subtitle || '';
    state.titleEl.classList.add('show');
  }

  function updateHud(payload) {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('sp-h-phase', (payload.phase || '—').replace('_', ' ').toUpperCase());
    set('sp-h-served', state.peopleServed.toLocaleString());
    set('sp-h-sites', (payload.sites?.length || 0));
    set('sp-h-avg', `${payload.stats?.avg_km?.toFixed?.(2) || 0} km`);
    set('sp-h-latency', `${payload.stats?.elapsed_ms || 0}`);
  }

  function highlightPill(name) {
    Object.entries(state.pillEls).forEach(([id, b]) => {
      if (id === name || (id === 'auto' && state.auto)) b.classList.add('active');
      else if (id !== 'auto') b.classList.remove('active');
    });
  }

  // ─────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────
  window.ScenarioPlayer = {
    init,
    startLoop,
    stopLoop,
    runScenario,
    setAuto(on) { state.auto = !!on; if (on) startLoop(); else stopLoop(); },
  };
})();
