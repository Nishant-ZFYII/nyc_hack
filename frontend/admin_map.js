/* ═══════════════════════════════════════════════════════════════════════
   admin_map.js — Admin Live Ops 3D map
   ═════════════════════════════════════════════════════════════════════════
   Self-contained. Mount a <div id="adminMapEl"> and call:
       AdminOpsMap.init('adminMapEl')
       AdminOpsMap.startPolling(interval_ms = 4000)   // live refresh

   Talks to:
       GET /api/admin/ops_snapshot   one-shot dashboard payload
   ═════════════════════════════════════════════════════════════════════════ */
(function () {
  const NYC = [-73.95, 40.74];

  // Site color ramp (cool for low load, hot for overload)
  function loadColor(load, cap) {
    const r = Math.min(1, (load || 0) / Math.max(1, cap));
    // 0.0 → cyan (0,229,255) ; 0.5 → magenta (178,75,255) ; 1.0 → red (255,85,119)
    if (r < 0.5) {
      const t = r / 0.5;
      return [
        Math.round(0   + t * (178 - 0)),
        Math.round(229 + t * (75  - 229)),
        Math.round(255 + t * (255 - 255)),
        230,
      ];
    } else {
      const t = (r - 0.5) / 0.5;
      return [
        Math.round(178 + t * (255 - 178)),
        Math.round(75  + t * (85  - 75)),
        Math.round(255 + t * (119 - 255)),
        240,
      ];
    }
  }

  const state = {
    map: null,
    overlay: null,
    data: { sites: [], cases: [], arcs: [], stats: {} },
    pollHandle: null,
    pulseT: 0,
    orbitAngle: -10,
    orbitHandle: null,
    orbitPaused: false,
    containerId: null,
    lighting: null,
  };

  function init(containerId) {
    state.containerId = containerId;
    if (state.map) return;
    state.map = new maplibregl.Map({
      container: containerId,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: NYC,
      zoom: 10.4,
      pitch: 55,
      bearing: state.orbitAngle,
      antialias: true,
    });
    state.map.addControl(new maplibregl.NavigationControl(), 'top-left');
    state.map.on('load', () => {
      state.map.resize();
      refresh();
    });
    state.map.on('mousedown', () => { state.orbitPaused = true; });

    try {
      state.lighting = new deck.LightingEffect({
        ambient: new deck.AmbientLight({ color: [255,255,255], intensity: 1.0 }),
        dir: new deck.DirectionalLight({ color: [255,255,255], intensity: 1.4, direction: [-2,-8,-1] }),
      });
    } catch (e) { state.lighting = null; }

    // Pulse driver — keeps ScatterplotLayer re-rendering so dots breathe
    setInterval(() => {
      state.pulseT += 0.12;
      renderLayers();
    }, 90);

    // Auto-orbit
    state.orbitHandle = setInterval(() => {
      if (!state.map || state.orbitPaused) return;
      state.orbitAngle = (state.orbitAngle + 0.12) % 360;
      try { state.map.setBearing(state.orbitAngle); } catch (e) {}
    }, 130);
    // Resume orbit after idle
    let idleT = null;
    const cont = document.getElementById(containerId);
    if (cont) ['mousedown','wheel','touchstart','keydown'].forEach(ev => cont.addEventListener(ev, () => {
      if (idleT) clearTimeout(idleT);
      idleT = setTimeout(() => { state.orbitPaused = false; }, 6000);
    }));

    injectHud();
  }

  async function refresh() {
    try {
      const r = await fetch('/api/admin/ops_snapshot');
      if (!r.ok) return;
      state.data = await r.json();
      updateHud(state.data.stats);
      renderLayers();
    } catch (e) { /* silent */ }
  }

  function startPolling(ms = 4000) {
    if (state.pollHandle) clearInterval(state.pollHandle);
    state.pollHandle = setInterval(refresh, ms);
    refresh();
  }
  function stopPolling() { if (state.pollHandle) { clearInterval(state.pollHandle); state.pollHandle = null; } }

  function renderLayers() {
    if (!state.map || !state.data) return;
    const { sites, cases, arcs } = state.data;
    const layers = [];

    // Heatmap — open-case density
    if (cases && cases.length) {
      layers.push(new deck.HeatmapLayer({
        id: 'admin-heat',
        data: cases,
        getPosition: d => [d.lon, d.lat],
        getWeight: d => d.status === 'open' ? (d.urgency === 'critical' ? 5 : d.urgency === 'high' ? 3 : 1) : 0,
        radiusPixels: 90,
        intensity: 1.2,
        threshold: 0.04,
        colorRange: [[20,0,80,0],[80,30,150,140],[180,60,200,200],[255,100,180,230],[255,85,119,250]],
      }));
    }

    // Sites as columns — load drives height + color
    if (sites && sites.length) {
      layers.push(new deck.ColumnLayer({
        id: 'admin-sites',
        data: sites,
        diskResolution: 18,
        radius: 80,
        extruded: true,
        pickable: false,
        getPosition: d => [d.lon, d.lat],
        getElevation: d => 60 + Math.log1p((d.load || 0) + 1) * 200 + Math.log1p(d.cap || 30) * 30,
        getFillColor: d => loadColor(d.load || 0, d.cap || 30),
        material: { ambient: 0.5, diffuse: 0.8, shininess: 80, specularColor: [200, 220, 255] },
      }));
    }

    // Case pulses — ScatterplotLayer with time-varying radius
    if (cases && cases.length) {
      const pulse = 0.6 + 0.4 * Math.sin(state.pulseT);
      layers.push(new deck.ScatterplotLayer({
        id: 'admin-cases',
        data: cases,
        getPosition: d => [d.lon, d.lat],
        getRadius: d => {
          const base = d.urgency === 'critical' ? 160 : d.urgency === 'high' ? 110 : 70;
          return base * (0.8 + 0.35 * pulse);
        },
        radiusMinPixels: 5,
        radiusMaxPixels: 18,
        getFillColor: d => {
          const c = d.color || [150,170,200,180];
          const fade = d.status === 'resolved' ? 0.4 : 1.0;
          return [c[0], c[1], c[2], Math.round((c[3] || 200) * fade)];
        },
        stroked: true,
        lineWidthMinPixels: 1.2,
        getLineColor: [255, 255, 255, 180],
        updateTriggers: { getRadius: state.pulseT },
      }));
    }

    // Case → site arcs
    if (arcs && arcs.length) {
      layers.push(new deck.ArcLayer({
        id: 'admin-arcs',
        data: arcs,
        getSourcePosition: d => d.from,
        getTargetPosition: d => d.to,
        getSourceColor: d => d.color || [0,229,255,200],
        getTargetColor: d => {
          const c = d.color || [0,229,255,200];
          return [Math.min(255,c[0]+40), Math.min(255,c[1]+40), Math.min(255,c[2]+40), 255];
        },
        getWidth: 2.5,
        widthMinPixels: 1.5,
        greatCircle: false,
      }));
    }

    const props = { layers };
    if (state.lighting) props.effects = [state.lighting];

    if (state.overlay) state.overlay.setProps(props);
    else { state.overlay = new deck.MapboxOverlay(props); state.map.addControl(state.overlay); }
  }

  // HUD — bottom-left for admin (doesn't collide with user-portal stats-hud)
  function injectHud() {
    const h = document.createElement('div');
    h.className = 'hud-panel';
    h.id = 'admin-hud';
    h.style.cssText = 'position:fixed;top:16px;left:50%;transform:translateX(-50%);padding:12px 22px;font-family:var(--mono);font-size:12px;letter-spacing:0.1em;display:flex;gap:28px;align-items:center;z-index:700';
    h.innerHTML = `
      <div>
        <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em">OPEN CASES</div>
        <div id="adm-open" class="ticker" style="font-size:20px">—</div>
      </div>
      <div>
        <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em">CRITICAL</div>
        <div id="adm-crit" class="neon-red" style="font-size:20px;font-weight:600">—</div>
      </div>
      <div>
        <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em">RESOLVED</div>
        <div id="adm-done" class="neon-green" style="font-size:20px;font-weight:600">—</div>
      </div>
      <div>
        <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em">SITES TRACKED</div>
        <div id="adm-sites" class="ticker" style="font-size:20px">—</div>
      </div>
    `;
    document.body.appendChild(h);
  }

  function updateHud(stats) {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('adm-open', (stats?.open ?? 0));
    set('adm-crit', (stats?.critical ?? 0));
    set('adm-done', (stats?.resolved ?? 0));
    set('adm-sites', (state.data.sites?.length || 0).toLocaleString());
  }

  window.AdminOpsMap = { init, refresh, startPolling, stopPolling };
})();
