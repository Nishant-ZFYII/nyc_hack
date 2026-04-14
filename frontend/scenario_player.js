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
    auto: false,                 // default: no scenario cycle on load
    loopHandle: null,
    orbitHandle: null,
    orbitAngle: -20,
    orbitPaused: false,          // true = user is interacting, orbit stops
    statsEl: null,
    titleEl: null,
    pillEls: {},
    tickStart: Date.now(),
    peopleServed: 0,
    // Particle animation
    particleT: 0,                // 0..1 rolling
    particleRafHandle: null,
    // Cinematic intro
    introComplete: false,
    // Type filter (dropdown) — 'all' or a resource_type string
    typeFilter: 'all',
    // Master autoplay switch — default OFF. Controls BOTH the scenario
    // cycle AND the camera orbit so one pill gives predictable behaviour.
    autoplayOn: false,
  };

  const PHASES = ['cold_emergency', 'migrant_bus', 'citywide_storm', 'reset'];
  const PHASE_DURATIONS = {
    cold_emergency: 9000,
    migrant_bus: 9000,
    citywide_storm: 11000,   // the money shot — give it room
    reset: 2500,
  };

  // Each scenario operates on a primary resource type. When the user fires
  // a scenario we sync the dropdown filter to that type so the viewer sees
  // exactly what's being computed (no blank-screen confusion). null = "all".
  const SCENARIO_PRIMARY_TYPE = {
    cold_emergency: 'shelter',
    migrant_bus:    'community_center',
    citywide_storm: null,            // storm intentionally touches all types
    reset:          null,
  };

  // Plain-English caption shown BIG on mute for each phase
  const PHASE_CAPTIONS = {
    cold_emergency: {
      main: 'COLD SNAP · CODE BLUE',
      sub:  '2,500 PEOPLE CITYWIDE — EVERY SHELTER + WARMING CENTER ACTIVATES',
    },
    migrant_bus: {
      main: 'MIGRANT BUS ARRIVAL',
      sub:  '500 PEOPLE DISPERSED TO INTAKE SITES ACROSS ALL 5 BOROUGHS',
    },
    citywide_storm: {
      main: 'CITYWIDE FLOW',
      sub:  '4,000 CONCURRENT ROUTINGS · ALL 5 BOROUGHS · UNDER ONE SECOND',
    },
    reset: { main: '', sub: '' },
  };

  // ─────────────────────────────────────────────────────────────────────
  // Map init
  // ─────────────────────────────────────────────────────────────────────
  function init(containerId) {
    state.containerId = containerId;
    const el = document.getElementById(containerId);
    if (!el) { console.warn('[ScenarioPlayer] container missing:', containerId); return; }
    if (state.map) return;

    // CINEMATIC INTRO: start tight over Manhattan at steep angle,
    // then ease out to the full NYC overview as the skyline reveals.
    state.map = new maplibregl.Map({
      container: containerId,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [-73.9857, 40.7580],   // Midtown Manhattan for the intro
      zoom: 14.0,
      pitch: 62,
      bearing: -8,
      antialias: true,
    });
    state.map.addControl(new maplibregl.NavigationControl(), 'top-left');
    state.map.on('load', async () => {
      state.map.resize();
      // Only fetch the resource layer — no more PLUTO pillars, no vulnerability
      await loadResources();
      renderLayers();
      // Intro pull-back — short and cinematic, no auto-loop afterwards
      setTimeout(() => {
        state.map.easeTo({
          center: NYC, zoom: 10.7, pitch: 52, bearing: state.orbitAngle,
          duration: 3400, easing: t => t * (2 - t),
        });
        setTimeout(() => { state.introComplete = true; }, 3500);
      }, 400);
    });
    state.map.on('mousedown', () => { state.orbitPaused = true; state.introComplete = true; });
    state.map.on('touchstart', () => { state.orbitPaused = true; state.introComplete = true; });

    startParticleLoop();

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

  // ─────────────────────────────────────────────────────────────────────
  // Particle animation loop — drives the flowing dots on every arc
  // ─────────────────────────────────────────────────────────────────────
  function startParticleLoop() {
    if (state.particleRafHandle) cancelAnimationFrame(state.particleRafHandle);
    let last = performance.now();
    function frame(now) {
      const dt = (now - last) / 1000;
      last = now;
      // Slower cycle (~5.5 s full loop) so individual trails have time to read
      state.particleT = (state.particleT + dt * 0.18) % 1;
      // Only repaint particle layer (not the expensive 28K building layer)
      if (state.overlay && state.currentScenario && state.currentScenario.arcs?.length) {
        state.overlay.setProps({ layers: buildLayers() });
      }
      state.particleRafHandle = requestAnimationFrame(frame);
    }
    state.particleRafHandle = requestAnimationFrame(frame);
  }

  // Haversine distance in METERS — used so each trip's apex scales with arc
  // length (long arcs rise high, short arcs stay low).
  function _distanceMeters(a, b) {
    const R = 6371000;
    const toRad = v => v * Math.PI / 180;
    const dLat = toRad(b[1] - a[1]);
    const dLon = toRad(b[0] - a[0]);
    const lat1 = toRad(a[1]);
    const lat2 = toRad(b[1]);
    const h = Math.sin(dLat/2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon/2) ** 2;
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  // Convert arcs → trip paths for deck.gl TripsLayer. Each arc becomes a
  // 40-sample 3D parabola (lon, lat, altitude) with monotonic timestamps.
  // TripsLayer then animates a trail along that exact geometry, so the
  // visible line AND the animated highlight share one source — no possible
  // drift between "arc" and "particle".
  function arcsToTrips(arcs) {
    const N = 40;
    const trips = [];
    for (let i = 0; i < arcs.length; i++) {
      const a = arcs[i];
      if (!a.from || !a.to) continue;
      const dm = _distanceMeters(a.from, a.to);
      const apex = Math.max(180, dm * 0.45);
      const path = [];
      const timestamps = [];
      // stagger each arc's start offset so trips don't all fire in lock-step
      const start = (i * 0.071) % 1;
      for (let k = 0; k < N; k++) {
        const t = k / (N - 1);
        const lon = a.from[0] + (a.to[0] - a.from[0]) * t;
        const lat = a.from[1] + (a.to[1] - a.from[1]) * t;
        const alt = Math.sin(t * Math.PI) * apex;
        path.push([lon, lat, alt]);
        timestamps.push(start + t * 0.4); // each trip spans 0.4 of the loop
      }
      trips.push({
        path,
        timestamps,
        color: a.color || [0, 229, 255, 220],
      });
    }
    return trips;
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
  function buildLayers() {
    const layers = [];
    // Design note (Tufte): the dark-matter basemap already provides the
    // "this is NYC" context via its street grid + 2D building footprints.
    // Our data job is ONLY to show the resources + the flow. No extra
    // ambient layers (no PLUTO pillars, no vulnerability heatmap).

    // All ~2.5K resources as glowing neon "beacon" columns — hexagonal
    // footprint to distinguish them from the square buildings underneath.
    // Filter by the dropdown selection when user has picked a specific type.
    const filteredResources = (state.typeFilter === 'all')
      ? state.allResources
      : state.allResources.filter(r => r.resource_type === state.typeFilter);

    // When a scenario is active, compute the demand bounding box so we can
    // dim beacons outside the action zone. This makes it obvious WHERE the
    // movement is happening and stops e.g. Brooklyn shelters from looking
    // "on" while Cold Emergency only routes in the Bronx.
    let scenarioBbox = null;
    const _sc = state.currentScenario;
    if (_sc && _sc.demand && _sc.demand.length) {
      const lons = _sc.demand.map(d => d.lon);
      const lats = _sc.demand.map(d => d.lat);
      const pad = 0.02;
      scenarioBbox = {
        west: Math.min(...lons) - pad,
        east: Math.max(...lons) + pad,
        south: Math.min(...lats) - pad,
        north: Math.max(...lats) + pad,
      };
    }
    const inBbox = d => !scenarioBbox
      || (d.lon >= scenarioBbox.west && d.lon <= scenarioBbox.east
          && d.lat >= scenarioBbox.south && d.lat <= scenarioBbox.north);

    if (filteredResources.length) {
      layers.push(new deck.ColumnLayer({
        id: 'sp-city',
        data: filteredResources,
        diskResolution: 6,          // hex — reads as infrastructure
        radius: 38,
        extruded: true,
        pickable: true,
        getPosition: d => [d.lon, d.lat],
        getFillColor: d => {
          const c = TYPE_COLORS_SP[d.resource_type] || COLOR_DEFAULT_SP;
          // Dim to ~20% opacity + desaturated if outside the active scenario
          if (!inBbox(d)) return [Math.round(c[0] * 0.4), Math.round(c[1] * 0.4), Math.round(c[2] * 0.4), 70];
          return [c[0], c[1], c[2], 225];
        },
        // Heights between 60 m and ~450 m — shrink outside-scenario beacons too
        getElevation: d => {
          const base = 120 + Math.log1p(d.capacity || 20) * 110;
          return inBbox(d) ? base : base * 0.35;
        },
        material: { ambient: 0.55, diffuse: 0.8, shininess: 90, specularColor: [200, 230, 255] },
        onHover: info => showTooltip(info),
        updateTriggers: {
          getFillColor: [state.typeFilter, scenarioBbox ? JSON.stringify(scenarioBbox) : null],
          getElevation: [state.typeFilter, scenarioBbox ? JSON.stringify(scenarioBbox) : null],
        },
      }));
    }

    // Scenario-specific layers. When the dropdown has a specific type
    // selected, also filter the scenario's sites and arcs so everything
    // the viewer sees is coherent with their choice.
    const sc = state.currentScenario;
    if (sc && sc.arcs && sc.arcs.length) {
      const filterSites = state.typeFilter === 'all'
        ? (sc.sites || [])
        : (sc.sites || []).filter(s => s.type === state.typeFilter);
      // Build a fast lookup of the ids we care about
      const filterSiteIds = new Set(filterSites.map(s => s.id));
      const filterArcs = state.typeFilter === 'all'
        ? sc.arcs
        : sc.arcs.filter(a => !a.to_id || filterSiteIds.has(a.to_id));

      // Bright rim around each active site (highlight where demand lands)
      if (filterSites.length) {
        layers.push(new deck.ColumnLayer({
          id: 'sp-sites',
          data: filterSites,
          diskResolution: 18,
          radius: 95,
          extruded: true,
          pickable: false,
          getPosition: d => [d.lon, d.lat],
          getFillColor: d => {
            const c = TYPE_COLORS_SP[d.type] || COLOR_DEFAULT_SP;
            const loadFrac = Math.min(1, (d.used || 0) / Math.max(1, d.cap));
            const saturate = 0.75 + 0.25 * loadFrac;
            return [
              Math.min(255, Math.floor(c[0] * saturate)),
              Math.min(255, Math.floor(c[1] * saturate)),
              Math.min(255, Math.floor(c[2] * saturate)),
              245,
            ];
          },
          getElevation: d => Math.log1p((d.used || 1) + (d.cap || 30)) * 150 + 200,
          material: { ambient: 0.7, diffuse: 0.95, shininess: 160, specularColor: [255, 255, 255] },
        }));
      }
      // FLOW via deck.gl TripsLayer — one geometry source for both the
      // path (faded line) and the moving trail (animated head). Can't
      // drift apart because they're literally the same vertex data.
      if (deck.TripsLayer) {
        const trips = arcsToTrips(filterArcs);
        layers.push(new deck.TripsLayer({
          id: 'sp-trips',
          data: trips,
          getPath: d => d.path,
          getTimestamps: d => d.timestamps,
          getColor: d => d.color,
          getWidth: 4,
          widthMinPixels: 2.5,
          opacity: 0.92,
          trailLength: 0.32,
          currentTime: state.particleT * 1.6,
          capRounded: true,
          jointRounded: true,
          fadeTrail: true,
          shadowEnabled: false,
        }));
      } else {
        // Fallback: plain static arcs if TripsLayer is missing in this bundle
        layers.push(new deck.ArcLayer({
          id: 'sp-arcs-fallback',
          data: filterArcs,
          getSourcePosition: d => d.from,
          getTargetPosition: d => d.to,
          getSourceColor: d => d.color || [0, 229, 255, 200],
          getTargetColor: d => d.color || [0, 229, 255, 200],
          getWidth: 2,
          getHeight: 0.45,
        }));
      }
    }

    return layers;
  }

  function renderLayers() {
    if (!state.map) return;
    const props = { layers: buildLayers() };
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
      // Sync the type filter so the dropdown reflects the scenario's domain.
      // Cold Emergency → shelter, Migrant Bus → community_center, Storm → all.
      const targetType = SCENARIO_PRIMARY_TYPE.hasOwnProperty(name)
        ? SCENARIO_PRIMARY_TYPE[name] : null;
      const newFilter = targetType || 'all';
      if (state.typeFilter !== newFilter) {
        state.typeFilter = newFilter;
        const sel = document.getElementById('sp-filter-sel');
        if (sel) sel.value = newFilter;
      }
      updateTitle(payload.title, payload.subtitle);
      updateHud(payload);
      highlightPill(name);
      renderLayers();
      // Big mute-readable caption for the phase
      const cap = PHASE_CAPTIONS[name] || { main: payload.title, sub: payload.subtitle };
      if (cap.main) {
        showCaption(cap.main, cap.sub);
      } else {
        showCaption('', '');
      }
      // Fly in on the action so the arcs are actually visible
      if (payload.demand && payload.demand.length && state.map) {
        const cx = payload.demand.reduce((s, d) => s + d.lon, 0) / payload.demand.length;
        const cy = payload.demand.reduce((s, d) => s + d.lat, 0) / payload.demand.length;
        // Citywide storm needs to stay wide; other phases zoom in
        const zoom = name === 'citywide_storm' ? 10.4 : 11.8;
        state.map.easeTo({ center: [cx, cy], zoom, pitch: 56, duration: 1400 });
      } else if (state.map && name === 'reset') {
        state.map.easeTo({ center: NYC, zoom: 10.8, pitch: 56, duration: 1400 });
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
      if (!state.introComplete) return;   // let cinematic intro finish
      if (!state.autoplayOn) return;      // only rotate when user wants autoplay
      state.orbitAngle = (state.orbitAngle + 0.55) % 360;
      try { state.map.setBearing(state.orbitAngle); } catch (e) {}
    }, 90);
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
  // Hover tooltip — tells the viewer what a column is. Auto-hides 1.5 s
  // after the last hover so stale tooltips don't linger.
  let _tipHideT = null;
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
    // Build with DOM nodes, not innerHTML templates — guaranteed no raw `${}`
    // ever appears in the output.
    const typeName = String((o.resource_type || o.type || 'resource')).replace(/_/g, ' ').toUpperCase();
    el.innerHTML = '';
    const row1 = document.createElement('div');
    row1.style.cssText = 'color:var(--accent);font-weight:700;margin-bottom:4px';
    row1.textContent = typeName;
    const row2 = document.createElement('div');
    row2.style.cssText = 'color:var(--t1);font-size:12px';
    row2.textContent = o.name || '(unnamed)';
    const row3 = document.createElement('div');
    row3.style.cssText = 'color:var(--t3);font-size:10px;margin-top:3px';
    row3.textContent = (o.address || '') + (o.borough ? ' · ' + o.borough : '');
    const row4 = document.createElement('div');
    row4.style.cssText = 'color:var(--t2);font-size:10px;margin-top:4px';
    const capLabel = document.createElement('span'); capLabel.textContent = 'capacity: ';
    const capVal = document.createElement('b');
    capVal.className = 'neon-cyan';
    capVal.textContent = (o.capacity || o.cap || '—');
    row4.appendChild(capLabel); row4.appendChild(capVal);
    el.appendChild(row1); el.appendChild(row2); el.appendChild(row3); el.appendChild(row4);
    el.style.left = (info.x + 14) + 'px';
    el.style.top  = (info.y + 14) + 'px';
    el.style.display = 'block';
    // Auto-hide after inactivity
    if (_tipHideT) clearTimeout(_tipHideT);
    _tipHideT = setTimeout(() => { if (el) el.style.display = 'none'; }, 1800);
  }

  // Legend card — explains what each visual element MEANS
  function injectLegend() {
    const el = document.createElement('div');
    el.className = 'hud-panel';
    el.id = 'sp-legend';
    el.style.cssText = 'position:fixed;bottom:16px;right:16px;padding:12px 14px;font-family:var(--mono);font-size:10.5px;letter-spacing:0.04em;line-height:1.65;z-index:700;max-width:260px';
    el.innerHTML = `
      <div style="color:var(--t3);font-size:9px;letter-spacing:0.18em;margin-bottom:8px">LEGEND</div>
      <div style="display:grid;grid-template-columns:24px 1fr;gap:8px 10px;align-items:center">
        <div style="width:5px;height:12px;border-radius:1px;background:#00e5ff;opacity:0.55;margin-left:9px"></div>
        <div><b style="color:#e8f0ff">Thin column</b> = resource exists (<span style="color:var(--t3)">inventory</span>)</div>
        <div style="width:10px;height:18px;border-radius:2px;background:#00e5ff;box-shadow:0 0 14px #00e5ff;margin-left:6px"></div>
        <div><b class="neon-cyan">Thick tall column</b> = active in scenario (height = load)</div>
        <div style="width:20px;height:3px;background:linear-gradient(90deg,transparent,#00e5ff,#b24bff);margin-left:0"></div>
        <div><b class="neon-cyan">Neon trail</b> = person being routed to a resource</div>
        <div style="width:8px;height:8px;border-radius:50%;background:#444a5e;margin-left:7px"></div>
        <div><b style="color:var(--t2)">Dimmed column</b> = exists but not in this scenario</div>
      </div>
      <div style="color:var(--t3);font-size:9px;margin-top:9px;letter-spacing:0.1em;border-top:1px solid rgba(255,255,255,0.08);padding-top:7px">
        Hover any column for details &nbsp;·&nbsp; Running locally, no cloud
      </div>
    `;
    document.body.appendChild(el);
  }

  // Big centered caption that describes the phase in plain English —
  // this is the mute-readable narration.
  function injectCaption() {
    const c = document.createElement('div');
    c.id = 'sp-caption';
    c.style.cssText = `
      position:fixed; left:50%; top:26%; transform:translate(-50%,0);
      padding:22px 42px; border-radius:14px;
      background:rgba(10,14,22,0.78); backdrop-filter:blur(22px);
      -webkit-backdrop-filter:blur(22px);
      border:1px solid rgba(0,229,255,0.35);
      box-shadow:0 0 48px rgba(0,229,255,0.25);
      text-align:center; font-family:var(--mono);
      color:var(--t1); z-index:850;
      pointer-events:none; min-width:320px; max-width:720px;
      opacity:0; transition:opacity 500ms ease;
    `;
    c.innerHTML = `
      <div id="sp-caption-main" style="font-size:22px;font-weight:700;color:var(--accent);letter-spacing:0.28em;text-shadow:0 0 16px rgba(0,229,255,0.55);line-height:1.2">—</div>
      <div id="sp-caption-sub"  style="font-size:12px;color:var(--t2);letter-spacing:0.18em;margin-top:10px">—</div>
    `;
    document.body.appendChild(c);
    return c;
  }

  function showCaption(main, sub) {
    let c = document.getElementById('sp-caption');
    if (!c) c = injectCaption();
    if (!main) { c.style.opacity = '0'; return; }
    document.getElementById('sp-caption-main').textContent = main;
    document.getElementById('sp-caption-sub').textContent = sub || '';
    c.style.opacity = '1';
  }

  // Type-filter dropdown — lets the viewer hide schools/childcare noise
  // or isolate one category (shelters, food, hospitals, etc).
  function injectTypeFilter() {
    if (document.getElementById('sp-filter')) return;
    const wrap = document.createElement('div');
    wrap.id = 'sp-filter';
    wrap.className = 'hud-panel';
    wrap.style.cssText = `
      position:fixed; top:16px; left:50%; transform:translateX(-50%);
      padding:8px 14px; font-family:var(--mono); font-size:11px;
      letter-spacing:0.08em; z-index:700;
      display:flex; align-items:center; gap:10px;
    `;
    wrap.innerHTML = `
      <span style="color:var(--t3);font-size:9px;letter-spacing:0.18em">SHOW</span>
      <select id="sp-filter-sel" style="
        background:rgba(7,8,12,0.85); color:var(--accent);
        border:1px solid rgba(0,229,255,0.3); border-radius:6px;
        padding:6px 10px; font-family:var(--mono); font-size:11px;
        cursor:pointer; min-width:180px;
        text-transform:uppercase; letter-spacing:0.08em;
        outline:none;
      "></select>
    `;
    document.body.appendChild(wrap);

    const sel = document.getElementById('sp-filter-sel');
    // Populate options from the resource types we actually have
    function refreshOptions() {
      const types = Array.from(new Set(state.allResources.map(r => r.resource_type))).sort();
      sel.innerHTML = '';
      const optAll = document.createElement('option');
      optAll.value = 'all';
      optAll.textContent = `All types (${state.allResources.length.toLocaleString()})`;
      sel.appendChild(optAll);
      for (const t of types) {
        if (!t) continue;
        const count = state.allResources.filter(r => r.resource_type === t).length;
        const o = document.createElement('option');
        o.value = t;
        o.textContent = `${t.replace(/_/g, ' ')} (${count})`;
        sel.appendChild(o);
      }
      sel.value = state.typeFilter || 'all';
    }
    refreshOptions();
    // Retry once a second until resources load
    const ret = setInterval(() => {
      if (state.allResources.length) { refreshOptions(); clearInterval(ret); }
    }, 500);

    sel.onchange = () => {
      state.typeFilter = sel.value;
      renderLayers();
    };
  }

  function injectOverlays() {
    injectLegend();
    injectCaption();
    injectTypeFilter();
    // Initial orientation caption during the intro pull-back
    showCaption('NYC SOCIAL SERVICES', 'LIVE OPS · 7,759 RESOURCES · RUNNING LOCALLY');
    setTimeout(() => showCaption('', ''), 4200);
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
        if (id === 'auto') {
          // Master switch — scenario cycle + camera orbit
          state.autoplayOn = !state.autoplayOn;
          state.auto = state.autoplayOn;
          if (state.autoplayOn) {
            startLoop();
            state.orbitPaused = false;
            b.classList.add('active');
          } else {
            stopLoop();
            state.orbitPaused = true;
            b.classList.remove('active');
          }
          return;
        }
        // Any specific scenario pill click disables autoplay + orbit
        state.autoplayOn = false;
        state.auto = false;
        state.orbitPaused = true;
        stopLoop();
        const autoBtn = state.pillEls['auto'];
        if (autoBtn) autoBtn.classList.remove('active');
        runScenario(id);
      };
      state.pillEls[id] = b;
      return b;
    };
    p.appendChild(mk('cold_emergency', 'Cold Snap'));
    p.appendChild(mk('migrant_bus', 'Migrant Bus'));
    p.appendChild(mk('citywide_storm', 'Citywide'));
    p.appendChild(mk('reset', 'Reset'));
    const autoPill = mk('auto', 'AUTO');
    // starts inactive — default UX is calm/static; user opts in
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

  // Smoothly lerp an on-screen integer from its current value to `target`
  // over `durationMs`. Reads the element's current text, eases with
  // easeOutCubic, writes toLocaleString().
  function animateNumber(el, target, durationMs) {
    if (!el) return;
    const cur = parseInt((el.textContent || '0').replace(/[^\d-]/g, ''), 10) || 0;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / durationMs);
      const e = 1 - Math.pow(1 - p, 3);
      const v = Math.round(cur + (target - cur) * e);
      el.textContent = v.toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function updateHud(payload) {
    const byId = id => document.getElementById(id);
    if (byId('sp-h-phase')) byId('sp-h-phase').textContent = (payload.phase || '—').replace(/_/g, ' ').toUpperCase();
    animateNumber(byId('sp-h-served'), state.peopleServed, 1400);
    animateNumber(byId('sp-h-sites'), payload.sites?.length || 0, 900);
    const avg = payload.stats?.avg_km?.toFixed?.(2) || '0';
    if (byId('sp-h-avg')) byId('sp-h-avg').textContent = `${avg} km`;
    animateNumber(byId('sp-h-latency'), payload.stats?.elapsed_ms || 0, 900);
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
