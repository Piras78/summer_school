"use strict";

// ═══════════════════════════════════════════════════════════════
//  Cryo-EM Viewer
// ═══════════════════════════════════════════════════════════════

// ── Plasma colormap (matches matplotlib) ──────────────────────
function plasmaColor(t) {
  const stops = [
    [13,8,135],[84,2,163],[139,10,165],[185,50,137],
    [219,92,104],[244,136,73],[252,253,191],
  ];
  const n   = stops.length - 1;
  const pos = Math.max(0, Math.min(1, t)) * n;
  const i   = Math.min(Math.floor(pos), n - 1);
  const f   = pos - i;
  const a   = stops[i], b = stops[i + 1];
  return [(a[0]+(b[0]-a[0])*f)/255, (a[1]+(b[1]-a[1])*f)/255, (a[2]+(b[2]-a[2])*f)/255];
}

// ═══════════════════════════════════════════════════════════════
//  Volume3DRenderer  –  Three.js point-cloud viewer
// ═══════════════════════════════════════════════════════════════
class Volume3DRenderer {
  constructor(container) {
    this.container = container;
    this._raf = null;
    this._points = null;
    this._orbit = { theta: 0.5, phi: 1.1, radius: 200, dragging: false, lx: 0, ly: 0 };

    this._setup();
    this._bindEvents();
    this._animate();
  }

  _setup() {
    const w = this.container.clientWidth  || 600;
    const h = this.container.clientHeight || 400;

    this._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this._renderer.setClearColor(0x060810, 1);

    this._canvas = this._renderer.domElement;
    this._canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;";
    this.container.appendChild(this._canvas);

    this._scene = new THREE.Scene();

    this._camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000);

    // Bounding box wireframe (128³ volume outline)
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(128, 128, 128)),
      new THREE.LineBasicMaterial({ color: 0x2a3258, transparent: true, opacity: 0.5 })
    );
    this._scene.add(box);

    // Subtle grid on the floor
    const grid = new THREE.GridHelper(128, 8, 0x1a2040, 0x1a2040);
    grid.position.y = -64;
    this._scene.add(grid);
  }

  _bindEvents() {
    const el = this._canvas;

    el.addEventListener("mousedown", e => {
      this._orbit.dragging = true;
      this._orbit.lx = e.clientX;
      this._orbit.ly = e.clientY;
      e.stopPropagation();
    });
    document.addEventListener("mousemove", e => {
      if (!this._orbit.dragging) return;
      const dx = e.clientX - this._orbit.lx;
      const dy = e.clientY - this._orbit.ly;
      this._orbit.theta -= dx * 0.007;
      this._orbit.phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._orbit.phi + dy * 0.007));
      this._orbit.lx = e.clientX;
      this._orbit.ly = e.clientY;
    });
    document.addEventListener("mouseup",   () => { this._orbit.dragging = false; });

    el.addEventListener("wheel", e => {
      e.preventDefault();
      e.stopPropagation();
      this._orbit.radius = Math.max(60, Math.min(500, this._orbit.radius + e.deltaY * 0.25));
    }, { passive: false });
  }

  loadPointCloud(data) {
    if (this._points) { this._scene.remove(this._points); this._points.geometry.dispose(); this._points.material.dispose(); }

    const n = data.x.length;
    if (!n) return;

    const pos   = new Float32Array(n * 3);
    const color = new Float32Array(n * 3);

    for (let i = 0; i < n; i++) {
      pos[i*3]   = data.x[i];
      pos[i*3+1] = data.z[i];   // swap Y/Z so "up" is Y in Three.js
      pos[i*3+2] = data.y[i];
      const [r, g, b] = plasmaColor(data.intensity[i]);
      color[i*3] = r; color[i*3+1] = g; color[i*3+2] = b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos,   3));
    geo.setAttribute("color",    new THREE.BufferAttribute(color, 3));

    const mat = new THREE.PointsMaterial({ size: 3.0, vertexColors: true, transparent: true, opacity: 0.9, sizeAttenuation: true });
    this._points = new THREE.Points(geo, mat);
    this._scene.add(this._points);
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
  }

  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    if (!this._orbit.dragging) this._orbit.theta += 0.004;
    const { theta, phi, radius } = this._orbit;
    this._camera.position.set(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
    this._camera.lookAt(0, 0, 0);
    this._renderer.render(this._scene, this._camera);
  }

  dispose() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._points) { this._points.geometry.dispose(); this._points.material.dispose(); }
    this._renderer.dispose();
    this._canvas.remove();
  }
}

// ═══════════════════════════════════════════════════════════════
//  CryoViewer  –  main application controller
// ═══════════════════════════════════════════════════════════════
class CryoViewer {
  constructor() {
    this.samples         = [];
    this.currentSample   = null;
    this.currentSNR      = null;
    this.currentParticle = 0;
    this.currentSlice    = null;
    this.nParticles      = 1;
    this.nSlices         = 128;
    this.gtMode          = "mip";   // "mip" | "slice" | "3d"

    // Shared 2-D zoom/pan (absolute tx, ty)
    this.transform = { scale: 1, tx: 0, ty: 0 };
    this.drag      = { active: false, x0: 0, y0: 0, tx0: 0, ty0: 0 };

    this._renderer3d = null;

    this._init();
  }

  // ── Boot ──────────────────────────────────────────────────────

  async _init() {
    try {
      const [cfg, samples] = await Promise.all([
        fetch("/api/config").then(r => r.json()),
        fetch("/api/samples").then(r => r.json()),
      ]);
      this.samples    = samples;
      this.currentSNR = cfg.snr_levels?.[0] ?? null;

      this._buildSNRTabs(cfg.snr_levels ?? []);
      this._buildSampleList(samples);
      this._bindPanelEvents();
      this._bindNavControls();
      this._bindZoomButtons();
      this._bind3DToggle();

      document.getElementById("sample-count").textContent = samples.length;
      if (samples.length) this._selectSample(samples[0]);
    } catch (e) { console.error("Init error:", e); }
  }

  // ── SNR tabs ──────────────────────────────────────────────────

  _buildSNRTabs(levels) {
    const el = document.getElementById("snr-tabs");
    if (!levels.length) { el.closest(".snr-group").style.display = "none"; return; }

    // Human-readable labels
    const labels = { "snr0.001": "0.001 — very noisy", "snr0.005": "0.005 — noisy", "snr0.01": "0.01 — less noisy" };
    el.innerHTML = levels.map((snr, i) =>
      `<button class="snr-tab${i===0?" active":""}" data-snr="${snr}">${labels[snr] ?? snr}</button>`
    ).join("");

    el.addEventListener("click", e => {
      const btn = e.target.closest(".snr-tab");
      if (!btn) return;
      el.querySelectorAll(".snr-tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      this.currentSNR = btn.dataset.snr;
      this._updateStatus();
      if (this.currentSample && this.gtMode !== "3d") this._loadNoisyImage();
    });
  }

  // ── Sample list ───────────────────────────────────────────────

  _buildSampleList(samples) {
    const list = document.getElementById("sample-list");
    list.innerHTML = "";
    if (!samples.length) { list.innerHTML = `<div class="list-placeholder"><span>No samples found</span></div>`; return; }

    const frag = document.createDocumentFragment();
    samples.forEach(s => {
      const snr0 = s.snr_levels?.[0] ?? "";
      const card = document.createElement("div");
      card.className  = "sample-card";
      card.dataset.id = s.id;
      card.innerHTML  = `
        <div class="sample-thumb-wrap">
          <img class="sample-thumb" loading="lazy"
               src="/api/thumbnail/noisy/${s.id}${snr0?"?snr="+snr0:""}" alt=""/>
        </div>
        <div class="sample-info">
          <div class="sample-name">Sample ${s.id}</div>
          <div class="sample-meta">${s.snr_levels?.length??0} noise levels · 1000 projections</div>
        </div>`;
      card.addEventListener("click", () => this._selectSample(s));
      frag.appendChild(card);
    });
    list.appendChild(frag);
  }

  // ── Select sample ─────────────────────────────────────────────

  async _selectSample(sample) {
    this.currentSample   = sample;
    this.currentParticle = 0;
    this.currentSlice    = null;
    this.gtMode          = "mip";

    document.querySelectorAll(".sample-card").forEach(c =>
      c.classList.toggle("active", c.dataset.id === sample.id)
    );
    document.querySelector(".sample-card.active")?.scrollIntoView({ block:"nearest", behavior:"smooth" });

    document.getElementById("empty-noisy")?.classList.add("hidden");
    document.getElementById("empty-gt")?.classList.add("hidden");

    // Reset 3D view if open
    if (this._renderer3d) { this._renderer3d.dispose(); this._renderer3d = null; }
    this._set3DActive(false);
    this._updateGTBadge();
    this._updateSliceNavVisibility();

    this._setLoading(true);
    try {
      await this._fetchMetadata(sample);
      await Promise.all([this._loadNoisyImage(), this._loadGTImage()]);
    } finally {
      this._setLoading(false);
    }
  }

  // ── Image URLs ────────────────────────────────────────────────

  _noisyURL() {
    const snr = this.currentSNR ? `snr=${encodeURIComponent(this.currentSNR)}&` : "";
    return `/api/image/noisy/${this.currentSample.id}?${snr}particle=${this.currentParticle}`;
  }
  _gtURL() {
    if (this.gtMode === "mip") return `/api/image/gt/${this.currentSample.id}?axis=0`;
    return `/api/image/gt/${this.currentSample.id}?axis=0&slice=${this.currentSlice??0}`;
  }

  _loadNoisyImage() { return this._setImage("img-noisy", this._noisyURL(), true); }
  _loadGTImage()    { return this._setImage("img-gt",    this._gtURL(),    false); }

  _setImage(imgId, url, center) {
    return new Promise(resolve => {
      const el    = document.getElementById(imgId);
      const proxy = new Image();
      proxy.onload = () => {
        el.src = proxy.src;
        el.width  = proxy.naturalWidth;
        el.height = proxy.naturalHeight;
        if (center) {
          const c  = el.closest(".image-container");
          this.transform.tx = (c.clientWidth  - proxy.naturalWidth  * this.transform.scale) / 2;
          this.transform.ty = (c.clientHeight - proxy.naturalHeight * this.transform.scale) / 2;
          this._applyTransform();
          this._updateZoomLabel();
        } else {
          this._applyTransform();
        }
        el.classList.remove("fade-in"); void el.offsetWidth; el.classList.add("fade-in");
        resolve();
      };
      proxy.onerror = () => resolve();
      proxy.src = url;
    });
  }

  // ── Transform ────────────────────────────────────────────────

  _applyTransform() {
    const { scale, tx, ty } = this.transform;
    const t = `translate(${tx}px,${ty}px) scale(${scale})`;
    ["noisy","gt"].forEach(type => {
      const img = document.getElementById(`img-${type}`);
      if (img) img.style.transform = t;
    });
  }

  _resetTransform() {
    const img = document.getElementById("img-noisy");
    const c   = document.getElementById("container-noisy");
    if (img?.naturalWidth && c) {
      this.transform = { scale:1,
        tx: (c.clientWidth  - img.naturalWidth)  / 2,
        ty: (c.clientHeight - img.naturalHeight) / 2 };
    } else {
      this.transform = { scale:1, tx:0, ty:0 };
    }
    this._applyTransform();
    this._updateZoomLabel();
  }

  _fitToWindow() {
    const img = document.getElementById("img-noisy");
    const c   = document.getElementById("container-noisy");
    if (!img?.naturalWidth || !c) { this._resetTransform(); return; }
    const scale = Math.min(c.clientWidth/img.naturalWidth, c.clientHeight/img.naturalHeight) * 0.9;
    this.transform = { scale, tx:(c.clientWidth-img.naturalWidth*scale)/2, ty:(c.clientHeight-img.naturalHeight*scale)/2 };
    this._applyTransform();
    this._updateZoomLabel();
  }

  _zoom(factor, cx, cy) {
    const oldScale = this.transform.scale;
    const newScale = Math.max(0.05, Math.min(80, oldScale * factor));
    const ratio    = newScale / oldScale;
    this.transform.tx    = cx - ratio * (cx - this.transform.tx);
    this.transform.ty    = cy - ratio * (cy - this.transform.ty);
    this.transform.scale = newScale;
    this._applyTransform();
    this._updateZoomLabel();
  }

  _zoomCenter(factor) {
    const c = document.getElementById("container-noisy");
    if (c) this._zoom(factor, c.clientWidth/2, c.clientHeight/2);
  }

  _updateZoomLabel() {
    document.getElementById("zoom-label").textContent = Math.round(this.transform.scale*100)+"%";
  }

  // ── Panel mouse events ────────────────────────────────────────

  _bindPanelEvents() {
    ["noisy","gt"].forEach(type => {
      const el = document.getElementById(`container-${type}`);

      el.addEventListener("wheel", e => {
        e.preventDefault();
        const r = el.getBoundingClientRect();
        this._zoom(e.deltaY < 0 ? 1.15 : 1/1.15, e.clientX - r.left, e.clientY - r.top);
      }, { passive: false });

      el.addEventListener("mousedown", e => {
        if (e.button !== 0) return;
        e.preventDefault();
        this.drag = { active:true, x0:e.clientX, y0:e.clientY, tx0:this.transform.tx, ty0:this.transform.ty };
        document.body.style.cursor = "grabbing";
      });

      el.addEventListener("dblclick", () => this._resetTransform());
    });

    document.addEventListener("mousemove", e => {
      if (!this.drag.active) return;
      this.transform.tx = this.drag.tx0 + (e.clientX - this.drag.x0);
      this.transform.ty = this.drag.ty0 + (e.clientY - this.drag.y0);
      this._applyTransform();
    });
    document.addEventListener("mouseup", () => {
      if (this.drag.active) { this.drag.active = false; document.body.style.cursor = ""; }
    });
  }

  // ── Metadata ──────────────────────────────────────────────────

  async _fetchMetadata(sample) {
    const snrPart = this.currentSNR ? `?snr=${encodeURIComponent(this.currentSNR)}` : "";
    try {
      const meta = await fetch(`/api/metadata/${sample.id}${snrPart}`).then(r => r.json());
      this.nParticles = meta.noisy?.n_particles ?? 1;
      this.nSlices    = meta.gt?.shape?.[0] ?? 128;
      if (this.currentSlice === null) this.currentSlice = Math.floor(this.nSlices / 2);
      this._updateNavUI();
      this._renderInfoBar(meta);
    } catch { /* non-fatal */ }
  }

  _renderInfoBar(meta) {
    const items = [];
    if (meta.noisy) {
      const n = meta.noisy;
      items.push(`<span class="info-item"><span class="info-label">Noisy file</span>${n.file}</span>`);
      items.push(`<span class="info-item"><span class="info-label">Projections</span>${n.n_particles.toLocaleString()}</span>`);
      items.push(`<span class="info-item"><span class="info-label">Particle size</span>${n.particle_shape.join("×")} px</span>`);
      if (n.voxel_size_angstrom) items.push(`<span class="info-item"><span class="info-label">Pixel</span>${n.voxel_size_angstrom} Å</span>`);
    }
    if (meta.gt) {
      const g = meta.gt;
      items.push(`<span class="info-sep">│</span>`);
      items.push(`<span class="info-item"><span class="info-label">GT file</span>${g.file}</span>`);
      items.push(`<span class="info-item"><span class="info-label">Volume</span>${g.shape.join("×")} voxels</span>`);
      if (g.voxel_size?.x) items.push(`<span class="info-item"><span class="info-label">Voxel</span>${g.voxel_size.x.toFixed(2)} Å</span>`);
    }
    document.getElementById("info-bar").innerHTML = items.join("");
  }

  _updateStatus() {
    const s = this.currentSample;
    document.getElementById("status-sample").textContent   = s ? `Sample ${s.id}` : "No sample selected";
    document.getElementById("status-particle").textContent = s ? `Projection ${this.currentParticle} / ${this.nParticles-1}` : "—";
    document.getElementById("status-snr").textContent      = this.currentSNR ? `SNR ${this.currentSNR.replace("snr","")}` : "—";
  }

  // ── 3D toggle ─────────────────────────────────────────────────

  _bind3DToggle() {
    document.getElementById("btn-view-2d").addEventListener("click", () => {
      if (this.gtMode === "3d") this._switch2D();
    });
    document.getElementById("btn-view-3d").addEventListener("click", () => {
      if (this.gtMode !== "3d") this._switch3D();
    });
  }

  _switch3D() {
    if (!this.currentSample) return;
    this.gtMode = "3d";
    this._set3DActive(true);
    this._updateGTBadge();
    this._updateSliceNavVisibility();
    this._load3DVolume();
  }

  _switch2D() {
    if (this._renderer3d) { this._renderer3d.dispose(); this._renderer3d = null; }
    this.gtMode = this.currentSlice !== null ? "slice" : "mip";
    this._set3DActive(false);
    this._updateGTBadge();
    this._updateSliceNavVisibility();
    this._loadGTImage();
  }

  _set3DActive(on) {
    document.getElementById("container-gt").classList.toggle("hidden", on);
    document.getElementById("three-container").classList.toggle("hidden", !on);
    document.getElementById("btn-view-2d").classList.toggle("active", !on);
    document.getElementById("btn-view-3d").classList.toggle("active",  on);

    // Disable 2D zoom controls in 3D mode (they'd be confusing)
    ["btn-zoom-in","btn-zoom-out","btn-reset","btn-fit"].forEach(id => {
      document.getElementById(id).disabled = on;
      document.getElementById(id).style.opacity = on ? "0.3" : "";
    });
  }

  async _load3DVolume() {
    const sample = this.currentSample;
    if (!sample) return;

    const overlay = document.getElementById("three-overlay");
    overlay.classList.remove("hidden");

    try {
      const data = await fetch(`/api/volume3d/${sample.id}`).then(r => r.json());

      // Create or reuse renderer
      const container = document.getElementById("three-container");
      if (!this._renderer3d) {
        this._renderer3d = new Volume3DRenderer(container);
      }

      overlay.classList.add("hidden");
      this._renderer3d.loadPointCloud(data);

      // Update stats overlay
      const snr = this.currentSNR?.replace("snr","") ?? "";
      document.getElementById("three-stats").innerHTML =
        `Sample ${sample.id}<br>${data.x.length.toLocaleString()} points shown<br>${data.n_total.toLocaleString()} total voxels`;

      // Resize after container becomes visible
      requestAnimationFrame(() => this._renderer3d?.resize());

    } catch (e) {
      console.error("3D load error:", e);
      overlay.classList.add("hidden");
    }
  }

  // ── Navigation ────────────────────────────────────────────────

  _bindNavControls() {
    const bind = (id, fn) => document.getElementById(id)?.addEventListener("click", fn);
    bind("btn-first-particle", () => this._setParticle(0));
    bind("btn-last-particle",  () => this._setParticle(this.nParticles - 1));
    bind("btn-prev-particle",  () => this._setParticle(this.currentParticle - 1));
    bind("btn-next-particle",  () => this._setParticle(this.currentParticle + 1));
    document.getElementById("particle-input").addEventListener("change", e =>
      this._setParticle(parseInt(e.target.value, 10) || 0)
    );

    bind("btn-first-slice", () => this._setSlice(0));
    bind("btn-last-slice",  () => this._setSlice(this.nSlices - 1));
    bind("btn-prev-slice",  () => this._setSlice(this.currentSlice - 1));
    bind("btn-next-slice",  () => this._setSlice(this.currentSlice + 1));
    document.getElementById("slice-input").addEventListener("change", e =>
      this._setSlice(parseInt(e.target.value, 10) || 0)
    );
    bind("btn-back-mip", () => this._backToMIP());

    // Keyboard shortcuts
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT") return;
      switch (e.key) {
        case "ArrowLeft":  e.preventDefault(); this._setParticle(this.currentParticle - 1); break;
        case "ArrowRight": e.preventDefault(); this._setParticle(this.currentParticle + 1); break;
        case "ArrowUp":    e.preventDefault(); this._setSlice(this.currentSlice - 1); break;
        case "ArrowDown":  e.preventDefault(); this._setSlice(this.currentSlice + 1); break;
        case "r": case "R": this._resetTransform(); break;
        case "f": case "F": this._fitToWindow(); break;
        case "+": case "=": this._zoomCenter(1.2); break;
        case "-": case "_": this._zoomCenter(1/1.2); break;
        case "3": this._switch3D(); break;
        case "2": this._switch2D(); break;
      }
    });
  }

  _setParticle(idx) {
    idx = Math.max(0, Math.min(this.nParticles - 1, idx));
    if (idx === this.currentParticle) return;
    this.currentParticle = idx;
    document.getElementById("particle-input").value = idx;
    this._updateStatus();
    if (this.gtMode !== "3d") this._loadNoisyImage();
  }

  _setSlice(idx) {
    idx = Math.max(0, Math.min(this.nSlices - 1, idx));
    if (idx === this.currentSlice && this.gtMode === "slice") return;
    this.currentSlice = idx;
    this.gtMode = "slice";
    document.getElementById("slice-input").value = idx;
    this._updateGTBadge();
    this._updateSliceNavVisibility();
    this._loadGTImage();
  }

  _backToMIP() {
    this.gtMode = "mip";
    this._updateGTBadge();
    this._updateSliceNavVisibility();
    this._loadGTImage();
  }

  _updateNavUI() {
    document.getElementById("particle-input").value = this.currentParticle;
    document.getElementById("particle-input").max   = this.nParticles - 1;
    document.getElementById("particle-max").textContent = `/ ${this.nParticles-1}`;

    document.getElementById("slice-input").value = this.currentSlice ?? Math.floor(this.nSlices/2);
    document.getElementById("slice-input").max   = this.nSlices - 1;
    document.getElementById("slice-max").textContent = `/ ${this.nSlices-1}`;

    this._updateSliceNavVisibility();
    this._updateStatus();
  }

  _updateSliceNavVisibility() {
    // Hide Z-slice controls unless we're actually in slice mode
    const show = (this.gtMode === "slice");
    document.getElementById("slice-nav")?.classList.toggle("hidden", !show);
    document.getElementById("slice-sep")?.classList.toggle("hidden", !show);
  }

  _updateGTBadge() {
    const badge = document.getElementById("gt-mode-badge");
    const sub   = document.getElementById("gt-sub");
    if (!badge) return;
    if (this.gtMode === "3d") {
      badge.textContent = "3D";
      badge.style.background = "rgba(78,158,255,.12)";
      badge.style.color = "var(--accent)";
      badge.style.borderColor = "rgba(78,158,255,.3)";
      if (sub) sub.textContent = "Interactive 3D point cloud";
    } else if (this.gtMode === "slice") {
      badge.textContent = `Z=${this.currentSlice}`;
      badge.style.background = "rgba(67,197,158,.12)";
      badge.style.color = "var(--gt-color)";
      badge.style.borderColor = "rgba(67,197,158,.25)";
      if (sub) sub.textContent = `Slice ${this.currentSlice} of ${this.nSlices-1}`;
    } else {
      badge.textContent = "MIP";
      badge.style.background = "rgba(67,197,158,.12)";
      badge.style.color = "var(--gt-color)";
      badge.style.borderColor = "rgba(67,197,158,.25)";
      if (sub) sub.textContent = "Top-down projection of full volume";
    }
  }

  _updateNoisySub() {
    const sub = document.getElementById("noisy-sub");
    if (!sub || !this.currentSample) return;
    const snrLabel = this.currentSNR?.replace("snr","") ?? "?";
    sub.textContent = `Projection ${this.currentParticle} / ${this.nParticles-1}  ·  SNR ${snrLabel}`;
  }

  // ── Zoom buttons ──────────────────────────────────────────────

  _bindZoomButtons() {
    const bind = (id, fn) => document.getElementById(id)?.addEventListener("click", fn);
    bind("btn-zoom-in",  () => this._zoomCenter(1.25));
    bind("btn-zoom-out", () => this._zoomCenter(1/1.25));
    bind("btn-reset",    () => this._resetTransform());
    bind("btn-fit",      () => this._fitToWindow());
  }

  // ── Helpers ───────────────────────────────────────────────────

  _setLoading(on) {
    document.getElementById("loading-overlay").style.display = on ? "flex" : "none";
  }
}

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => { window.viewer = new CryoViewer(); });
