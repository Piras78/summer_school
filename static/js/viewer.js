"use strict";

// ═══════════════════════════════════════════════════════════════
//  Plasma colormap helper (for 3-D point cloud)
// ═══════════════════════════════════════════════════════════════
function plasmaColor(t) {
  const stops = [[13,8,135],[84,2,163],[139,10,165],[185,50,137],[219,92,104],[244,136,73],[252,253,191]];
  const n = stops.length - 1;
  const pos = Math.max(0, Math.min(1, t)) * n;
  const i = Math.min(Math.floor(pos), n - 1);
  const f = pos - i;
  const a = stops[i], b = stops[i + 1];
  return [(a[0]+(b[0]-a[0])*f)/255, (a[1]+(b[1]-a[1])*f)/255, (a[2]+(b[2]-a[2])*f)/255];
}

// ═══════════════════════════════════════════════════════════════
//  ProjectionSphereWidget  –  3-D widget showing viewing directions
// ═══════════════════════════════════════════════════════════════
class ProjectionSphereWidget {
  constructor(container) {
    this.container    = container;
    this._orientations = [];
    this._orbit       = { theta: 0.4, phi: 1.05, dragging: false, lx: 0, ly: 0 };
    this._raf         = null;
    this._points      = null;
    this._marker      = null;
    this._viewLine    = null;
    this._setup();
    this._bindEvents();
    this._animate();
  }

  _setup() {
    const w = 148, h = 148;
    this._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this._renderer.setClearColor(0x000000, 0);

    this._canvas = this._renderer.domElement;
    this._canvas.style.cssText = "display:block;width:148px;height:148px;";
    // Insert before the legend div
    const legend = this.container.querySelector(".sphere-legend");
    this.container.insertBefore(this._canvas, legend);

    this._scene  = new THREE.Scene();
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this._camera.position.z = 2.8;

    // Wireframe sphere
    this._scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 10),
      new THREE.MeshBasicMaterial({ color: 0x1e2236, wireframe: true, transparent: true, opacity: 0.22 })
    ));

    // North-pole marker = MIP direction (0,1,0) — green
    const npMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0x43c59e })
    );
    npMesh.position.set(0, 1, 0);
    this._scene.add(npMesh);

    // Thin line from center to north pole
    const npLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,1,0)]),
      new THREE.LineBasicMaterial({ color: 0x43c59e, transparent: true, opacity: 0.35 })
    );
    this._scene.add(npLine);
  }

  setOrientations(orientations) {
    this._orientations = orientations;
    if (this._points) {
      this._scene.remove(this._points);
      this._points.geometry.dispose();
      this._points.material.dispose();
      this._points = null;
    }
    if (!orientations.length) return;

    const n   = orientations.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    orientations.forEach((o, i) => {
      pos[i*3]   = o.x; pos[i*3+1] = o.y; pos[i*3+2] = o.z;
      col[i*3]   = 0.28; col[i*3+1] = 0.35; col[i*3+2] = 0.52;
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color",    new THREE.BufferAttribute(col, 3));
    this._points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.055, vertexColors: true, transparent: true, opacity: 0.85, sizeAttenuation: true,
    }));
    this._scene.add(this._points);
  }

  highlight(idx) {
    // Remove previous orange marker & line
    if (this._marker)   { this._scene.remove(this._marker);   this._marker.geometry.dispose();   this._marker.material.dispose(); }
    if (this._viewLine) { this._scene.remove(this._viewLine); this._viewLine.geometry.dispose(); this._viewLine.material.dispose(); }
    this._marker = this._viewLine = null;

    if (!this._orientations.length || idx >= this._orientations.length) return;
    const o = this._orientations[idx];
    const v = new THREE.Vector3(o.x, o.y, o.z);

    this._marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.1, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xf0a050 })
    );
    this._marker.position.copy(v);
    this._scene.add(this._marker);

    this._viewLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), v.clone()]),
      new THREE.LineBasicMaterial({ color: 0xf0a050, transparent: true, opacity: 0.75 })
    );
    this._scene.add(this._viewLine);
  }

  _bindEvents() {
    const el = this._canvas;
    el.addEventListener("mousedown", e => {
      this._orbit.dragging = true;
      this._orbit.lx = e.clientX; this._orbit.ly = e.clientY;
      e.stopPropagation(); e.preventDefault();
    });
    document.addEventListener("mousemove", e => {
      if (!this._orbit.dragging) return;
      this._orbit.theta -= (e.clientX - this._orbit.lx) * 0.012;
      this._orbit.phi    = Math.max(0.1, Math.min(Math.PI - 0.1, this._orbit.phi + (e.clientY - this._orbit.ly) * 0.012));
      this._orbit.lx = e.clientX; this._orbit.ly = e.clientY;
    });
    document.addEventListener("mouseup", () => { this._orbit.dragging = false; });
    el.addEventListener("wheel", e => { e.preventDefault(); e.stopPropagation(); }, { passive: false });
  }

  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    if (!this._orbit.dragging) this._orbit.theta += 0.007;
    const { theta, phi } = this._orbit;
    const r = 2.8;
    this._camera.position.set(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    );
    this._camera.lookAt(0, 0, 0);
    this._renderer.render(this._scene, this._camera);
  }

  dispose() {
    if (this._raf) cancelAnimationFrame(this._raf);
    [this._points, this._marker, this._viewLine].forEach(o => {
      if (o) { this._scene.remove(o); o.geometry?.dispose(); o.material?.dispose(); }
    });
    this._renderer.dispose();
    this._canvas.remove();
  }
}

// ═══════════════════════════════════════════════════════════════
//  ColormapEngine  –  apply scientific colormaps via canvas LUT
// ═══════════════════════════════════════════════════════════════
class ColormapEngine {
  constructor() {
    this.current = "gray";
    this._luts = {
      gray:    null,
      hot:     this._buildLUT([[0,0,0],[255,0,0],[255,255,0],[255,255,255]]),
      viridis: this._buildLUT([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]),
      plasma:  this._buildLUT([[13,8,135],[84,2,163],[185,50,137],[244,136,73],[252,253,191]]),
    };
  }

  _buildLUT(stops) {
    const lut = new Uint8Array(256 * 3);
    const n   = stops.length - 1;
    for (let i = 0; i < 256; i++) {
      const t   = i / 255;
      const pos = t * n;
      const idx = Math.min(Math.floor(pos), n - 1);
      const f   = pos - idx;
      const a   = stops[idx], b = stops[idx + 1];
      lut[i*3]   = Math.round(a[0] + (b[0]-a[0]) * f);
      lut[i*3+1] = Math.round(a[1] + (b[1]-a[1]) * f);
      lut[i*3+2] = Math.round(a[2] + (b[2]-a[2]) * f);
    }
    return lut;
  }

  // Apply current colormap from srcImg → destImg (synchronous).
  applyToImg(srcImg, destImg) {
    if (this.current === "gray") {
      // For gray mode, set src directly from proxy; origSrc already stored by caller
      if (destImg !== srcImg) destImg.src = srcImg.src;
      return;
    }
    const lut    = this._luts[this.current];
    const canvas = document.createElement("canvas");
    canvas.width  = srcImg.naturalWidth;
    canvas.height = srcImg.naturalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(srcImg, 0, 0);
    const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d  = id.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i];
      d[i]   = lut[v*3];
      d[i+1] = lut[v*3+1];
      d[i+2] = lut[v*3+2];
    }
    ctx.putImageData(id, 0, 0);
    destImg.src = canvas.toDataURL("image/png");
  }

  // Re-apply colormap to an img that has dataset.origSrc set.
  reapply(imgEl) {
    const orig = imgEl.dataset.origSrc;
    if (!orig) return;
    if (this.current === "gray") { imgEl.src = orig; return; }
    const proxy = new Image();
    proxy.crossOrigin = "anonymous";
    proxy.onload = () => this.applyToImg(proxy, imgEl);
    proxy.src    = orig;
  }
}

// ═══════════════════════════════════════════════════════════════
//  Volume3DRenderer  –  Three.js point-cloud viewer
// ═══════════════════════════════════════════════════════════════
class Volume3DRenderer {
  constructor(container) {
    this.container = container;
    this._raf       = null;
    this._points    = null;
    this._viewArrow = null;
    this._orbit     = { theta:0.5, phi:1.1, radius:200, dragging:false, lx:0, ly:0 };
    this._setup();
    this._bindEvents();
    this._animate();
  }

  _setup() {
    const w = this.container.clientWidth  || 600;
    const h = this.container.clientHeight || 400;

    this._renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this._renderer.setClearColor(0x060810, 1);

    this._canvas = this._renderer.domElement;
    this._canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;";
    this.container.appendChild(this._canvas);

    this._scene  = new THREE.Scene();
    this._camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000);

    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(128, 128, 128)),
      new THREE.LineBasicMaterial({ color:0x2a3258, transparent:true, opacity:0.5 })
    );
    this._scene.add(box);

    const grid = new THREE.GridHelper(128, 8, 0x1a2040, 0x1a2040);
    grid.position.y = -64;
    this._scene.add(grid);
  }

  _bindEvents() {
    const el = this._canvas;
    el.addEventListener("mousedown", e => {
      this._orbit.dragging = true;
      this._orbit.lx = e.clientX; this._orbit.ly = e.clientY;
      e.stopPropagation();
    });
    document.addEventListener("mousemove", e => {
      if (!this._orbit.dragging) return;
      this._orbit.theta -= (e.clientX - this._orbit.lx) * 0.007;
      this._orbit.phi    = Math.max(0.05, Math.min(Math.PI-0.05, this._orbit.phi + (e.clientY - this._orbit.ly) * 0.007));
      this._orbit.lx = e.clientX; this._orbit.ly = e.clientY;
    });
    document.addEventListener("mouseup", () => { this._orbit.dragging = false; });
    el.addEventListener("wheel", e => {
      e.preventDefault(); e.stopPropagation();
      this._orbit.radius = Math.max(60, Math.min(500, this._orbit.radius + e.deltaY * 0.25));
    }, { passive:false });
  }

  loadPointCloud(data) {
    if (this._points) { this._scene.remove(this._points); this._points.geometry.dispose(); this._points.material.dispose(); }
    const n = data.x.length;
    if (!n) return;
    const pos   = new Float32Array(n * 3);
    const color = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i*3]   = data.x[i];
      pos[i*3+1] = data.z[i];
      pos[i*3+2] = data.y[i];
      const [r, g, b] = plasmaColor(data.intensity[i]);
      color[i*3] = r; color[i*3+1] = g; color[i*3+2] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos,   3));
    geo.setAttribute("color",    new THREE.BufferAttribute(color, 3));
    this._points = new THREE.Points(geo, new THREE.PointsMaterial({ size:3, vertexColors:true, transparent:true, opacity:0.9, sizeAttenuation:true }));
    this._scene.add(this._points);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this._camera.aspect = w / h; this._camera.updateProjectionMatrix();
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

  // Show/update an arrow from the viewing direction toward the molecule centre.
  setViewArrow(x, y, z) {
    if (this._viewArrow) { this._scene.remove(this._viewArrow); this._viewArrow = null; }

    const origin = new THREE.Vector3(x, y, z).normalize().multiplyScalar(85);
    const dir    = new THREE.Vector3(-x, -y, -z).normalize();
    this._viewArrow = new THREE.ArrowHelper(dir, origin, 55, 0xf0a050, 12, 6);
    this._scene.add(this._viewArrow);
  }

  clearViewArrow() {
    if (this._viewArrow) { this._scene.remove(this._viewArrow); this._viewArrow = null; }
  }

  dispose() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._points)    { this._points.geometry.dispose(); this._points.material.dispose(); }
    if (this._viewArrow) { this._scene.remove(this._viewArrow); }
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

    this.transform = { scale:1, tx:0, ty:0 };
    this.drag      = { active:false, x0:0, y0:0, tx0:0, ty0:0 };

    this.colormap  = new ColormapEngine();
    this.splitMode = false;
    this.splitPos  = 50;           // percentage from left

    this._renderer3d          = null;
    this._scrubDebounce       = null;
    this._splitHandleDragging = false;
    this._sphereWidget        = null;
    this._orientations        = [];   // [{x,y,z,rot,tilt,psi}] for current sample

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
      this._bindColormapControls();
      this._bindScrubber();
      this._bindSplitView();
      this._bindOrientationWidget();

      document.getElementById("sample-count").textContent = samples.length;
      if (samples.length) this._selectSample(samples[0]);
    } catch (e) { console.error("Init error:", e); }
  }

  // ── SNR tabs ──────────────────────────────────────────────────

  _buildSNRTabs(levels) {
    const el = document.getElementById("snr-tabs");
    if (!levels.length) { el.closest(".snr-group").style.display = "none"; return; }
    const labels = {
      "snr0.001": "0.001 — very noisy",
      "snr0.005": "0.005 — noisy",
      "snr0.01":  "0.01 — less noisy",
    };
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
    if (!samples.length) {
      list.innerHTML = `<div class="list-placeholder"><span>No samples found</span></div>`;
      return;
    }
    const frag = document.createDocumentFragment();
    samples.forEach(s => {
      const snr0 = s.snr_levels?.[0] ?? "";
      const card = document.createElement("div");
      card.className  = "sample-item";
      card.dataset.id = s.id;
      card.innerHTML  = `
        <img class="sample-thumb" loading="lazy"
             src="/api/thumbnail/noisy/${s.id}${snr0?"?snr="+snr0:""}" alt=""/>
        <div class="sample-info">
          <div class="sample-name">Sample ${s.id}</div>
          <div class="sample-meta">${s.snr_levels?.length ?? 0} noise levels · 1000 proj.</div>
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

    document.querySelectorAll(".sample-item").forEach(c =>
      c.classList.toggle("active", c.dataset.id === sample.id)
    );
    document.querySelector(".sample-item.active")?.scrollIntoView({ block:"nearest", behavior:"smooth" });

    document.getElementById("empty-noisy")?.classList.add("hidden");
    document.getElementById("empty-gt")?.classList.add("hidden");

    if (this._renderer3d) { this._renderer3d.dispose(); this._renderer3d = null; }
    this._set3DActive(false);
    this._updateGTBadge();
    this._updateSliceNavVisibility();

    // Leave split mode on sample change
    if (this.splitMode) this._deactivateSplit();

    this._orientations = [];
    this._sphereWidget?.setOrientations([]);
    document.getElementById("sphere-widget")?.classList.add("hidden");

    this._setLoading(true);
    try {
      await this._fetchMetadata(sample);
      await Promise.all([this._loadNoisyImage(), this._loadGTImage()]);
      this._loadOrientations(sample);   // async, non-blocking
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
    return `/api/image/gt/${this.currentSample.id}?axis=0&slice=${this.currentSlice ?? 0}`;
  }

  _loadNoisyImage() { return this._setImage("img-noisy", this._noisyURL(), true); }
  _loadGTImage()    { return this._setImage("img-gt",    this._gtURL(),    false); }

  // Core image loader: fetches via proxy, stores origSrc, applies colormap.
  _setImage(imgId, url, center) {
    return new Promise(resolve => {
      const el    = document.getElementById(imgId);
      const proxy = new Image();
      proxy.crossOrigin = "anonymous";
      proxy.onload = () => {
        // Store the original URL so colormap can be re-applied later
        el.dataset.origSrc = url;
        el.width  = proxy.naturalWidth;
        el.height = proxy.naturalHeight;

        if (center) {
          const cont = document.getElementById("container-noisy");
          const scale = this.transform.scale;
          this.transform.tx = (cont.clientWidth  - proxy.naturalWidth  * scale) / 2;
          this.transform.ty = (cont.clientHeight - proxy.naturalHeight * scale) / 2;
        }

        this.colormap.applyToImg(proxy, el);
        this._applyTransform();
        this._updateZoomLabel();

        // Mirror to split view if active
        if (this.splitMode) {
          const splitId = imgId === "img-noisy" ? "split-img-noisy" : "split-img-gt";
          const splitEl = document.getElementById(splitId);
          if (splitEl) {
            splitEl.dataset.origSrc = url;
            splitEl.width  = proxy.naturalWidth;
            splitEl.height = proxy.naturalHeight;
            this.colormap.applyToImg(proxy, splitEl);
          }
        }

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
    if (this.splitMode) {
      const sn = document.getElementById("split-img-noisy");
      const sg = document.getElementById("split-img-gt");
      if (sn) sn.style.transform = t;
      if (sg) sg.style.transform = t;
    }
  }

  _resetTransform() {
    const img  = document.getElementById("img-noisy");
    const cont = document.getElementById("container-noisy");
    if (img?.naturalWidth && cont) {
      this.transform = {
        scale: 1,
        tx: (cont.clientWidth  - img.naturalWidth)  / 2,
        ty: (cont.clientHeight - img.naturalHeight) / 2,
      };
    } else {
      this.transform = { scale:1, tx:0, ty:0 };
    }
    this._applyTransform();
    this._updateZoomLabel();
  }

  _resetTransformSplit() {
    const overlay = document.getElementById("split-overlay");
    const img     = document.getElementById("split-img-noisy");
    if (!img?.naturalWidth || !overlay) return;
    this.transform = {
      scale: 1,
      tx: (overlay.clientWidth  - img.naturalWidth)  / 2,
      ty: (overlay.clientHeight - img.naturalHeight) / 2,
    };
    this._applyTransform();
    this._updateZoomLabel();
  }

  _fitToWindow() {
    const img  = document.getElementById("img-noisy");
    const cont = this.splitMode
      ? document.getElementById("split-overlay")
      : document.getElementById("container-noisy");
    if (!img?.naturalWidth || !cont) { this._resetTransform(); return; }
    const scale = Math.min(cont.clientWidth / img.naturalWidth, cont.clientHeight / img.naturalHeight) * 0.9;
    this.transform = {
      scale,
      tx: (cont.clientWidth  - img.naturalWidth  * scale) / 2,
      ty: (cont.clientHeight - img.naturalHeight * scale) / 2,
    };
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
    const cont = this.splitMode
      ? document.getElementById("split-overlay")
      : document.getElementById("container-noisy");
    if (cont) this._zoom(factor, cont.clientWidth / 2, cont.clientHeight / 2);
  }

  _updateZoomLabel() {
    document.getElementById("zoom-label").textContent = Math.round(this.transform.scale * 100) + "%";
  }

  // ── Panel mouse events ────────────────────────────────────────

  _bindPanelEvents() {
    ["noisy","gt"].forEach(type => {
      const el = document.getElementById(`container-${type}`);

      el.addEventListener("wheel", e => {
        e.preventDefault();
        const r = el.getBoundingClientRect();
        this._zoom(e.deltaY < 0 ? 1.15 : 1/1.15, e.clientX - r.left, e.clientY - r.top);
      }, { passive:false });

      el.addEventListener("mousedown", e => {
        if (e.button !== 0) return;
        e.preventDefault();
        this.drag = { active:true, x0:e.clientX, y0:e.clientY, tx0:this.transform.tx, ty0:this.transform.ty };
        document.body.style.cursor = "grabbing";
      });

      el.addEventListener("dblclick", () => this._resetTransform());
    });

    document.addEventListener("mousemove", e => {
      if (!this.drag.active || this._splitHandleDragging) return;
      this.transform.tx = this.drag.tx0 + (e.clientX - this.drag.x0);
      this.transform.ty = this.drag.ty0 + (e.clientY - this.drag.y0);
      this._applyTransform();
    });
    document.addEventListener("mouseup", () => {
      if (this.drag.active) { this.drag.active = false; document.body.style.cursor = ""; }
      this._splitHandleDragging = false;
    });
  }

  // ── Colormap ─────────────────────────────────────────────────

  _bindColormapControls() {
    document.querySelectorAll(".cmap-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".cmap-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.colormap.current = btn.dataset.cmap;

        // Re-apply to main panel images
        this.colormap.reapply(document.getElementById("img-noisy"));
        this.colormap.reapply(document.getElementById("img-gt"));

        // Re-apply to split view images
        if (this.splitMode) {
          this.colormap.reapply(document.getElementById("split-img-noisy"));
          this.colormap.reapply(document.getElementById("split-img-gt"));
        }
      });
    });
  }

  // ── Particle scrubber ─────────────────────────────────────────

  _bindScrubber() {
    const range   = document.getElementById("particle-range");
    const scrPrev = document.getElementById("scr-prev");
    const scrNext = document.getElementById("scr-next");

    range.addEventListener("input", () => {
      const idx = parseInt(range.value, 10);
      this.currentParticle = idx;
      document.getElementById("particle-input").value = idx;
      this._updateScrubberUI();
      this._updateStatus();
      this._updateNoisySub();
      this._sphereWidget?.highlight(idx);
      this._updateOrientationLabel(idx);
      // Debounce the actual image fetch
      clearTimeout(this._scrubDebounce);
      this._scrubDebounce = setTimeout(() => {
        if (this.currentSample && this.gtMode !== "3d") this._loadNoisyImage();
      }, 80);
    });

    scrPrev.addEventListener("click", () => this._setParticle(this.currentParticle - 1));
    scrNext.addEventListener("click", () => this._setParticle(this.currentParticle + 1));
  }

  _updateScrubberUI() {
    const max   = Math.max(0, this.nParticles - 1);
    const range = document.getElementById("particle-range");
    const label = document.getElementById("scr-label");
    const pct   = max > 0 ? (this.currentParticle / max) * 100 : 0;
    if (range) {
      range.max   = max;
      range.value = this.currentParticle;
      range.style.setProperty("--range-pct", pct.toFixed(2) + "%");
    }
    if (label) label.textContent = `${this.currentParticle} / ${max}`;
  }

  // ── Split view ────────────────────────────────────────────────

  _bindSplitView() {
    document.getElementById("btn-split").addEventListener("click", () => {
      this.splitMode ? this._deactivateSplit() : this._activateSplit();
    });

    const overlay = document.getElementById("split-overlay");
    const handle  = document.getElementById("split-handle");

    // Wheel: zoom (same as panels)
    overlay.addEventListener("wheel", e => {
      e.preventDefault();
      const r = overlay.getBoundingClientRect();
      this._zoom(e.deltaY < 0 ? 1.15 : 1/1.15, e.clientX - r.left, e.clientY - r.top);
    }, { passive:false });

    // Pan on overlay (but not when clicking the handle)
    overlay.addEventListener("mousedown", e => {
      if (e.button !== 0) return;
      if (e.target.closest(".split-handle")) return;
      e.preventDefault();
      this.drag = { active:true, x0:e.clientX, y0:e.clientY, tx0:this.transform.tx, ty0:this.transform.ty };
      document.body.style.cursor = "grabbing";
    });

    overlay.addEventListener("dblclick", e => {
      if (!e.target.closest(".split-handle")) this._resetTransformSplit();
    });

    // Handle drag: move the split divider
    handle.addEventListener("mousedown", e => {
      e.preventDefault();
      e.stopPropagation();
      this._splitHandleDragging = true;
    });

    document.addEventListener("mousemove", e => {
      if (!this._splitHandleDragging) return;
      const r   = overlay.getBoundingClientRect();
      const pct = Math.max(5, Math.min(95, ((e.clientX - r.left) / r.width) * 100));
      this._setSplitPos(pct);
    });

    // mouseup is handled in _bindPanelEvents (clears splitHandleDragging)
  }

  _activateSplit() {
    if (!this.currentSample) return;

    // Auto-switch from 3D to 2D MIP if needed
    if (this.gtMode === "3d") this._switch2D();

    this.splitMode = true;
    this.splitPos  = 50;

    document.getElementById("btn-split").classList.add("active");

    // Copy current images + origSrc into split view
    const copyToSplit = (mainId, splitId) => {
      const main  = document.getElementById(mainId);
      const split = document.getElementById(splitId);
      split.src            = main.src;
      split.dataset.origSrc = main.dataset.origSrc || "";
      split.width  = main.naturalWidth;
      split.height = main.naturalHeight;
    };
    copyToSplit("img-noisy", "split-img-noisy");
    copyToSplit("img-gt",    "split-img-gt");

    // Recentre/fit transform for the full split overlay container
    const overlay = document.getElementById("split-overlay");
    const img     = document.getElementById("img-noisy");
    if (img.naturalWidth && overlay.clientWidth) {
      const scale = Math.min(
        overlay.clientWidth  / img.naturalWidth,
        overlay.clientHeight / img.naturalHeight
      ) * 0.9;
      this.transform = {
        scale,
        tx: (overlay.clientWidth  - img.naturalWidth  * scale) / 2,
        ty: (overlay.clientHeight - img.naturalHeight * scale) / 2,
      };
    }

    overlay.classList.remove("hidden");
    this._setSplitPos(50);
    this._applyTransform();
    this._updateZoomLabel();
  }

  _deactivateSplit() {
    this.splitMode = false;
    document.getElementById("btn-split").classList.remove("active");
    document.getElementById("split-overlay").classList.add("hidden");
    this._fitToWindow();
  }

  _setSplitPos(pct) {
    this.splitPos = pct;
    const sideNoisy = document.getElementById("split-side-noisy");
    const sideGT    = document.getElementById("split-side-gt");
    const handle    = document.getElementById("split-handle");
    if (sideNoisy) sideNoisy.style.clipPath = `inset(0 ${(100-pct).toFixed(2)}% 0 0)`;
    if (sideGT)    sideGT.style.clipPath    = `inset(0 0 0 ${pct.toFixed(2)}%)`;
    if (handle)    handle.style.left        = `${pct}%`;
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
      items.push(`<span class="info-item"><span class="info-key">File</span><span class="info-val">${n.file}</span></span>`);
      items.push(`<span class="info-item"><span class="info-key">Projections</span><span class="info-val">${n.n_particles.toLocaleString()}</span></span>`);
      items.push(`<span class="info-item"><span class="info-key">Size</span><span class="info-val">${n.particle_shape.join("×")} px</span></span>`);
      if (n.voxel_size_angstrom) items.push(`<span class="info-item"><span class="info-key">Pixel</span><span class="info-val">${n.voxel_size_angstrom} Å</span></span>`);
      items.push(`<span class="info-item"><span class="info-key">Noisy file</span><span class="info-val">${n.size_mb} MB</span></span>`);
    }
    if (meta.gt) {
      const g = meta.gt;
      items.push(`<span style="color:var(--border-light);margin:0 4px">│</span>`);
      items.push(`<span class="info-item"><span class="info-key">GT file</span><span class="info-val">${g.file}</span></span>`);
      items.push(`<span class="info-item"><span class="info-key">Volume</span><span class="info-val">${g.shape.join("×")} voxels</span></span>`);
      if (g.voxel_size?.x) items.push(`<span class="info-item"><span class="info-key">Voxel</span><span class="info-val">${g.voxel_size.x.toFixed(2)} Å</span></span>`);
    }
    document.getElementById("info-bar").innerHTML = items.join("");
  }

  _updateStatus() {
    const s = this.currentSample;
    document.getElementById("status-sample").textContent   = s ? `Sample ${s.id}` : "No sample selected";
    document.getElementById("status-particle").textContent = s ? `Proj. ${this.currentParticle} / ${this.nParticles-1}` : "—";
    document.getElementById("status-snr").textContent      = this.currentSNR ? `SNR ${this.currentSNR.replace("snr","")}` : "—";
  }

  _updateNoisySub() {
    const sub = document.getElementById("noisy-sub");
    if (!sub || !this.currentSample) return;
    const snrLabel = this.currentSNR?.replace("snr","") ?? "?";
    sub.textContent = `Projection ${this.currentParticle} / ${this.nParticles-1}  ·  SNR ${snrLabel}`;
  }

  // ── 3D toggle ─────────────────────────────────────────────────

  _bind3DToggle() {
    document.getElementById("btn-view-2d").addEventListener("click", () => {
      if (this.gtMode === "3d") this._switch2D();
    });
    document.getElementById("btn-view-3d").addEventListener("click", () => {
      if (this.gtMode !== "3d") this._switch3D();
    });

    // Clicking the MIP/Z badge toggles between MIP and Z-slice mode
    document.getElementById("gt-mode-badge").addEventListener("click", () => {
      if (!this.currentSample || this.gtMode === "3d") return;
      if (this.gtMode === "mip") {
        // Enter slice mode at the middle Z
        this._setSlice(Math.floor(this.nSlices / 2));
      } else {
        this._backToMIP();
      }
    });
  }

  _switch3D() {
    if (!this.currentSample) return;
    if (this.splitMode) this._deactivateSplit();
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
    ["btn-zoom-in","btn-zoom-out","btn-reset","btn-fit"].forEach(id => {
      const el = document.getElementById(id);
      el.disabled     = on;
      el.style.opacity = on ? "0.3" : "";
    });
  }

  async _load3DVolume() {
    const sample = this.currentSample;
    if (!sample) return;
    const overlay = document.getElementById("three-overlay");
    overlay.classList.remove("hidden");
    try {
      const data = await fetch(`/api/volume3d/${sample.id}`).then(r => r.json());
      const container = document.getElementById("three-container");
      if (!this._renderer3d) this._renderer3d = new Volume3DRenderer(container);
      overlay.classList.add("hidden");
      this._renderer3d.loadPointCloud(data);
      document.getElementById("three-stats").innerHTML =
        `Sample ${sample.id}<br>${data.x.length.toLocaleString()} pts shown / ${data.n_total.toLocaleString()} total`;
      requestAnimationFrame(() => this._renderer3d?.resize());
      // Show viewing-direction arrow for current particle
      const o = this._orientations[this.currentParticle];
      if (o) this._renderer3d.setViewArrow(o.x, o.y, o.z);
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
        case "r": case "R": this.splitMode ? this._resetTransformSplit() : this._resetTransform(); break;
        case "f": case "F": this._fitToWindow(); break;
        case "+": case "=": this._zoomCenter(1.2); break;
        case "-": case "_": this._zoomCenter(1/1.2); break;
        case "3": this._switch3D(); break;
        case "2": this._switch2D(); break;
        case "s": case "S": this.splitMode ? this._deactivateSplit() : this._activateSplit(); break;
      }
    });
  }

  _setParticle(idx) {
    idx = Math.max(0, Math.min(this.nParticles - 1, idx));
    if (idx === this.currentParticle) return;
    this.currentParticle = idx;
    document.getElementById("particle-input").value = idx;
    this._updateScrubberUI();
    this._updateStatus();
    this._updateNoisySub();
    // Update sphere highlight and angle label
    this._sphereWidget?.highlight(idx);
    this._updateOrientationLabel(idx);
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
    this._updateScrubberUI();
    this._updateSliceNavVisibility();
    this._updateStatus();
    this._updateNoisySub();
  }

  _updateSliceNavVisibility() {
    // Z-slice controls are always visible; only hide the "↩ MIP" button when already in MIP mode
    const inSlice = (this.gtMode === "slice");
    const in3D    = (this.gtMode === "3d");
    document.getElementById("slice-nav")?.classList.toggle("hidden", in3D);
    document.getElementById("slice-sep")?.classList.toggle("hidden", in3D);
    document.getElementById("btn-back-mip")?.classList.toggle("hidden", !inSlice);
  }

  _updateGTBadge() {
    const badge = document.getElementById("gt-mode-badge");
    const sub   = document.getElementById("gt-sub");
    if (!badge) return;
    if (this.gtMode === "3d") {
      badge.textContent = "3D";
      badge.style.cssText = "background:rgba(78,158,255,.12);color:var(--accent);border-color:rgba(78,158,255,.3)";
      if (sub) sub.textContent = "Interactive 3D point cloud";
    } else if (this.gtMode === "slice") {
      badge.textContent = `Z=${this.currentSlice}`;
      badge.style.cssText = "background:rgba(67,197,158,.12);color:var(--gt-color);border-color:rgba(67,197,158,.25)";
      if (sub) sub.textContent = `Slice ${this.currentSlice} of ${this.nSlices-1}`;
    } else {
      badge.textContent = "MIP";
      badge.style.cssText = "background:rgba(67,197,158,.12);color:var(--gt-color);border-color:rgba(67,197,158,.25)";
      if (sub) sub.textContent = "Top-down projection of full volume";
    }
  }

  // ── Orientation sphere widget ─────────────────────────────────

  _bindOrientationWidget() {
    // Lazy-create the Three.js widget when first needed; just prep the container.
    // Actual widget creation happens in _loadOrientations after data arrives.
  }

  async _loadOrientations(sample) {
    const snr = this.currentSNR || sample.snr_levels?.[0];
    if (!snr) return;

    try {
      const data = await fetch(`/api/orientations/${sample.id}?snr=${encodeURIComponent(snr)}`).then(r => r.json());
      this._orientations = data.orientations || [];

      if (!this._orientations.length) return;

      // Create widget on first load
      if (!this._sphereWidget) {
        const container = document.getElementById("sphere-widget");
        this._sphereWidget = new ProjectionSphereWidget(container);
      }

      document.getElementById("sphere-widget").classList.remove("hidden");
      this._sphereWidget.setOrientations(this._orientations);
      this._sphereWidget.highlight(this.currentParticle);
      this._updateOrientationLabel(this.currentParticle);

    } catch (e) {
      console.warn("Orientations unavailable:", e);
    }
  }

  // Update noisy subtitle and GT 3D arrow for the given particle index.
  _updateOrientationLabel(idx) {
    const o = this._orientations[idx];
    if (!o) return;

    // Update noisy panel subtitle with angle info
    const sub = document.getElementById("noisy-sub");
    if (sub && this.currentSample) {
      const snrLabel = this.currentSNR?.replace("snr","") ?? "?";
      sub.textContent = `Proj. ${idx} / ${this.nParticles-1}  ·  SNR ${snrLabel}  ·  Tilt ${Math.round(o.tilt)}°`;
    }

    // Update arrow in GT 3D view if active
    if (this.gtMode === "3d" && this._renderer3d) {
      this._renderer3d.setViewArrow(o.x, o.y, o.z);
    }
  }

  // ── Zoom buttons ──────────────────────────────────────────────

  _bindZoomButtons() {
    const bind = (id, fn) => document.getElementById(id)?.addEventListener("click", fn);
    bind("btn-zoom-in",  () => this._zoomCenter(1.25));
    bind("btn-zoom-out", () => this._zoomCenter(1/1.25));
    bind("btn-reset",    () => this.splitMode ? this._resetTransformSplit() : this._resetTransform());
    bind("btn-fit",      () => this._fitToWindow());
  }

  // ── Helpers ───────────────────────────────────────────────────

  _setLoading(on) {
    document.getElementById("loading-overlay").classList.toggle("hidden", !on);
  }
}

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => { window.viewer = new CryoViewer(); });
