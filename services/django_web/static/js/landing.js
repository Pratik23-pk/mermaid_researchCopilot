(() => {
  const bgCanvas = document.getElementById("bgCanvas");
  const particleCanvas = document.getElementById("particleCanvas");
  const bgCtx = bgCanvas.getContext("2d");
  const pCtx = particleCanvas.getContext("2d");

  const modal = document.getElementById("authModal");
  const openButtons = document.querySelectorAll("[data-open-auth]");
  const closeBtn = document.getElementById("closeAuthBtn");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = {
    login: document.getElementById("loginTab"),
    signup: document.getElementById("signupTab"),
  };

  const mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2, vx: 0 };
  let rotY = 0;
  let rotX = -0.16;
  let lastParticleAt = 0;

  const spherePoints = [];
  const particles = [];

  function createSpherePoints() {
    spherePoints.length = 0;
    const count = 560;
    for (let i = 0; i < count; i += 1) {
      const u = Math.random();
      const v = Math.random();
      const theta = 2 * Math.PI * u;
      const phi = Math.acos(2 * v - 1);
      const x = Math.sin(phi) * Math.cos(theta);
      const y = Math.sin(phi) * Math.sin(theta);
      const z = Math.cos(phi);
      spherePoints.push({ x, y, z, glow: Math.random() });
    }
  }

  function resize() {
    bgCanvas.width = window.innerWidth;
    bgCanvas.height = window.innerHeight;
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
  }

  function rotatePoint(p, ax, ay) {
    const cosY = Math.cos(ay);
    const sinY = Math.sin(ay);
    const cosX = Math.cos(ax);
    const sinX = Math.sin(ax);

    let x = p.x * cosY - p.z * sinY;
    let z = p.x * sinY + p.z * cosY;
    let y = p.y;

    const y2 = y * cosX - z * sinX;
    const z2 = y * sinX + z * cosX;

    return { x, y: y2, z: z2 };
  }

  function renderGlobe() {
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);

    const cx = bgCanvas.width * 0.5;
    const cy = bgCanvas.height * 0.54;
    const radius = Math.min(bgCanvas.width, bgCanvas.height) * 0.28;

    bgCtx.save();
    bgCtx.globalAlpha = 0.28;
    const ring = bgCtx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.25);
    ring.addColorStop(0, "rgba(200, 173, 130, 0.2)");
    ring.addColorStop(1, "rgba(16, 23, 34, 0)");
    bgCtx.fillStyle = ring;
    bgCtx.beginPath();
    bgCtx.arc(cx, cy, radius * 1.25, 0, Math.PI * 2);
    bgCtx.fill();
    bgCtx.restore();

    for (const p of spherePoints) {
      const r = rotatePoint(p, rotX, rotY);
      const perspective = 1 / (1.6 - r.z);
      const px = cx + r.x * radius * perspective;
      const py = cy + r.y * radius * perspective;
      const size = Math.max(0.4, 1.6 * perspective);
      const alpha = Math.min(0.86, Math.max(0.07, 0.08 + perspective * 0.52));

      bgCtx.fillStyle = `rgba(171, 190, 206, ${alpha})`;
      bgCtx.beginPath();
      bgCtx.arc(px, py, size, 0, Math.PI * 2);
      bgCtx.fill();

      if (Math.random() < 0.03 + p.glow * 0.02) {
        bgCtx.fillStyle = `rgba(211, 178, 126, ${alpha * 0.55})`;
        bgCtx.beginPath();
        bgCtx.arc(px, py, size * 0.55, 0, Math.PI * 2);
        bgCtx.fill();
      }
    }

    const shadow = bgCtx.createRadialGradient(cx, cy + radius * 0.36, radius * 0.1, cx, cy + radius * 0.36, radius * 1.4);
    shadow.addColorStop(0, "rgba(6, 11, 24, 0.35)");
    shadow.addColorStop(1, "rgba(6, 11, 24, 0)");
    bgCtx.fillStyle = shadow;
    bgCtx.beginPath();
    bgCtx.ellipse(cx, cy + radius * 0.36, radius * 0.95, radius * 0.28, 0, 0, Math.PI * 2);
    bgCtx.fill();
  }

  function renderParticles() {
    pCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);

    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.002;
      p.life -= 1;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      pCtx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${p.life / p.maxLife})`;
      pCtx.beginPath();
      pCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      pCtx.fill();
    }
  }

  function animate() {
    rotY += 0.0007 + mouse.vx * 0.00003;
    rotX += (mouse.y / window.innerHeight - 0.5) * 0.00025;
    mouse.vx *= 0.92;

    renderGlobe();
    renderParticles();

    requestAnimationFrame(animate);
  }

  function spawnParticles(x, y) {
    const now = performance.now();
    if (now - lastParticleAt < 24) {
      return;
    }
    lastParticleAt = now;

    const count = 2;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        life: 24 + Math.random() * 8,
        maxLife: 30,
        size: 0.8 + Math.random() * 1.2,
        r: 186 + Math.floor(Math.random() * 26),
        g: 164 + Math.floor(Math.random() * 22),
        b: 136 + Math.floor(Math.random() * 18),
      });
    }
  }

  function setActiveTab(tab) {
    tabButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    });
    Object.keys(tabPanels).forEach((key) => {
      tabPanels[key].classList.toggle("active", key === tab);
    });
  }

  openButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modal.classList.remove("hidden");
      setActiveTab(btn.dataset.authTab || "signup");
    });
  });

  closeBtn?.addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  modal?.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.add("hidden");
    }
  });

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });

  window.addEventListener("mousemove", (e) => {
    const dx = e.clientX - mouse.x;
    mouse.vx = dx;
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    spawnParticles(e.clientX, e.clientY);
  });

  window.addEventListener("resize", resize);

  resize();
  createSpherePoints();
  animate();
})();
