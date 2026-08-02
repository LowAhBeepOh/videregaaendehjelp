(function () {
  'use strict';

  const STORAGE_KEY = 'videregaaendehjelp:profile';
  const DB_NAME = 'vhjelp-profile';
  const STORE = 'avatars';
  const AVATAR_ID = 'me';

  // Subset of the :root palette. Backgrounds paired with a foreground that
  // passes WCAG AA contrast for >=14px bold text. The foreground isn't used
  // directly — textFgFor() picks white or #000 at runtime based on the hash,
  // but the explicit pairing here documents intent.
  const PALETTE = [
    { bg: '#5865F2', fg: '#fff' }, // --card-blue
    { bg: '#30D158', fg: '#000' }, // --card-green  (bright green needs dark text)
    { bg: '#3a225d', fg: '#fff' }, // --card-purple (deep)
    { bg: '#d40b00', fg: '#fff' }, // --card-red
    { bg: '#FF9F0A', fg: '#000' }, // --card-orange (bright needs dark text)
    { bg: '#FFD60A', fg: '#000' }, // --card-yellow (fails 4.5:1 with white)
  ];

  // Fresh users start with no trinn selected — "Ingen trinn" (state, not its
  // own label). The form starts with no radio checked, and pressing save
  // persists grade=null. The .grade-pill element simply hides in this state.
  const DEFAULT_GRADE = null;
  // Pre-update trinn-radios used the en-dash string for VG1–VG3. Map that
  // legacy value to a sensible current trinn so existing profiles still work
  // even when DEFAULT_GRADE itself is null.
  const LEGACY_GRADE = 'VG1\u2013VG3';

  // Per-trinn colors for .grade-pill (bg + text contrast). Values reference
  // the page's :root palette tokens so the pill stays in sync with the rest
  // of the design system. Alumni/Lærer are not real trinn in the school sense,
  // but they're unrolled here so the grade-pill renders consistently with
  // the radio-options on profile.html.
  const GRADE_STYLES = {
    '8-10':    { bg: 'var(--card-lime)',   fg: '#000' },
    'VG1-VG2': { bg: 'var(--card-blue)',   fg: '#fff' },
    'Alumni':  { bg: 'var(--card-orange)', fg: '#000' },
    'Lærer':   { bg: 'var(--card-red)',    fg: '#fff' },
  };

  // Grade → tint color for the body-bg overlay. Mirrors the bright card
  // palette so the tint literally broadcasts "this is your trinn".
  // 'Ingen trinn' (grade === null) yields no tint at all, per spec.
  const GRADE_TINT = {
    '8-10':    '#D0F54E',
    'VG1-VG2': '#5865F2',
    'Alumni':  '#FF9F0A',
    'Lærer':   '#d40b00',
  };

  // Tint intensity presets. Opacities were picked to keep lime (#D0F54E) and
  // bright orange acceptable when blended over white bg (light mode) — 6%
  // stays pastel, 12% keeps the grade identity readable without washing out
  // muted text colors. 'Av' resolves to 0 (no tint) and is treated identically
  // to 'no grade selected' by applyTint().
  const TINT_PRESETS = {
    off:    { alpha: 0,    label: 'Av' },
    subtle: { alpha: 0.06, label: 'Svak' },
    strong: { alpha: 0.12, label: 'Sterk' },
  };
  const DEFAULT_TINT_INTENSITY = 'subtle';
  const DEFAULT_TINT_SCOPE = 'bg';

  // Shared grade → picker-color name map. Used by both paintThemePicker()
  // and paintTintUI() on profile.html so a future trinn only needs to be
  // added in one place. Same shape as the existing inline GRADE_TO_PICKER_COLOR
  // constant in profile.html — keep them in sync by treating the version on
  // Assets/js/profile.js as the source of truth.
  const GRADE_TO_PICKER_COLOR = {
    '8-10': 'lime',
    'VG1-VG2': 'blue',
    'Alumni': 'orange',
    'Lærer': 'red',
  };

  // Base palette values used by applyTint to rebuild tinted variants on theme
  // switches. Mirror of what's declared on :root / [data-theme="light"] in
  // each page's <style>; keeping a copy here means applyTint can override
  // them on documentElement.style without needing a per-page loader script.
  const BASE_PALETTE = {
    dark:  { bg: '#050505', surface: '#1c1c1e', border: '#2c2c2e' },
    light: { bg: '#ffffff', surface: '#f2f2f7', border: '#e5e5ea' },
  };

  function defaultProfile() {
    return {
      name: 'Bruker',
      grade: null,
      county: null,
      username: null,
      handle: null,
      linked: false,
      displayGrade: null,
      fag: null,
      // Tint is purely cosmetic but follows the user so they don't have to
      // re-pick a preset on each device. Default = subtle + body-only so a
      // fresh build feels gentle-tinted by their grade.
      tintIntensity: 'subtle',
      tintScope: 'bg',
      syncThemeDevices: false,
      customTintColor: null,
    };
  }

  function get() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultProfile();
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return defaultProfile();
      const merged = Object.assign(defaultProfile(), parsed);
      // Old profiles used the en-dash string for VG1–VG3 — remap to a current
      // trinn so the rest of the app keeps working. We map explicitly (not via
      // DEFAULT_GRADE) so the legacy path is independent of the fresh-user state.
      if (merged.grade === LEGACY_GRADE) merged.grade = 'VG1-VG2';
      // Legacy Meninger roles are normalized into the unified 4-value trinn model.
      if (merged.grade === 'ungdom' || merged.grade === 'elev') merged.grade = '8-10';
      if (merged.grade === 'vg1' || merged.grade === 'vg2' || merged.grade === 'vg3') merged.grade = 'VG1-VG2';
      if (merged.grade === 'hoyskole') merged.grade = 'Alumni';
      if (merged.grade === 'laerer') merged.grade = 'Lærer';
      return merged;
    } catch {
      return defaultProfile();
    }
  }

  function set(patch) {
    const next = Object.assign({}, get(), patch || {});
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    document.dispatchEvent(new CustomEvent('vhjelp:profile-updated', { detail: next }));
    return next;
  }

  function clear() {
    localStorage.removeItem(STORAGE_KEY);
    if ('indexedDB' in window) {
      try {
        indexedDB.deleteDatabase(DB_NAME);
      } catch {
        // ignore
      }
    }
    document.dispatchEvent(new CustomEvent('vhjelp:profile-updated', { detail: defaultProfile() }));
  }

  function initialsOf(name) {
    if (!name) return 'B';
    const parts = String(name).trim().split(/\s+/).slice(0, 2);
    return parts.map((w) => (w[0] || '').toUpperCase()).join('') || 'B';
  }

  function hashCode(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = (h << 5) - h + s.charCodeAt(i);
      h |= 0;
    }
    return Math.abs(h);
  }

  // Apply the user's stored theme to <html data-theme="...">. The value
  // stored in localStorage is 'system' | 'light' | 'dark' but only the latter
  // two make sense for the [data-theme="light"] CSS rule — 'system' has to be
  // resolved to the user's prefers-color-scheme once we have access to the
  // matchMedia API. Called once on script load (via renderAll) and again any
  // time the saved value changes or the OS preference flips.
  function applyTheme() {
    // Secret themes force light mode — don't override them.
    if (localStorage.getItem('vhjelp:frutiger') === '1' ||
        localStorage.getItem('vhjelp:newspaper') === '1') {
      return;
    }
    const stored = localStorage.getItem('theme') || 'system';
    const prefersDark =
      window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const resolved = stored === 'system' ? (prefersDark ? 'dark' : 'light') : stored;
    if (document.documentElement.getAttribute('data-theme') !== resolved) {
      document.documentElement.setAttribute('data-theme', resolved);
    }
    document.dispatchEvent(
      new CustomEvent('vhjelp:theme-changed', { detail: { stored, resolved } })
    );
    return resolved;
  }

  // Re-apply when the stored theme changes (catch profile.html picker saves
  // from another tab) and when the OS preference flips while 'system' is set.
  window.addEventListener('storage', function (e) {
    if (e.key === 'theme') applyTheme();
  });
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
      if ((localStorage.getItem('theme') || 'system') === 'system') applyTheme();
    });
  }

  function initialsColor(name) {
    const entry = PALETTE[hashCode(String(name || '')) % PALETTE.length];
    return entry.bg;
  }

  // WCAG-style relative luminance. Picked from the sRGB→linear formula.
  // Returns 0 (black) – 1 (white). Threshold 0.5 keeps the decision stable
  // across the palette: dark bg gets white text, bright bg gets black.
  function luminance(hex) {
    const c = hex.replace('#', '');
    if (c.length !== 6) return 0;
    const r = parseInt(c.slice(0, 2), 16) / 255;
    const g = parseInt(c.slice(2, 4), 16) / 255;
    const b = parseInt(c.slice(4, 6), 16) / 255;
    const lin = (v) => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  }

  function textFgFor(bgHex) {
    return luminance(bgHex) > 0.55 ? '#000' : '#fff';
  }

  // IndexedDB helpers (avatar storage — localStorage is too small for base64 blobs).
  function openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) {
          req.result.createObjectStore(STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  function getAvatarBlob() {
    if (!('indexedDB' in window)) return Promise.resolve(null);
    return openDb().then(
      (db) =>
        new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readonly');
          const getReq = tx.objectStore(STORE).get(AVATAR_ID);
          getReq.onsuccess = () => resolve(getReq.result || null);
          getReq.onerror = () => reject(getReq.error);
        })
    ).catch(() => null);
  }

  function setAvatarBlob(blob) {
    if (!('indexedDB' in window) || !blob) return Promise.resolve();
    return openDb().then(
      (db) =>
        new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readwrite');
          tx.objectStore(STORE).put(blob, AVATAR_ID);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
        })
    );
  }

  function clearAvatarBlob() {
    if (!('indexedDB' in window)) return Promise.resolve();
    return openDb().then((db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).delete(AVATAR_ID);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      })
    ).catch(() => {});
  }

  function exportJson() {
    return JSON.stringify({ version: 1, profile: get() }, null, 2);
  }

  function importJson(text) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error('Ugyldig JSON');
    }
    if (!parsed || typeof parsed !== 'object' || !parsed.profile) {
      throw new Error('Ugyldig format');
    }
    set(parsed.profile);
  }

  function resolveGrade(raw) {
    if (!raw) return null;
    return raw === LEGACY_GRADE ? 'VG1-VG2' : raw;
  }

  // For Alumni / Lærer profiles the user can manually override which trinn
  // content they want to view via displayGrade. Plain 8-10/VG1-VG2 profiles
  // ignore displayGrade entirely.
  function resolveShowGrade(profile) {
    // The .grade-pill always reflects the user's stored trinn, regardless of
    // the trinn-toggle's displayGrade override on learn.html. The toggle can
    // still set displayGrade and mark its active button, but the pill itself
    // stays anchored to profile.grade so clicking a toggle doesn't visually
    // rewrite the user's identity.
    return profile.grade;
  }

  function setDisplayGrade(value) {
    const next = value === '8-10' || value === 'VG1-VG2' ? value : null;
    set({ displayGrade: next });
  }

  // Track the object URL bound to each .profile-pill element so the previous
  // one can be revoked on the next render. Without this, every profile update
  // leaks an Object URL until tab close.
  const elementUrls = new Map();

  // True when the user has actually saved a profile (vs. the unconfigured
  // placeholder state, which renders a different visual).
  function isProfileConfigured(profile) {
    return !!profile.name && profile.name !== 'Bruker';
  }

  async function renderHeader() {
    const pills = document.querySelectorAll('.profile-pill');
    if (pills.length === 0) return;
    const profile = get();
    const configured = isProfileConfigured(profile);
    const color = initialsColor(profile.name);
    const fg = textFgFor(color);
    const blob = await getAvatarBlob().catch(() => null);

    pills.forEach((pill) => {
      const prev = elementUrls.get(pill);
      if (prev) {
        URL.revokeObjectURL(prev);
        elementUrls.delete(pill);
      }

      pill.innerHTML = '';
      pill.setAttribute('aria-label', `Min profil: ${profile.name}`);
      pill.setAttribute('title', profile.name);

      if (blob) {
        const img = document.createElement('img');
        const url = URL.createObjectURL(blob);
        elementUrls.set(pill, url);
        img.src = url;
        img.alt = `Profilbilde for ${profile.name}`;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.borderRadius = '50%';
        pill.appendChild(img);
        return;
      }

      if (!configured) {
        // Default avatar for users who haven't created a profile yet.
        // Assets/DefaultMe.png is supplied by the project; if it's missing
        // (404), the error handler swaps in 'E' so the pill never breaks.
        pill.style.background = 'var(--card-blue)';
        pill.style.color = '#fff';
        const img = document.createElement('img');
        img.alt = 'Standard avatar';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.addEventListener(
          'error',
          () => {
            if (pill.firstChild === img) pill.removeChild(img);
            const span = document.createElement('span');
            span.textContent = 'E';
            pill.appendChild(span);
          },
          { once: true }
        );
        pill.appendChild(img);
        img.src = 'Assets/DefaultMe.png';
        return;
      }

      pill.style.background = color;
      pill.style.color = fg;
      const span = document.createElement('span');
      span.textContent = initialsOf(profile.name);
      pill.appendChild(span);
    });
  }

  function renderGradePill() {
    const pills = document.querySelectorAll('.grade-pill');
    if (pills.length === 0) return;
    const effective = resolveShowGrade(get());
    pills.forEach((p) => {
      const style = effective ? GRADE_STYLES[effective] : null;
      if (!style) {
        // No trinn selected (Ingen trinn) — render a muted dashed outline
        // pill so the state is explicit instead of silent. Using a transparent
        // bg + dashed muted border keeps the pill's outer dimensions stable
        // across states (no 1 px layout shift when toggling to a real trinn).
        p.hidden = false;
        p.textContent = 'Ingen trinn';
        p.style.background = 'transparent';
        p.style.color = 'var(--text-muted)';
        p.style.border = '1px dashed var(--text-muted)';
        return;
      }
      p.hidden = false;
      p.textContent = effective;
      p.style.background = style.bg;
      p.style.color = style.fg;
      p.style.border = '1px solid transparent';
    });
  }

  // Secret theme CSS + font injection helpers
  const THEMES = {
    frutiger: {
      css: 'Assets/css/frutiger-aero.css',
      fonts: ['https://fonts.googleapis.com/css2?family=PT+Sans:ital,wght@0,400;0,700;1,400;1,700&display=swap'],
      attr: { name: 'data-frutiger', value: 'aero' },
      storage: 'vhjelp:frutiger',
    },
    newspaper: {
      css: 'Assets/css/newspaper.css',
      fonts: [
        'https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700;900&display=swap',
        'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&display=swap',
      ],
      attr: { name: 'data-newspaper', value: 'true' },
      storage: 'vhjelp:newspaper',
    },
  };

  function injectLink(id, href, rel) {
    if (document.getElementById(id)) return;
    const link = document.createElement('link');
    link.id = id;
    link.rel = rel || 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  }

  function removeLink(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function applySecretThemes() {
    let anyActive = false;

    for (const key of Object.keys(THEMES)) {
      const t = THEMES[key];
      const enabled = localStorage.getItem(t.storage) === '1';
      const cssId = 'vhjelp-theme-css-' + key;
      const fontIds = t.fonts.map((_, i) => 'vhjelp-theme-font-' + key + '-' + i);

      if (enabled) {
        anyActive = true;
        document.documentElement.setAttribute(t.attr.name, t.attr.value);
        injectLink(cssId, t.css);
        t.fonts.forEach((href, i) => injectLink(fontIds[i], href));
      } else {
        document.documentElement.removeAttribute(t.attr.name);
        removeLink(cssId);
        fontIds.forEach(removeLink);
      }
    }

    // All secret themes are light-mode only — force light when any is active.
    // When none are active, let applyTheme() restore the user's preference.
    if (anyActive) {
      document.documentElement.setAttribute('data-theme', 'light');
    }
  }

  function renderAll() {
    applyTheme();
    applySecretThemes();
    applyTint();
    renderHeader();
    renderGradePill();
  }

  // Re-render whenever any page on the site updates the profile.
  document.addEventListener('vhjelp:profile-updated', renderAll);

  // Tint helpers — gradient-free. Instead of a `body::before` overlay (which
  // fights other stacking contexts and gets weird with sticky navigations),
  // applyTint() rewrites --bg / --surface on documentElement.style so the
  // tint lives inside the same cascade that already powers every page.
  function hexToRgb(hex) {
    const c = String(hex || '').replace('#', '');
    if (c.length !== 6) return null;
    const n = parseInt(c, 16);
    if (Number.isNaN(n)) return null;
    return { r: (n >> 16) & 0xff, g: (n >> 8) & 0xff, b: n & 0xff };
  }

  function rgbToHex(rgb) {
    return (
      '#' +
      [rgb.r, rgb.g, rgb.b]
        .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0'))
        .join('')
    );
  }

  function lerpRgb(a, b, t) {
    return {
      r: a.r + (b.r - a.r) * t,
      g: a.g + (b.g - a.g) * t,
      b: a.b + (b.b - a.b) * t,
    };
  }

  function currentBasePalette() {
    const theme = document.documentElement.getAttribute('data-theme') || 'dark';
    return BASE_PALETTE[theme] || BASE_PALETTE.dark;
  }

  function applyTint() {
    const profile = get();
    const base = currentBasePalette();
    // Always start from the un-tinted base. applyTint() must be idempotent —
    // if the user previously picked "Sterk / Alt" and now selects "Av", the
    // page should snap back to the original palette, not stay tinted.
    let bg = base.bg;
    let surface = base.surface;
    let border = base.border;

    const intensity = profile.tintIntensity || DEFAULT_TINT_INTENSITY;
    const scope = profile.tintScope || DEFAULT_TINT_SCOPE;
    // Plus users may override the tint color; fall back to grade-based tint.
    const isPlus = window.VHplus && VHplus.isActive();
    const tintHex = (isPlus && profile.customTintColor) || (profile.grade && GRADE_TINT[profile.grade]);
    const preset = TINT_PRESETS[intensity];
    if (tintHex && preset && preset.alpha > 0) {
      const tintRgb = hexToRgb(tintHex);
      const bgRgb = hexToRgb(base.bg);
      const surfRgb = hexToRgb(base.surface);
      const borderRgb = hexToRgb(base.border);
      if (tintRgb && bgRgb) {
        bg = rgbToHex(lerpRgb(bgRgb, tintRgb, preset.alpha));
      }
      // 'all' scope extends the lerp to surface + border. Slightly weaker
      // amp on the border keeps card outlines subtle so they don't compete
      // with the surface fill.
      if (scope === 'all' && tintRgb && surfRgb) {
        surface = rgbToHex(lerpRgb(surfRgb, tintRgb, preset.alpha * 0.85));
        border = rgbToHex(lerpRgb(borderRgb, tintRgb, preset.alpha * 0.7));
      }
    }

    // Inline overrides take precedence over the :root / [data-theme] tokens
    // defined per page, so the rest of the cascade picks up the new values
    // automatically without us touching any per-page CSS.
    const root = document.documentElement.style;
    root.setProperty('--bg', bg);
    root.setProperty('--surface', surface);
    if (scope === 'all' && tintHex && preset && preset.alpha > 0) {
      root.setProperty('--border', border);
    } else {
      // Drop the override so the per-page --border token is used again.
      root.removeProperty('--border');
    }

    document.dispatchEvent(
      new CustomEvent('vhjelp:tint-changed', {
        detail: { intensity, scope, bg, surface, border },
      })
    );
  }

  function setTint(intensity, scope, customTintColor) {
    const validIntensities = Object.keys(TINT_PRESETS);
    const validScopes = ['bg', 'all'];
    const next = {
      tintIntensity: validIntensities.includes(intensity) ? intensity : DEFAULT_TINT_INTENSITY,
      tintScope: validScopes.includes(scope) ? scope : DEFAULT_TINT_SCOPE,
    };
    if (customTintColor !== undefined) {
      next.customTintColor = customTintColor || null;
    }
    // Persist inside the existing profile JSON so the setting roams with the
    // user's identity. We deliberately do NOT dispatch vhjelp:profile-updated
    // here — a tint tweak shouldn't re-run renderHeader / renderGradePill
    // (which would also revoke / re-create the avatar blob URL).
    const current = get();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.assign({}, current, next)));
    applyTint();
    return next;
  }

  // Re-tint whenever the OS / page theme flips, because the base palette
  // values (--bg, --surface) are tied to the active theme.
  document.addEventListener('vhjelp:theme-changed', applyTint);
  // Re-tint when Plus status changes (custom color may become active/inactive).
  document.addEventListener('vhjelp:plus-changed', applyTint);

  window.VHprofile = {
    get,
    set,
    clear,
    initialsOf,
    initialsColor,
    textFgFor,
    applyTheme,
    applyTint,
    setTint,
    getAvatarBlob,
    setAvatarBlob,
    clearAvatarBlob,
    exportJson,
    importJson,
    renderHeader,
    renderGradePill,
    renderAll,
    resolveGrade,
    resolveShowGrade,
    setDisplayGrade,
    isProfileConfigured,
    TINT_PRESETS,
    GRADE_TINT,
    GRADE_TO_PICKER_COLOR,
  };

  // Secret themes — console-only easter eggs.
  // Activating one automatically deactivates the others (mutual exclusion).
  function activateSecretTheme(key) {
    for (const k of Object.keys(THEMES)) {
      if (k === key) {
        localStorage.setItem(THEMES[k].storage, '1');
      } else {
        localStorage.removeItem(THEMES[k].storage);
      }
    }
    applySecretThemes();
  }

  function deactivateAllSecretThemes() {
    for (const k of Object.keys(THEMES)) {
      localStorage.removeItem(THEMES[k].storage);
    }
    applySecretThemes();
    applyTheme();
  }

  window.fun = window.fun || {};
  window.fun.secret = window.fun.secret || {};

  window.fun.secret.Frutiger = function (state) {
    if (state === 1) {
      activateSecretTheme('frutiger');
      console.log('%c Frutiger Aero aktivert ', 'background: #9fd4f0; color: #0d3b5c; font-weight: bold; padding: 4px 8px; border-radius: 4px;');
    } else {
      deactivateAllSecretThemes();
      console.log('%c Frutiger Aero deaktivert ', 'background: #e0e0e0; color: #333; font-weight: bold; padding: 4px 8px; border-radius: 4px;');
    }
  };

  window.fun.secret.Newspaper = function (state) {
    if (state === 1) {
      activateSecretTheme('newspaper');
      console.log('%c Newspaper aktivert ', 'background: #f4f1ea; color: #111; font-weight: bold; padding: 4px 8px; border-radius: 0; border: 1px solid #111;');
    } else {
      deactivateAllSecretThemes();
      console.log('%c Newspaper deaktivert ', 'background: #e0e0e0; color: #333; font-weight: bold; padding: 4px 8px; border-radius: 4px;');
    }
  };



  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAll);
  } else {
    renderAll();
  }
})();
