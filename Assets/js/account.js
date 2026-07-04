(function () {
  'use strict';

  const SUPABASE_URL = 'https://wqmkwcrlvgfczbueseip.supabase.co';
  const SUPABASE_ANON = 'sb_publishable_XCDS-VsKHGbOXaaiUWaHIg_sH3sCfLJ';
  const MAX_AVATAR_BYTES = 1024 * 1024;
  const PENDING_PROFILE_KEY = 'meninger_pending_profile';
  const BOOKMARKS_KEY = 'bookmarks';

  const ROLE_OPTIONS = [
    { value: '8-10', label: '8-10. klasse' },
    { value: 'VG1-VG2', label: 'VG1-VG2' },
    { value: 'Alumni', label: 'Alumni' },
    { value: 'Lærer', label: 'Lærer' },
  ];

  const COUNTY_OPTIONS = [
    '',
    'Østfold', 'Akershus', 'Oslo', 'Innlandet', 'Buskerud', 'Vestfold', 'Telemark',
    'Agder', 'Rogaland', 'Vestland', 'Møre og Romsdal', 'Trøndelag', 'Nordland',
    'Troms', 'Finnmark'
  ];

  function escHtml(str) {
    return String(str == null ? '' : str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function normalizeRole(role) {
    const value = String(role || '').trim().toLowerCase();
    if (!value) return '8-10';

    if (value === '8-10') return '8-10';
    if (value === 'vg1-vg2') return 'VG1-VG2';
    if (value === 'alumni') return 'Alumni';
    if (value === 'lærer' || value === 'laerer') return 'Lærer';

    if (value === 'ungdom' || value === 'elev') return '8-10';
    if (value === 'vg1' || value === 'vg2' || value === 'vg3') return 'VG1-VG2';
    if (value === 'hoyskole') return 'Alumni';
    return '8-10';
  }

  function displayRole(role) {
    return normalizeRole(role);
  }

  function normalizeHandle(value, fallbackName) {
    const source = String(value || '').trim() || String(fallbackName || '').trim();
    const cleaned = source
      .toLowerCase()
      .replace(/^@+/, '')
      .replace(/[^a-z0-9._-]+/g, '')
      .slice(0, 25);
    return '@' + (cleaned || 'bruker');
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async function dataUrlToBlob(dataUrl) {
    if (!dataUrl || typeof dataUrl !== 'string' || !/^data:image\//i.test(dataUrl)) return null;
    const response = await fetch(dataUrl);
    return response.blob();
  }

  function loadImageFromDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  async function processAvatarFile(file, allowGif) {
    if (!file) return null;
    const mime = String(file.type || '').toLowerCase();
    const isGif = mime === 'image/gif';

    if (isGif && !allowGif) {
      throw new Error('GIF er kun tillatt for moderator/admin.');
    }
    if (!/^image\/(png|jpe?g|webp|gif)$/i.test(mime)) {
      throw new Error('Kun PNG, JPG, WEBP eller GIF er støttet.');
    }

    if (isGif) {
      if (file.size > MAX_AVATAR_BYTES) {
        throw new Error('GIF må være maks 1MB.');
      }
      return blobToDataUrl(file);
    }

    const originalDataUrl = await blobToDataUrl(file);
    const img = await loadImageFromDataUrl(originalDataUrl);

    let width = img.naturalWidth;
    let height = img.naturalHeight;
    const maxSide = 512;
    if (Math.max(width, height) > maxSide) {
      const scale = maxSide / Math.max(width, height);
      width = Math.max(1, Math.round(width * scale));
      height = Math.max(1, Math.round(height * scale));
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    const tryEncode = async function (type, quality) {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, type, quality));
      if (!blob) return null;
      return { size: blob.size, dataUrl: await blobToDataUrl(blob) };
    };

    let best = null;
    const tries = [
      ['image/webp', 0.88],
      ['image/webp', 0.76],
      ['image/jpeg', 0.84],
      ['image/jpeg', 0.72],
      ['image/jpeg', 0.62],
    ];

    for (const [type, quality] of tries) {
      const out = await tryEncode(type, quality);
      if (!out) continue;
      if (!best || out.size < best.size) best = out;
      if (out.size <= MAX_AVATAR_BYTES) return out.dataUrl;
    }

    if (best && best.size <= MAX_AVATAR_BYTES) return best.dataUrl;
    throw new Error('Bildet ble fortsatt over 1MB. Prøv et mindre bilde.');
  }

  const sb = (window.supabase && window.supabase.createClient)
    ? window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON)
    : null;

  let currentUser = null;
  let currentProfile = null;

  function normalizeBookmarks(raw) {
    if (!Array.isArray(raw)) return [];
    return raw
      .map((v) => String(v || '').trim())
      .filter((v) => /^Guides\//i.test(v));
  }

  function normalizeSyncData(raw) {
    const src = raw && typeof raw === 'object' ? raw : {};
    const out = {
      tintIntensity: ['off', 'subtle', 'strong'].includes(src.tintIntensity) ? src.tintIntensity : null,
      tintScope: ['bg', 'all'].includes(src.tintScope) ? src.tintScope : null,
      syncThemeDevices: !!src.syncThemeDevices,
      theme: ['system', 'light', 'dark'].includes(src.theme) ? src.theme : null,
    };
    return out;
  }

  function normalizeProfile(profile) {
    if (!profile) return null;
    const normalizedBookmarks = normalizeBookmarks(profile.bookmarks);
    const normalizedSyncData = normalizeSyncData(profile.sync_data);
    return {
      id: profile.id,
      display_name: String(profile.display_name || '').trim(),
      handle: normalizeHandle(profile.handle, profile.display_name),
      avatar_url: String(profile.avatar_url || '').trim(),
      role: normalizeRole(profile.role),
      county: profile.county || '',
      mod_role: profile.mod_role || 'user',
      warnings: Number(profile.warnings || 0),
      timeout_until: profile.timeout_until || null,
      banned: !!profile.banned,
      bookmarks: normalizedBookmarks,
      sync_data: normalizedSyncData,
    };
  }

  function getLocalBookmarks() {
    try {
      return normalizeBookmarks(JSON.parse(localStorage.getItem(BOOKMARKS_KEY) || '[]'));
    } catch {
      return [];
    }
  }

  function setLocalBookmarks(bookmarks, dispatchEvent) {
    const next = normalizeBookmarks(bookmarks);
    localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(next));
    if (dispatchEvent !== false) {
      document.dispatchEvent(new CustomEvent('vhjelp:bookmarks-updated', { detail: { bookmarks: next } }));
    }
    return next;
  }

  function getLocalSyncSettings() {
    const profile = window.VHprofile && VHprofile.get ? VHprofile.get() : {};
    const theme = localStorage.getItem('theme') || 'system';
    return {
      theme: ['system', 'light', 'dark'].includes(theme) ? theme : 'system',
      tintIntensity: profile.tintIntensity || 'subtle',
      tintScope: profile.tintScope || 'bg',
      syncThemeDevices: !!profile.syncThemeDevices,
    };
  }

  function applyRemoteSyncToLocal(profile) {
    if (!profile || !window.VHprofile) return;
    const syncData = normalizeSyncData(profile.sync_data);
    const patch = {};
    if (syncData.tintIntensity) patch.tintIntensity = syncData.tintIntensity;
    if (syncData.tintScope) patch.tintScope = syncData.tintScope;
    patch.syncThemeDevices = !!syncData.syncThemeDevices;
    if (Object.keys(patch).length) {
      window.VHprofile.set(patch);
    }

    if (syncData.syncThemeDevices && syncData.theme && ['system', 'light', 'dark'].includes(syncData.theme)) {
      localStorage.setItem('theme', syncData.theme);
      if (window.VHprofile && VHprofile.applyTheme) VHprofile.applyTheme();
    }

    const remoteBookmarks = normalizeBookmarks(profile.bookmarks);
    if (profile.bookmarks != null) {
      setLocalBookmarks(remoteBookmarks, true);
    }
  }

  async function safeUpsertProfile(payload) {
    const full = Object.assign({}, payload);
    let res = await sb.from('profiles').upsert(full, { onConflict: 'id' });
    if (!res.error) return res;

    // Backward compatibility: if DB migration for sync columns hasn't run yet,
    // retry without sync_data/bookmarks so core profile edits still work.
    const msg = String(res.error.message || '').toLowerCase();
    const missingSyncCols = msg.includes('sync_data') || msg.includes('bookmarks') || msg.includes('column');
    if (!missingSyncCols) return res;

    const fallback = Object.assign({}, full);
    delete fallback.sync_data;
    delete fallback.bookmarks;
    res = await sb.from('profiles').upsert(fallback, { onConflict: 'id' });
    return res;
  }

  async function loadProfile(userId) {
    if (!sb || !userId) return null;
    const res = await sb.from('profiles').select('*').eq('id', userId).single();
    if (res.error) return null;
    return normalizeProfile(res.data);
  }

  async function applyPendingProfile(userId) {
    if (!sb || !userId) return;
    const raw = localStorage.getItem(PENDING_PROFILE_KEY);
    if (!raw) return;
    try {
      const pending = JSON.parse(raw);
      if (!pending || !pending.display_name) return;

      const existing = await loadProfile(userId);
      const payload = {
        id: userId,
        display_name: String(pending.display_name || existing && existing.display_name || 'Bruker').trim(),
        handle: normalizeHandle(pending.handle, pending.display_name || existing && existing.display_name),
        avatar_url: pending.avatar_url || existing && existing.avatar_url || null,
        role: normalizeRole(pending.role || existing && existing.role),
        county: pending.county || existing && existing.county || null,
        mod_role: existing && existing.mod_role || 'user',
      };

      const upsertRes = await safeUpsertProfile(payload);
      if (!upsertRes.error) localStorage.removeItem(PENDING_PROFILE_KEY);
    } catch {
      // ignore malformed pending payload
    }
  }

  async function syncProfileToLocal(profile) {
    if (!profile || !window.VHprofile) return;
    const patch = {
      linked: true,
      username: profile.display_name || null,
      handle: profile.handle || null,
      grade: normalizeRole(profile.role),
      county: profile.county || null,
    };
    window.VHprofile.set(patch);

    if (profile.avatar_url && /^data:image\//i.test(profile.avatar_url)) {
      const blob = await dataUrlToBlob(profile.avatar_url).catch(() => null);
      if (blob) {
        await window.VHprofile.setAvatarBlob(blob).catch(() => null);
      }
    }
    if (!profile.avatar_url) {
      await window.VHprofile.clearAvatarBlob().catch(() => null);
    }
    applyRemoteSyncToLocal(profile);
  }

  function dispatchAccountChange() {
    document.dispatchEvent(new CustomEvent('vhjelp:account-changed', {
      detail: {
        user: currentUser,
        profile: currentProfile,
        linked: !!(currentUser && currentProfile),
      },
    }));
  }

  async function refreshAuth() {
    if (!sb) return { user: null, profile: null };
    const sessionRes = await sb.auth.getSession();
    currentUser = sessionRes.data && sessionRes.data.session ? sessionRes.data.session.user : null;
    if (!currentUser) {
      currentProfile = null;
      if (window.VHprofile) window.VHprofile.set({ linked: false, username: null, handle: null });
      dispatchAccountChange();
      return { user: null, profile: null };
    }

    await applyPendingProfile(currentUser.id);
    currentProfile = await loadProfile(currentUser.id);
    if (currentProfile) {
      await syncProfileToLocal(currentProfile);
    }
    dispatchAccountChange();
    return { user: currentUser, profile: currentProfile };
  }

  async function signIn(email, password) {
    if (!sb) throw new Error('Supabase er ikke lastet.');
    const res = await sb.auth.signInWithPassword({ email: email, password: password });
    if (res.error) throw res.error;
    await refreshAuth();
    return res.data;
  }

  async function signOut() {
    if (!sb) return;
    await sb.auth.signOut();
    currentUser = null;
    currentProfile = null;
    if (window.VHprofile) window.VHprofile.set({ linked: false, username: null, handle: null });
    dispatchAccountChange();
  }

  async function upsertProfileForUser(userId, patch) {
    if (!sb || !userId) throw new Error('Mangler bruker.');
    const payload = {
      id: userId,
      display_name: String(patch.display_name || '').trim() || 'Bruker',
      handle: normalizeHandle(patch.handle, patch.display_name),
      avatar_url: patch.avatar_url || null,
      role: normalizeRole(patch.role),
      county: patch.county || null,
      mod_role: patch.mod_role || 'user',
      sync_data: patch.sync_data != null ? patch.sync_data : null,
      bookmarks: patch.bookmarks != null ? patch.bookmarks : null,
    };
    const res = await safeUpsertProfile(payload);
    if (res.error) throw res.error;
    currentProfile = await loadProfile(userId);
    if (currentProfile) await syncProfileToLocal(currentProfile);
    dispatchAccountChange();
    return currentProfile;
  }

  async function signUp(options) {
    if (!sb) throw new Error('Supabase er ikke lastet.');
    const email = String(options.email || '').trim();
    const password = String(options.password || '');
    const displayName = String(options.display_name || '').trim();
    const handle = normalizeHandle(options.handle, displayName);
    const role = normalizeRole(options.role);
    const county = options.county || null;
    const avatarUrl = options.avatar_url || null;

    const signUpRes = await sb.auth.signUp({
      email: email,
      password: password,
      options: {
        emailRedirectTo: options.emailRedirectTo || window.location.href,
      },
    });
    if (signUpRes.error) throw signUpRes.error;

    const session = signUpRes.data && signUpRes.data.session ? signUpRes.data.session : null;
    if (session && session.user && session.user.id) {
      await upsertProfileForUser(session.user.id, {
        display_name: displayName,
        handle: handle,
        role: role,
        county: county,
        avatar_url: avatarUrl,
        mod_role: 'user',
      });
    }

    return signUpRes.data;
  }

  async function updateProfile(patch) {
    if (!currentUser) throw new Error('Ikke logget inn.');
    const existing = currentProfile || { mod_role: 'user' };
    const payload = {
      display_name: patch.display_name != null ? patch.display_name : existing.display_name,
      handle: patch.handle != null ? patch.handle : existing.handle,
      role: patch.role != null ? patch.role : existing.role,
      county: patch.county != null ? patch.county : existing.county,
      avatar_url: patch.avatar_url !== undefined ? patch.avatar_url : existing.avatar_url,
      mod_role: existing.mod_role || 'user',
      sync_data: patch.sync_data !== undefined ? patch.sync_data : (existing.sync_data || null),
      bookmarks: patch.bookmarks !== undefined ? patch.bookmarks : (existing.bookmarks || null),
    };
    return upsertProfileForUser(currentUser.id, payload);
  }

  async function syncSettingsToAccount() {
    if (!currentUser) return null;
    const syncData = getLocalSyncSettings();
    return updateProfile({ sync_data: syncData });
  }

  async function syncBookmarksToAccount(bookmarks) {
    if (!currentUser) return null;
    const normalized = normalizeBookmarks(bookmarks != null ? bookmarks : getLocalBookmarks());
    return updateProfile({ bookmarks: normalized });
  }

  async function syncAllLocalToAccount() {
    if (!currentUser) return null;
    const syncData = getLocalSyncSettings();
    const bookmarks = getLocalBookmarks();
    return updateProfile({ sync_data: syncData, bookmarks: bookmarks });
  }

  async function updatePassword(newPassword) {
    if (!sb) throw new Error('Supabase er ikke lastet.');
    const res = await sb.auth.updateUser({ password: String(newPassword || '') });
    if (res.error) throw res.error;
    return res.data;
  }

  if (sb) {
    sb.auth.onAuthStateChange(function () {
      refreshAuth();
    });
  }

  window.VHaccount = {
    SUPABASE_URL: SUPABASE_URL,
    SUPABASE_ANON: SUPABASE_ANON,
    client: sb,
    escHtml: escHtml,
    ROLE_OPTIONS: ROLE_OPTIONS,
    COUNTY_OPTIONS: COUNTY_OPTIONS,
    normalizeRole: normalizeRole,
    displayRole: displayRole,
    normalizeHandle: normalizeHandle,
    processAvatarFile: processAvatarFile,
    blobToDataUrl: blobToDataUrl,
    dataUrlToBlob: dataUrlToBlob,
    getLocalBookmarks: getLocalBookmarks,
    setLocalBookmarks: setLocalBookmarks,
    getLocalSyncSettings: getLocalSyncSettings,
    getCurrentUser: function () { return currentUser; },
    getCurrentProfile: function () { return currentProfile; },
    refreshAuth: refreshAuth,
    signIn: signIn,
    signOut: signOut,
    signUp: signUp,
    updateProfile: updateProfile,
    updatePassword: updatePassword,
    syncSettingsToAccount: syncSettingsToAccount,
    syncBookmarksToAccount: syncBookmarksToAccount,
    syncAllLocalToAccount: syncAllLocalToAccount,
    loadProfile: loadProfile,
  };
}());
