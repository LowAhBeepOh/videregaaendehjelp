(function () {
  'use strict';

  var PLUS_KEY = 'vhjelp:plus';
  var PLUS_EVENT = 'vhjelp:plus-changed';
  var REQUIRED_COUNTY = 'Møre og Romsdal';

  function defaultPlusData() {
    return {
      enabled: false,
      activatedAt: null,
      settings: {},
    };
  }

  function get() {
    try {
      var raw = localStorage.getItem(PLUS_KEY);
      if (!raw) return defaultPlusData();
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return defaultPlusData();
      return Object.assign(defaultPlusData(), parsed);
    } catch (e) {
      return defaultPlusData();
    }
  }

  function set(patch) {
    var next = Object.assign({}, get(), patch || {});
    localStorage.setItem(PLUS_KEY, JSON.stringify(next));
    document.dispatchEvent(new CustomEvent(PLUS_EVENT, { detail: next }));
    return next;
  }

  function clear() {
    localStorage.removeItem(PLUS_KEY);
    document.dispatchEvent(new CustomEvent(PLUS_EVENT, { detail: defaultPlusData() }));
  }

  function canActivate() {
    var profile = window.VHprofile ? VHprofile.get() : null;
    if (!profile || !profile.name || profile.name === 'Bruker') return false;
    if (profile.county !== REQUIRED_COUNTY) return false;
    var user = window.VHaccount ? VHaccount.getCurrentUser() : null;
    if (!user) return false;
    var linkedProfile = window.VHaccount ? VHaccount.getCurrentProfile() : null;
    if (!linkedProfile) return false;
    return true;
  }

  function isActive() {
    return get().enabled === true;
  }

  function activate() {
    if (!canActivate()) return false;
    set({ enabled: true, activatedAt: new Date().toISOString() });
    renderPlusBadge();
    return true;
  }

  function deactivate() {
    set({ enabled: false });
    renderPlusBadge();
  }

  function renderPlusBadge() {
    var logos = document.querySelectorAll('.logo');
    var active = isActive();
    logos.forEach(function (logo) {
      var badge = logo.querySelector('.plus-badge');
      if (active) {
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'plus-badge';
          badge.textContent = '+';
          badge.setAttribute('aria-label', 'Videregående Hjelp+');
          logo.appendChild(badge);
        }
        badge.style.display = '';
      } else {
        if (badge) badge.style.display = 'none';
      }
    });
  }

  function showDeactivationModal(options) {
    var backdrop = document.getElementById('plusDeactivateModal');
    if (!backdrop) return;
    backdrop.hidden = false;
    backdrop.setAttribute('aria-hidden', 'false');

    var keepBtn = document.getElementById('plusKeepBtn');
    var keepDataBtn = document.getElementById('plusDeactKeepDataBtn');
    var deleteDataBtn = document.getElementById('plusDeactDeleteDataBtn');

    var cleanup = function () {
      backdrop.hidden = true;
      backdrop.setAttribute('aria-hidden', 'true');
      keepBtn.removeEventListener('click', onKeep);
      keepDataBtn.removeEventListener('click', onKeepData);
      deleteDataBtn.removeEventListener('click', onDeleteData);
    };

    function onKeep() {
      cleanup();
      if (options && options.onCancel) options.onCancel();
    }

    function onKeepData() {
      cleanup();
      if (options && options.onDeactivate) options.onDeactivate(false);
    }

    function onDeleteData() {
      cleanup();
      if (options && options.onDeactivate) options.onDeactivate(true);
    }

    keepBtn.addEventListener('click', onKeep);
    keepDataBtn.addEventListener('click', onKeepData);
    deleteDataBtn.addEventListener('click', onDeleteData);

    // Close on Escape
    function onKeydown(e) {
      if (e.key === 'Escape') {
        cleanup();
        document.removeEventListener('keydown', onKeydown);
        if (options && options.onCancel) options.onCancel();
      }
    }
    document.addEventListener('keydown', onKeydown);

    // Close on backdrop click
    backdrop.addEventListener('click', function (e) {
      if (e.target === backdrop) {
        cleanup();
        if (options && options.onCancel) options.onCancel();
      }
    });
  }

  function clearPlusData() {
    localStorage.removeItem(PLUS_KEY);
  }

  // Run on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderPlusBadge);
  } else {
    renderPlusBadge();
  }

  // Re-render badge on plus changes
  document.addEventListener(PLUS_EVENT, renderPlusBadge);

  window.VHplus = {
    get: get,
    set: set,
    clear: clear,
    canActivate: canActivate,
    isActive: isActive,
    activate: activate,
    deactivate: deactivate,
    renderPlusBadge: renderPlusBadge,
    showDeactivationModal: showDeactivationModal,
    clearPlusData: clearPlusData,
    REQUIRED_COUNTY: REQUIRED_COUNTY,
    PLUS_EVENT: PLUS_EVENT,
    PLUS_KEY: PLUS_KEY,
  };
})();
