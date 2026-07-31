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
    try {
      localStorage.setItem(PLUS_KEY, JSON.stringify(next));
    } catch (e) {
      // Storage unavailable or quota exceeded
    }
    document.dispatchEvent(new CustomEvent(PLUS_EVENT, { detail: next }));
    return next;
  }

  function clear() {
    try {
      localStorage.removeItem(PLUS_KEY);
    } catch (e) {
      // Storage unavailable
    }
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

  function trapFocus(element) {
    var focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    var firstFocusable = focusableElements[0];
    var lastFocusable = focusableElements[focusableElements.length - 1];

    function onTrapKeydown(e) {
      if (e.key !== 'Tab') return;
      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable.focus();
        }
      }
    }

    element.addEventListener('keydown', onTrapKeydown);
    return function removeTrap() {
      element.removeEventListener('keydown', onTrapKeydown);
    };
  }

  function showDeactivationModal(options) {
    var backdrop = document.getElementById('plusDeactivateModal');
    if (!backdrop) return;
    backdrop.hidden = false;
    backdrop.setAttribute('aria-hidden', 'false');

    var modalTitle = document.getElementById('plusModalTitle');
    var modalDesc = document.getElementById('plusModalDesc');
    var keepBtn = document.getElementById('plusKeepBtn');
    var keepDataBtn = document.getElementById('plusDeactKeepDataBtn');
    var deleteDataBtn = document.getElementById('plusDeactDeleteDataBtn');

    if (!keepBtn || !keepDataBtn || !deleteDataBtn) return;

    // Store original text for restoration
    var originalTitle = modalTitle ? modalTitle.textContent : '';
    var originalDesc = modalDesc ? modalDesc.textContent : '';
    var originalKeepBtnHTML = keepBtn ? keepBtn.innerHTML : '';

    // Apply custom copy if provided
    if (options && options.title && modalTitle) {
      modalTitle.textContent = options.title;
    }
    if (options && options.description && modalDesc) {
      modalDesc.textContent = options.description;
    }
    if (options && options.keepBtnText && keepBtn) {
      var keepBtnIcon = keepBtn.querySelector('.material-icons');
      var keepBtnIconText = keepBtnIcon ? keepBtnIcon.outerHTML : '';
      keepBtn.innerHTML = keepBtnIconText + (options.keepBtnText || 'Nei, behold Pluss!');
    }

    var previousFocus = document.activeElement;
    keepBtn.focus();
    var removeTrap = trapFocus(backdrop);

    function onBackdropClick(e) {
      if (e.target === backdrop) {
        cleanup();
        if (options && options.onCancel) options.onCancel();
      }
    }

    function onKeydown(e) {
      if (e.key === 'Escape') {
        cleanup();
        if (options && options.onCancel) options.onCancel();
      }
    }

    var cleanup = function () {
      backdrop.hidden = true;
      backdrop.setAttribute('aria-hidden', 'true');
      keepBtn.removeEventListener('click', onKeep);
      keepDataBtn.removeEventListener('click', onKeepData);
      deleteDataBtn.removeEventListener('click', onDeleteData);
      backdrop.removeEventListener('click', onBackdropClick);
      document.removeEventListener('keydown', onKeydown);
      removeTrap();
      // Restore original modal text
      if (modalTitle) modalTitle.textContent = originalTitle;
      if (modalDesc) modalDesc.textContent = originalDesc;
      if (keepBtn) keepBtn.innerHTML = originalKeepBtnHTML;
      if (previousFocus) previousFocus.focus();
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
    document.addEventListener('keydown', onKeydown);
    backdrop.addEventListener('click', onBackdropClick);
  }

  function clearPlusData() {
    try {
      localStorage.removeItem(PLUS_KEY);
    } catch (e) {
      // Storage unavailable
    }
    var data = defaultPlusData();
    document.dispatchEvent(new CustomEvent(PLUS_EVENT, { detail: data }));
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
    trapFocus: trapFocus,
    REQUIRED_COUNTY: REQUIRED_COUNTY,
    PLUS_EVENT: PLUS_EVENT,
    PLUS_KEY: PLUS_KEY,
  };
})();
