(function () {
  'use strict';

  function initHeaderMenu() {
    const burgerBtn = document.getElementById('burgerBtn');
    const navDropdown = document.getElementById('navDropdown');
    if (!burgerBtn || !navDropdown) return;

    burgerBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      navDropdown.classList.toggle('active');
    });

    document.addEventListener('click', () => {
      navDropdown.classList.remove('active');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeaderMenu);
  } else {
    initHeaderMenu();
  }
}());
