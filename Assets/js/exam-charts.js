// Charts inside exam task text are declared as
//   <div class='fig-chart-wrap'><canvas data-chart='bar|line' data-labels='a,b,c'
//     data-datasets='Label|v1,v2,v3|#hex;…' data-emphasis='2,6'></canvas></div>
// and rendered with Chart.js (loaded from CDN in the page <head>).
//
// Used by both Tools/exam-simulator.html (student) and Tools/eksamen-admin.html
// (sensor) so graphs are readable everywhere.
(function () {
  'use strict';

  // Chart instances from earlier renders must be destroyed when the panel is
  // re-rendered (task navigation, new detail view), otherwise Chart.js v4
  // keeps ResizeObservers/listeners attached to removed canvases.
  var activeCharts = [];

  window.initTaskCharts = function (container) {
    activeCharts.forEach(function (c) { c.destroy(); });
    activeCharts = [];

    if (!window.Chart) {
      // Chart.js not loaded yet (CDN slow or blocked) – show a note instead of
      // a blank box, and retry once the page has fully loaded (covers restored
      // sessions where a task renders before the deferred script executes).
      container.querySelectorAll('.fig-chart-wrap').forEach(function (wrap) {
        if (!wrap.querySelector('.fig-chart-fallback')) {
          var note = document.createElement('p');
          note.className = 'fig-chart-fallback';
          note.textContent = 'Diagrammet kunne ikke lastes inn.';
          wrap.appendChild(note);
        }
      });
      if (document.readyState !== 'complete') {
        window.addEventListener('load', function () {
          container.querySelectorAll('.fig-chart-wrap').forEach(function (wrap) {
            var note = wrap.querySelector('.fig-chart-fallback');
            if (note) note.remove();
          });
          window.initTaskCharts(container);
        }, { once: true });
      }
      return;
    }

    container.querySelectorAll('canvas[data-chart]').forEach(function (canvas) {
      var labels = (canvas.dataset.labels || '').split(',').map(function (s) { return s.trim(); });
      var datasets = (canvas.dataset.datasets || '').split(';').map(function (part) {
        var pieces = part.split('|');
        var label = pieces[0] || '';
        var values = (pieces[1] || '').split(',').map(function (v) { return parseFloat(v); });
        var color = pieces[2] || '#5865F2';
        return { label: label, data: values, backgroundColor: color, borderColor: color, borderWidth: 2 };
      }).filter(function (d) { return d.data.length; });

      var isLine = canvas.dataset.chart === 'line';

      // For line graphs, let the task optionally emphasize specific x-values
      // (e.g. the measured points (2, 12) and (6, 36) in the sunflower task)
      // with larger points.
      if (isLine) {
        var emphasis = (canvas.dataset.emphasis || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
        datasets.forEach(function (ds) {
          ds.fill = false;
          ds.tension = 0.3;
          ds.pointRadius = labels.map(function (l) {
            return emphasis.indexOf(l) !== -1 ? 7 : 3;
          });
          ds.pointBackgroundColor = labels.map(function (l) {
            return emphasis.indexOf(l) !== -1 ? '#000' : ds.borderColor;
          });
        });
      }

      activeCharts.push(new Chart(canvas, {
        type: isLine ? 'line' : 'bar',
        data: { labels: labels, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 14, font: { family: 'Manrope', size: 12 } } }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(148, 163, 184, 0.25)' },
              ticks: { font: { family: 'Manrope', size: 12 } }
            },
            x: {
              grid: { display: false },
              ticks: { font: { family: 'Manrope', size: 12 } }
            }
          }
        }
      }));
    });
  };
})();
