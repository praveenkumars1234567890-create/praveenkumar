// Draws the charts. Every page still works (tables, tape) if Chart.js can't be loaded.
(function () {
  if (typeof Chart === "undefined") return;
  var colors = {normal: "#1E8E7E", DoS: "#C8322F", Probe: "#D99A00", R2L: "#6B4FBB", U2R: "#223A8C"};
  Chart.defaults.font.family = '"Source Sans 3","Segoe UI",system-ui,sans-serif';
  Chart.defaults.color = "#5B6B7A";

  function data(el) { try { return JSON.parse(el.getAttribute("data-chart")); } catch (e) { return null; } }

  document.querySelectorAll("canvas[data-kind=classes]").forEach(function (el) {
    var d = data(el); if (!d) return;
    var labels = Object.keys(d).filter(function (k) { return d[k] > 0; });
    if (!labels.length) return;
    new Chart(el, {type: "doughnut",
      data: {labels: labels, datasets: [{data: labels.map(function (k) { return d[k]; }),
        backgroundColor: labels.map(function (k) { return colors[k]; }), borderWidth: 2, borderColor: "#fff"}]},
      options: {maintainAspectRatio: false, cutout: "58%", plugins: {legend: {position: "bottom"}}}});
  });

  document.querySelectorAll("canvas[data-kind=scans]").forEach(function (el) {
    var d = data(el); if (!d || !d.length) return;
    new Chart(el, {type: "bar",
      data: {labels: d.map(function (s, i) { return s.name.length > 16 ? s.name.slice(0, 15) + "…" : s.name; }),
        datasets: [{label: "Normal", data: d.map(function (s) { return s.normal; }), backgroundColor: colors.normal},
                   {label: "Attack", data: d.map(function (s) { return s.attack; }), backgroundColor: colors.DoS}]},
      options: {maintainAspectRatio: false, scales: {x: {stacked: true, grid: {display: false}}, y: {stacked: true, beginAtZero: true}},
                plugins: {legend: {position: "bottom"}}}});
  });

  document.querySelectorAll("canvas[data-kind=history]").forEach(function (el) {
    var d = data(el); if (!d || !d.loss) return;
    var ds = [["loss", "Training loss", "#0E4F63"], ["val_loss", "Validation loss", "#D99A00"],
              ["accuracy", "Training accuracy", "#1E8E7E"], ["val_accuracy", "Validation accuracy", "#6B4FBB"]]
      .filter(function (x) { return d[x[0]]; })
      .map(function (x) { return {label: x[1], data: d[x[0]], borderColor: x[2], backgroundColor: x[2], tension: .25, pointRadius: 0,
                                  yAxisID: x[0].indexOf("loss") >= 0 ? "y" : "y2"}; });
    new Chart(el, {type: "line",
      data: {labels: d.loss.map(function (_, i) { return i + 1; }), datasets: ds},
      options: {maintainAspectRatio: false,
        scales: {x: {title: {display: true, text: "Epoch"}}, y: {title: {display: true, text: "Loss"}, beginAtZero: true},
                 y2: {position: "right", min: 0, max: 1, grid: {drawOnChartArea: false}, title: {display: true, text: "Accuracy"}}},
        plugins: {legend: {position: "bottom"}}}});
  });
})();
