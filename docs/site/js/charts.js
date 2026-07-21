/* MagicMindNet docs charts — Chart.js with graceful SVG fallback */
(function () {
  const palette = {
    teal: "#0d6b5c",
    tealBright: "#1a9a82",
    sand: "#c4a574",
    ink: "#14201c",
    soft: "#3a4a44",
    grid: "rgba(20, 32, 28, 0.08)",
  };

  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: palette.soft,
          font: { family: "Figtree, sans-serif", size: 12 },
        },
      },
      tooltip: {
        backgroundColor: palette.ink,
        titleFont: { family: "Figtree, sans-serif" },
        bodyFont: { family: "IBM Plex Mono, monospace", size: 12 },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.soft, maxRotation: 45, minRotation: 0 },
        grid: { color: palette.grid },
      },
      y: {
        ticks: { color: palette.soft },
        grid: { color: palette.grid },
        beginAtZero: true,
      },
    },
  };

  async function loadJson(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error("failed to load " + path);
    return res.json();
  }

  function fallbackMessage(canvas, msg) {
    const parent = canvas.parentElement;
    parent.innerHTML =
      '<p style="margin:2rem 0;color:#3a4a44;font-size:0.9rem">' +
      msg +
      " Open <code>docs/site/data/</code> or re-run benchmarks.</p>";
  }

  async function render() {
    if (typeof Chart === "undefined") {
      document.querySelectorAll("canvas").forEach((c) => {
        fallbackMessage(c, "Chart.js CDN unavailable.");
      });
      return;
    }

    Chart.defaults.font.family = "Figtree, sans-serif";
    Chart.defaults.color = palette.soft;

    try {
      const interop = await loadJson("data/interop_benchmark.json");
      const labels = interop.formats.map((f) => f.format);
      const sizes = interop.formats.map((f) => f.size_kib);
      const saveMs = interop.formats.map((f) => f.save_ms);
      const loadMs = interop.formats.map((f) => f.load_ms);

      new Chart(document.getElementById("chart-size"), {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "KiB",
              data: sizes,
              backgroundColor: palette.teal,
              borderRadius: 2,
              maxBarThickness: 28,
            },
          ],
        },
        options: {
          ...baseOptions,
          plugins: {
            ...baseOptions.plugins,
            legend: { display: false },
          },
        },
      });

      new Chart(document.getElementById("chart-io"), {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "save ms",
              data: saveMs,
              backgroundColor: palette.teal,
              borderRadius: 2,
              maxBarThickness: 18,
            },
            {
              label: "load ms",
              data: loadMs,
              backgroundColor: palette.sand,
              borderRadius: 2,
              maxBarThickness: 18,
            },
          ],
        },
        options: baseOptions,
      });
    } catch (err) {
      fallbackMessage(document.getElementById("chart-size"), String(err));
      fallbackMessage(document.getElementById("chart-io"), String(err));
    }

    try {
      const smoke = await loadJson("data/smoke_eval.json");
      const results = smoke.results || [];
      new Chart(document.getElementById("chart-smoke"), {
        type: "bar",
        data: {
          labels: results.map((r) => r.name),
          datasets: [
            {
              label: "ms",
              data: results.map((r) => Number(r.elapsed_ms.toFixed(2))),
              backgroundColor: palette.tealBright,
              borderRadius: 2,
              maxBarThickness: 16,
            },
          ],
        },
        options: {
          ...baseOptions,
          indexAxis: "y",
          plugins: {
            ...baseOptions.plugins,
            legend: { display: false },
          },
        },
      });
    } catch (err) {
      fallbackMessage(document.getElementById("chart-smoke"), String(err));
    }

    new Chart(document.getElementById("chart-parity"), {
      type: "doughnut",
      data: {
        labels: ["Shipped (W1–3)", "Planned (W4+)", "Pre-wave (core)"],
        datasets: [
          {
            data: [12, 4, 18],
            backgroundColor: [palette.teal, palette.sand, palette.ink],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: palette.soft, font: { family: "Figtree, sans-serif" } },
          },
        },
      },
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
