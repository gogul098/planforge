// Day / night theme: remembered per browser; the viewer listens for "pf-theme" to recolour its grid.
(function () {
  var root = document.documentElement;
  function set(t) {
    root.dataset.theme = t;
    try { localStorage.setItem("pf-theme", t); } catch (e) {}
    window.dispatchEvent(new CustomEvent("pf-theme", { detail: t }));
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("#btnTheme");
    if (b) set(root.dataset.theme === "day" ? "night" : "day");
  });
})();
