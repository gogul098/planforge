// Landing: quiet entrances, the scene counter, and nav ink on the light (sandstone) scene.
(function () {
  var nav = document.querySelector(".nav"), no = document.getElementById("sceneNo");
  var show = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); show.unobserve(e.target); } });
  }, { threshold: 0.2 });
  document.querySelectorAll(".reveal, .model, .plan").forEach(function (el) { show.observe(el); });
  var scenes = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      if (no) no.textContent = e.target.dataset.label;
      nav.classList.toggle("ink", e.target.classList.contains("sandstone"));
    });
  }, { rootMargin: "-45% 0px -45% 0px" });
  document.querySelectorAll(".scene").forEach(function (s) { scenes.observe(s); });
})();
