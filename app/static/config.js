// The conversion engine (Azure). The website itself is hosted only on Vercel.
// Override for testing: open the site with ?api=https://your-engine-url  (e.g. a local engine on http://localhost:8080)
(function () {
  var ENGINE = "https://planforge-app-bcgzgve2dhced6d8.malaysiawest-01.azurewebsites.net";
  window.PLANFORGE_API = window.PLANFORGE_API || ENGINE;
})();
