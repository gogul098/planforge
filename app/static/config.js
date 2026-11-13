// Where the conversion engine (app/server.py) runs.
// - Served by the engine itself (Azure Web App, local container): same site, leave empty.
// - Served from Vercel or any other static host: the Azure engine below.
// - Override for testing: open the site with ?api=https://your-engine-url
(function () {
  var ENGINE = "https://planforge-app-bcgzgve2dhced6d8.malaysiawest-01.azurewebsites.net";
  var sameSite = /\.azurewebsites\.net$/.test(location.hostname) || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  window.PLANFORGE_API = window.PLANFORGE_API || (sameSite ? "" : ENGINE);
})();
