# Hosting PlanForge on Azure

Two separate pieces, both in their own resources (nothing existing in the subscription is touched):

| Piece | Azure service | Cost |
|---|---|---|
| Website (app/static) | Static Web Apps, Free plan | free |
| Conversion engine (app/server.py, model) | Container Apps, consumption, scale to zero | free monthly grant (180,000 vCPU-s + 360,000 GiB-s); about 200-300 conversions a month fit |

Model weights never go in the repo, the site or the image. The engine downloads them from Hugging Face on first use with `HF_TOKEN`.

## One-time setup

1. **Service principal for GitHub** (Azure Cloud Shell):
   ```
   az ad sp create-for-rbac --name planforge-github --role contributor \
     --scopes /subscriptions/<SUBSCRIPTION_ID> --sdk-auth
   ```
   Copy the JSON output.
2. **GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret**
   - `AZURE_CREDENTIALS` = the JSON from step 1
   - `HF_TOKEN` = your Hugging Face read token
3. **Engine**: Actions -> "Engine -> Azure Container Apps" -> Run workflow. The last log line prints the engine URL.
4. **Make the image pullable**: GitHub profile -> Packages -> `planforge-engine` -> Package settings -> Change visibility -> Public (it holds code only, no weights).
   Then re-run the engine workflow once.
5. **Website**: Azure portal -> Create resource -> Static Web App -> Plan **Free**, resource group `planforge-rg`, Source **Other**.
   Open it -> Manage deployment token -> copy it into the GitHub secret `AZURE_STATIC_WEB_APPS_API_TOKEN`.
6. **Connect the two**: GitHub -> Settings -> Secrets and variables -> Actions -> Variables -> `PLANFORGE_API` = engine URL from step 3.
   Run "Website -> Azure Static Web Apps". The site's URL is on the Static Web App's overview page.

After this, every push to `main` redeploys whichever piece changed.

## Notes
- The first conversion after an idle period takes ~30-60 s extra (container start + model download).
- To stop all costs: delete the resource group `planforge-rg`.
- Set a budget alert: Cost Management -> Budgets -> e.g. 1 USD on `planforge-rg`.
