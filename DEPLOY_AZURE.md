# Hosting PlanForge

Two separate pieces:

| Piece | Where | Cost |
|---|---|---|
| Website (`app/static`) | Vercel (Hobby plan) | free |
| Conversion engine (`app/server.py` + models, Docker image) | Azure Web App for Containers, Linux **B1**, in its own resource group | billed to the Azure for Students credit |
| Model weights | Azure Blob Storage (copied from Hugging Face) | a few cents |

The engine is its own resource group (e.g. `planforge-rg`), so nothing else in the subscription is touched.
Model weights never go in the repo, the site or the image: the engine copies them from Hugging Face into Blob when a
new release appears and loads them from there.

## Before the first deploy: publish the ONNX models

The engine image has **no PyTorch**. It runs both models with ONNX Runtime and loads
`checkpoints/model.onnx` (floor-plan model) and `checkpoints/best.onnx` (furniture detector) from Hugging Face.
Run `kaggle/export_onnx.py` once in a Kaggle notebook (secret `HF_TOKEN` with write access) to create them,
and again after every retrain. It also moves the `prod` tag, so the engine picks the new version up on its next start.

## One-time setup

1. **Storage account** (Azure portal -> Storage account, same resource group, LRS). Copy its connection string
   (Security + networking -> Access keys). The engine creates the `models` container itself.
2. **Web App** (Create a resource -> Web App): Publish **Container**, OS **Linux**, plan **B1**, resource group `planforge-rg`.
   Image source: Other container registries -> `ghcr.io/gogul098/planforge-engine:latest`.
3. **Web App -> Settings -> Environment variables** (app settings):

   | Name | Value |
   |---|---|
   | `WEBSITES_PORT` | `8080` |
   | `HF_TOKEN` | Hugging Face read token |
   | `AZURE_STORAGE_CONNECTION_STRING` | from step 1 |
   | `ALLOWED_ORIGINS` | your site, e.g. `https://planforge.vercel.app` (comma separated; default `*` = any site) |
   | `PLANFORGE_MODEL_REV` | optional, Hugging Face tag/branch to serve (default `prod`, falls back to `main`) |
   | `PLANFORGE_JOB_TTL_H` | optional, hours a finished job's files are kept (default `24`) |
   | `PLANFORGE_FLIP_TTA` | optional, `1` = also read the mirrored plan (2x model time, slightly smoother masks; default off) |
   | `PLANFORGE_FURNITURE_ZOOMS` | optional, detector zooms, e.g. `0.75,1.0,1.4` (each zoom costs a full pass; default `1.0`) |
   | `PLANFORGE_PREVIEW_3D` | optional, `1` = 3D renders in preview.png (slow on CPU; default off) |

   Also turn **Always On** off if you want it to sleep when idle (first request then waits for a cold start).
4. **Make the image pullable:** GitHub profile -> Packages -> `planforge-engine` -> Package settings ->
   Change visibility -> Public (it holds code only, no weights).
5. **Automatic deploys:** Web App -> Overview -> *Download publish profile*. In GitHub:
   Settings -> Secrets and variables -> Actions
   - Secret `AZURE_WEBAPP_PUBLISH_PROFILE` = contents of that file
   - Variable `AZURE_WEBAPP_NAME` = the Web App's name

   From now on every push to `main` that touches `app/`, the `Dockerfile` or the workflow builds the image and deploys it
   (`.github/workflows/azure-engine.yml`). Health check: `https://<web-app>.azurewebsites.net/api/health`,
   running build: `/api/version`.
6. **Website:** on vercel.com -> Add New -> Project -> import `gogul098/planforge`, **Root Directory `app/static`**,
   Framework preset *Other*, no build command. Every push to `main` redeploys it.
   The engine URL is set in `app/static/config.js`; `?api=<engine URL>` overrides it for testing.

## Notes
- After an idle period the first conversion waits for the container to start and the models to load from Blob.
- To stop all costs: delete the resource group `planforge-rg`.
- Set a budget alert: Cost Management -> Budgets -> e.g. 5 USD on `planforge-rg`.

## Run the engine locally with Docker
```
docker build -t planforge-engine .
docker run -p 8080:8080 -e HF_TOKEN=<your token> planforge-engine
```
Health check: http://localhost:8080/api/health. Open the website with `?api=http://localhost:8080`.
Without `AZURE_STORAGE_CONNECTION_STRING` the engine downloads the models straight from Hugging Face.

## Optional: read the text with the DeepSeek OCR API

By default the text on a plan (dimensions, room sizes, stair notes) is read on the server with RapidOCR.
To send it to DeepSeek's OCR API instead, add these App Service settings (Configuration → Application settings):

| Setting | Value |
|---|---|
| `PLANFORGE_OCR` | `deepseek` |
| `DEEPSEEK_API_KEY` | your key from platform.deepseek.com (prepaid balance needed) |
| `PLANFORGE_OCR_TIMEOUT` | optional, seconds per request (default 90) |
| `PLANFORGE_OCR_TILE_FROM` | optional, longest side in px above which a sheet is sent as tiles (default 2600) |

Cost: $0.225 per 1,000 pages, one page per image or tile. If the API fails or times out, that image is read
with RapidOCR on the server, so a plan never fails because of the API. The zoomed re-reads around stairs
(UP / DN) always stay local. Note: plan images are sent to DeepSeek's servers.

## Optional: read the text with the free OCR.space API

Get a free key at ocr.space/ocrapi (email only, no card; 25,000 requests a month, 500 a day). Then add these
App Service settings and restart:

| Setting | Value |
|---|---|
| `PLANFORGE_OCR` | `ocrspace` |
| `OCRSPACE_API_KEY` | your free key |
| `PLANFORGE_OCR_ROTATED` | optional, `0` to skip the second read that finds vertical dimensions (halves requests) |
| `PLANFORGE_OCRSPACE_ENGINE` | optional, `1` or `2` (default `2`) |

A normal plan uses 2 requests (upright + turned 90 degrees for vertical dimension text), so the free key covers
about 250 plans a day. Images are kept under the free 1 MB limit (PNG, else JPEG). If a request fails or the
daily limit is reached, that image is read with RapidOCR on the server. The log shows `OCR.space failed (...)`
when that happens. Plan images are sent to OCR.space's servers.
