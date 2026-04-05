"""FastAPI server — serves UI + /api/evaluate endpoint"""

import traceback
from pathlib import Path

import pydantic
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# P3 imports (safe — already tested)
# ---------------------------------------------------------------------------
from config import get_provider
from schemas import EvaluateRequest, ErrorResponse, validate_ensemble_response

# ---------------------------------------------------------------------------
# P2 imports (wrap in try/except — stubs may not be ready yet)
# ---------------------------------------------------------------------------
try:
    from repo_reader import extract_model_profile
    from pipeline import run_ensemble
    _p2_available = True
    _p2_error = None
except ImportError as e:
    _p2_available = False
    _p2_error = str(e)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="UNICC AI Safety Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve templates relative to this file, not the cwd
_BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the single-page UI."""
    html_path = _BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health_check():
    """Quick readiness probe."""
    return {"status": "ok", "provider": get_provider()}


@app.post("/api/evaluate")
async def evaluate(request: Request):
    """Run the evaluation pipeline and return the ensemble result."""

    # --- Gate: are P2 modules loaded? ---
    if not _p2_available:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="P2_UNAVAILABLE",
                error_message=f"Pipeline modules failed to load: {_p2_error}",
            ).model_dump(),
        )

    # --- Parse & validate request body ---
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code="INVALID_JSON",
                error_message="Request body is not valid JSON.",
            ).model_dump(),
        )

    try:
        req = EvaluateRequest(**body)
    except pydantic.ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code="VALIDATION_ERROR",
                error_message=str(exc),
            ).model_dump(),
        )

    # --- Build model_profile dict ---
    try:
        if req.github_url:
            model_profile_dict = extract_model_profile(req.github_url)
        elif req.model_profile:
            model_profile_dict = req.model_profile.model_dump()
        else:
            # Should never happen (validator catches it), but just in case
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error_code="NO_INPUT",
                    error_message="Provide a GitHub URL or model profile.",
                ).model_dump(),
            )
    except Exception:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="PROFILE_EXTRACTION_FAILED",
                error_message="Failed to extract model profile from the repository.",
            ).model_dump(),
        )

    # --- Run the ensemble pipeline ---
    try:
        raw_result = run_ensemble(model_profile_dict)
    except Exception:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="PIPELINE_FAILED",
                error_message="The evaluation pipeline crashed. Check server logs.",
            ).model_dump(),
        )

    # --- Validate & return ---
    result = validate_ensemble_response(raw_result)

    if result.get("error"):
        return JSONResponse(status_code=500, content=result)

    return JSONResponse(status_code=200, content=result)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
