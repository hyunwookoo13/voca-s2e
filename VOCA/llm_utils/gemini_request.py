import os
from mimetypes import guess_type

import cv2
import numpy as np

# ------------------------------------------------------------------
# Vertex AI (ADC) configuration
# ------------------------------------------------------------------
_ADC_CANDIDATE_PATHS = [
    os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
    os.path.expanduser("~/gcloud/application_default_credentials.json"),
    os.path.abspath("gcloud/application_default_credentials.json"),
]
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    for _adc_path in _ADC_CANDIDATE_PATHS:
        if os.path.isfile(_adc_path):
            # Use ADC file automatically if explicit env var is not set.
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _adc_path
            break

TEXT_MODEL = os.getenv("VOCA_TEXT_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
VISION_MODEL = os.getenv("VOCA_VISION_MODEL", os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"))
VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))

try:
    MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
except ValueError:
    MAX_OUTPUT_TOKENS = 1024

DECISION_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Reason": {"type": "STRING"},
        "Angle": {"type": "INTEGER"},
        "Flag": {"type": "BOOLEAN"},
    },
    "required": ["Reason", "Angle", "Flag"],
}

PRIORS_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Supports": {"type": "ARRAY", "items": {"type": "STRING"}},
        "StrongCooccurs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "Gateways": {"type": "ARRAY", "items": {"type": "STRING"}},
        "Lookalikes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["Supports", "StrongCooccurs", "Gateways", "Lookalikes"],
}


_VERTEX_READY = False
_VERTEX_LIBS = None


def _image_to_vertex_bytes(image):
    """
    image: file path(str) or OpenCV np.ndarray(BGR)
    return: (mime_type, bytes)
    """
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        mime_type = mime_type or "image/jpeg"
        with open(image, "rb") as f:
            return mime_type, f.read()

    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("이미지 인코딩 실패")
        return "image/jpeg", buf.tobytes()

    raise TypeError("image must be a file path (str) or numpy.ndarray")


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    # Fallback for SDK variants that expose candidates/parts only.
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        parts = getattr(candidates[0].content, "parts", None) or []
        chunks = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                chunks.append(t)
        return "".join(chunks).strip()
    except Exception:
        return ""


def _read_project_from_gcloud_config():
    active_cfg = "default"
    active_cfg_path = os.path.expanduser("~/.config/gcloud/active_config")
    if os.path.isfile(active_cfg_path):
        try:
            with open(active_cfg_path, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name:
                    active_cfg = name
        except Exception:
            pass

    cfg_path = os.path.expanduser(f"~/.config/gcloud/configurations/config_{active_cfg}")
    if not os.path.isfile(cfg_path):
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("project"):
                    # format: project = xxx
                    _, v = s.split("=", 1)
                    project = v.strip()
                    if project:
                        return project
    except Exception:
        return None
    return None


def _resolve_vertex_project():
    # 1) explicit envs
    for key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        v = os.getenv(key)
        if v:
            return v

    # 2) google.auth.default() (ADC)
    try:
        import google.auth  # lazy import

        _, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if project_id:
            return project_id
    except Exception:
        pass

    # 3) gcloud config fallback
    project = _read_project_from_gcloud_config()
    if project:
        return project

    raise RuntimeError(
        "Vertex project id를 찾을 수 없습니다. "
        "GOOGLE_CLOUD_PROJECT를 설정하거나 gcloud config set project <PROJECT_ID>를 실행하세요."
    )


def _ensure_vertex_ready():
    global _VERTEX_READY, _VERTEX_LIBS

    if _VERTEX_LIBS is None:
        try:
            import vertexai  # lazy import
            from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
        except Exception as e:
            raise RuntimeError(
                "Vertex AI SDK를 찾을 수 없습니다. "
                "pip install google-cloud-aiplatform google-auth 를 설치하세요."
            ) from e
        _VERTEX_LIBS = (vertexai, GenerativeModel, Part, GenerationConfig)

    if not _VERTEX_READY:
        project_id = _resolve_vertex_project()
        _VERTEX_LIBS[0].init(project=project_id, location=VERTEX_LOCATION)
        _VERTEX_READY = True
        print(f"[VertexAI] initialized project={project_id} location={VERTEX_LOCATION}")

    _, GenerativeModel, Part, GenerationConfig = _VERTEX_LIBS
    return GenerativeModel, Part, GenerationConfig


def _make_generation_config(GenerationConfig, response_schema=None):
    if response_schema is None:
        raise ValueError("response_schema is required for strict JSON output.")

    try:
        return GenerationConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
    except Exception as e:
        raise RuntimeError(
            "현재 Vertex AI SDK에서 response_schema 강제 설정을 지원하지 않습니다. "
            "google-cloud-aiplatform을 최신 버전으로 업그레이드하세요."
        ) from e


def text_response(text_prompt, system_prompt=""):
    """
    Vertex AI Gemini text response.
    """
    GenerativeModel, _, GenerationConfig = _ensure_vertex_ready()
    gen_cfg = _make_generation_config(GenerationConfig, response_schema=PRIORS_JSON_SCHEMA)

    model = GenerativeModel(
        model_name=TEXT_MODEL,
        system_instruction=system_prompt or None,
    )
    resp = model.generate_content(
        text_prompt,
        generation_config=gen_cfg,
    )
    return _extract_text(resp)


def vision_response(text_prompt, image_prompt, system_prompt=""):
    """
    Vertex AI Gemini multimodal response.
    image_prompt: image path(str) or OpenCV np.ndarray(BGR)
    """
    GenerativeModel, Part, GenerationConfig = _ensure_vertex_ready()
    gen_cfg = _make_generation_config(GenerationConfig, response_schema=DECISION_JSON_SCHEMA)
    mime_type, image_bytes = _image_to_vertex_bytes(image_prompt)
    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)

    model = GenerativeModel(
        model_name=VISION_MODEL,
        system_instruction=system_prompt or None,
    )
    resp = model.generate_content(
        [text_prompt, image_part],
        generation_config=gen_cfg,
    )
    return _extract_text(resp)
